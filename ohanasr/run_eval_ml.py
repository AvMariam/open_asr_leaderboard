"""Open ASR Leaderboard multilingual evaluation (FLEURS / MCV) for the Armenian-only OhanASR models.

Metric-AI/OhanASR-v0.1 and Metric-AI/OhanASR-Streaming-v0.1 are fine-tuned from omniASR-LLM-1B-v2 and released in
the fairseq2 format (not the Transformers AutoModel format). Each repository ships its own loading helper
(`ohanasr.py` / `ohanasr_streaming.py`), which this script imports from the downloaded snapshot. Dataset handling
and scoring follow the original (fairseq2) omniASR `run_eval_ml.py`.
"""
import argparse
import os
import sys
import time

import evaluate
import numpy as np
import torch
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from normalizer import data_utils
from normalizer.eval_utils import normalize_compound_pairs

wer_metric = evaluate.load("wer")

# omnilingual_asr expects NLLB-style language codes; the OhanASR models are Armenian-only.
LANG_CODE = "hye_Armn"
SAMPLING_RATE = 16000
MAX_AUDIO_SEC = 40  # Offline model's max audio length


def load_ohanasr(model_id):
    """Return a function mapping a list of 16 kHz waveforms to their transcripts."""
    folder = snapshot_download(model_id, token=True)
    sys.path.insert(0, folder)

    if "Streaming" in model_id:
        from ohanasr_streaming import StreamingTranscriber, load_model

        model, tokenizer = load_model(folder)
        # Recommended settings from the model card (used for the reported scores).
        transcriber = StreamingTranscriber(
            model, tokenizer, beam=2, max_context_sec=15.0, keep_sec=5.0, lang=LANG_CODE
        )
        return lambda waveforms: [transcriber.transcribe(w) for w in waveforms]

    from ohanasr import load_model

    pipeline = load_model(folder)
    return lambda waveforms: pipeline.transcribe(
        [{"waveform": w[: MAX_AUDIO_SEC * SAMPLING_RATE], "sample_rate": SAMPLING_RATE} for w in waveforms],
        lang=[LANG_CODE] * len(waveforms),
        batch_size=len(waveforms),
    )


def main(args):
    CONFIG_NAME = args.config_name
    SPLIT_NAME = args.split
    LANGUAGE = args.language or CONFIG_NAME.split("_", 1)[1]

    # The helpers place the model on the current CUDA device (or CPU when none is available).
    if args.device >= 0:
        torch.cuda.set_device(args.device)
    transcribe = load_ohanasr(args.model_id)

    def benchmark(batch):
        minibatch_size = len(batch["audio"])
        waveforms = [np.asarray(audio["array"], dtype=np.float32) for audio in batch["audio"]]
        batch["audio_length_s"] = [len(w) / SAMPLING_RATE for w in waveforms]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(batch, minibatch_size)

        # START TIMING
        start_time = time.time()
        transcriptions = transcribe(waveforms)
        # END TIMING
        runtime = time.time() - start_time

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = transcriptions  # raw; normalization applied at scoring time
        batch["references"] = [
            data_utils.get_text({k: batch[k][i] for k in batch if k != "audio"})
            for i in range(minibatch_size)
        ]  # raw; normalization applied at scoring time
        return batch

    # Load dataset
    print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")
    dataset = load_dataset(
        args.dataset,
        CONFIG_NAME,
        split=SPLIT_NAME,
        streaming=args.streaming,
        token=True,
    )

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    # Resample audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # Filter out empty references
    dataset = dataset.filter(lambda x: data_utils.is_target_text_in_range(data_utils.get_text(x)))

    # Warmup
    if args.warmup_steps:
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))
        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Run evaluation
    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset,
        CONFIG_NAME,
        SPLIT_NAME,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
        audio_filepaths=all_results["audio_filepath"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    norm_refs = [data_utils.ml_normalizer(r, lang=LANGUAGE) for r in all_results["references"]]
    norm_preds = [data_utils.ml_normalizer(p, lang=LANGUAGE) for p in all_results["predictions"]]
    wer_refs, wer_preds = normalize_compound_pairs(norm_refs, norm_preds)
    wer = round(100 * wer_metric.compute(references=wer_refs, predictions=wer_preds), 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print(f"Dataset: {args.dataset}")
    print(f"Language: {LANGUAGE}")
    print(f"Config: {CONFIG_NAME}")
    print(f"Model: {args.model_id}")
    print(f"WER: {wer}%")
    print(f"RTFx: {rtfx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier on Hugging Face (e.g., 'Metric-AI/OhanASR-v0.1')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Metric-AI/open-asr-leaderboard-multilingual-datasets",
        help="Dataset path. Default is 'Metric-AI/open-asr-leaderboard-multilingual-datasets'",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Config name in format <dataset>_<lang> (e.g., fleurs_hy, mcv_hy).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., hy). If not provided, will be extracted from config_name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. Default is 'test'.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset lazily over the network instead of downloading it in full before the evaluation. Off by default for reproducible benchmark timings.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()

    main(args)
