#!/usr/bin/env python3
"""
STT CPU Matrix Benchmark Runner (Handy-Native with Real-Time Logging)
Evaluates speech-to-text models on CPU across Q8_0, Int8, and BIN formats exclusively via Handy.
Forces CPU execution via Handy CLI (--device-index 1 / persisted settings).
Measures latency, Real-Time Factor (RTF), Word Error Rate (WER), and accuracy.

Logs are written immediately after every single model inference completes:
- Individual model & quantization folder: logs/<run>/<Model>_<Format>_<Quant>/
  - <dataset>_<slice>_transcript.txt (detected text)
  - <dataset>_<slice>_reference.txt (ground-truth reference)
  - <dataset>_<slice>_metrics.json (precision and performance numbers)
  - results.json (cumulative results for this model)
- Master summary: BENCHMARK_SUMMARY.md and benchmark_results.json updated in real-time.
- Symlink: logs/latest points to the current/latest evaluation directory.
"""

import os
import sys
import json
import time
import subprocess
import argparse
import shutil
import re
import platform
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
LOGS_ROOT = os.path.join(PROJECT_DIR, "logs")

def get_cpu_info():
    """Dynamically query host CPU information for portable benchmark reporting."""
    cpu_name = "Unknown CPU"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        cpu_name = platform.processor() or "Generic CPU"
    threads = os.cpu_count() or 1
    return f"{cpu_name} ({threads} threads)"

# Evaluated models via Handy execution engine (All 20 available model & quant configurations)
MODELS = [
    # Parakeet TDT 0.6B v2 (Flagship English)
    {
        "name": "Parakeet TDT 0.6B v2",
        "family": "FastConformer (TDT)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 453,
        "model_id": "handy-computer/parakeet-tdt-0.6b-v2-gguf/parakeet-tdt-0.6b-v2-Q4_K_M.gguf"
    },
    {
        "name": "Parakeet TDT 0.6B v2",
        "family": "FastConformer (TDT)",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 695,
        "model_id": "handy-computer/parakeet-tdt-0.6b-v2-gguf/parakeet-tdt-0.6b-v2-Q8_0.gguf"
    },
    {
        "name": "Parakeet TDT 0.6B v2",
        "family": "FastConformer (TDT)",
        "quant": "Int8",
        "format": "ONNX",
        "size_mb": 451,
        "model_id": "parakeet-tdt-0.6b-v2"
    },
    # Parakeet TDT 0.6B v3 (Multilingual)
    {
        "name": "Parakeet TDT 0.6B v3",
        "family": "FastConformer (TDT)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 462,
        "model_id": "handy-computer/parakeet-tdt-0.6b-v3-gguf/parakeet-tdt-0.6b-v3-Q4_K_M.gguf"
    },
    {
        "name": "Parakeet TDT 0.6B v3",
        "family": "FastConformer (TDT)",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 705,
        "model_id": "handy-computer/parakeet-tdt-0.6b-v3-gguf/parakeet-tdt-0.6b-v3-Q8_0.gguf"
    },
    {
        "name": "Parakeet TDT 0.6B v3",
        "family": "FastConformer (TDT)",
        "quant": "Int8",
        "format": "ONNX",
        "size_mb": 456,
        "model_id": "parakeet-tdt-0.6b-v3"
    },
    # Canary 180M Flash
    {
        "name": "Canary 180M Flash",
        "family": "Conformer-AED",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 132,
        "model_id": "handy-computer/canary-180m-flash-gguf/canary-180m-flash-Q4_K_M.gguf"
    },
    {
        "name": "Canary 180M Flash",
        "family": "Conformer-AED",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 208,
        "model_id": "handy-computer/canary-180m-flash-gguf/canary-180m-flash-Q8_0.gguf"
    },
    {
        "name": "Canary 180M Flash",
        "family": "Conformer-AED",
        "quant": "FP32/Int8",
        "format": "ONNX",
        "size_mb": 146,
        "model_id": "canary-180m-flash"
    },
    # Parakeet TDT CTC 110M
    {
        "name": "Parakeet TDT CTC 110M",
        "family": "Conformer-CTC",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 85,
        "model_id": "handy-computer/parakeet-tdt_ctc-110m-gguf/parakeet-tdt_ctc-110m-Q4_K_M.gguf"
    },
    {
        "name": "Parakeet TDT CTC 110M",
        "family": "Conformer-CTC",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 129,
        "model_id": "handy-computer/parakeet-tdt_ctc-110m-gguf/parakeet-tdt_ctc-110m-Q8_0.gguf"
    },
    # SenseVoice Small
    {
        "name": "SenseVoice Small",
        "family": "SenseVoice (CTC)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 138,
        "model_id": "handy-computer/SenseVoiceSmall-gguf/SenseVoiceSmall-Q4_K_M.gguf"
    },
    {
        "name": "SenseVoice Small",
        "family": "SenseVoice (CTC)",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 240,
        "model_id": "handy-computer/SenseVoiceSmall-gguf/SenseVoiceSmall-Q8_0.gguf"
    },
    # Whisper Small.en
    {
        "name": "Whisper Small.en",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 163,
        "model_id": "handy-computer/whisper-small.en-gguf/whisper-small.en-Q4_K_M.gguf"
    },
    {
        "name": "Whisper Small.en",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 257,
        "model_id": "handy-computer/whisper-small.en-gguf/whisper-small.en-Q8_0.gguf"
    },
    # Whisper Medium
    {
        "name": "Whisper Medium.en",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 480,
        "model_id": "handy-computer/whisper-medium.en-gguf/whisper-medium.en-Q4_K_M.gguf"
    },
    {
        "name": "Whisper Medium",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q4_1",
        "format": "BIN",
        "size_mb": 469,
        "model_id": "medium"
    },
    # Whisper Base.en
    {
        "name": "Whisper Base.en",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "FP16/FP32",
        "format": "BIN",
        "size_mb": 141,
        "model_id": "ggml-base.en"
    },
    # Nemotron Streaming 0.6B
    {
        "name": "Nemotron Speech Streaming 0.6B",
        "family": "FastConformer Streaming",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 453,
        "model_id": "handy-computer/nemotron-speech-streaming-en-0.6b-gguf/nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf"
    },
    # Moonshine V2 Medium
    {
        "name": "Moonshine V2 Medium",
        "family": "Conformer Streaming",
        "quant": "Int8",
        "format": "ONNX",
        "size_mb": 192,
        "model_id": "moonshine-medium-streaming-en"
    }
]

def normalize_text(text: str) -> str:
    """Normalize text for consistent WER scoring."""
    t = text.lower()
    t = re.sub(r'[\.,\?!:;"\'\(\)\[\]\{\}\-_]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def calculate_wer(reference: str, hypothesis: str) -> dict:
    """
    Calculate Word Error Rate (WER) and error breakdown (substitutions,
    deletions, insertions) using Levenshtein dynamic programming.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    if not ref_words:
        return {
            "wer_pct": 0.0,
            "accuracy_pct": 100.0,
            "ref_words": 0,
            "hyp_words": len(hyp_words),
            "errors": len(hyp_words),
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_words)
        }

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost   # substitution
            )

    errors = d[len(ref_words)][len(hyp_words)]
    wer = round((errors / len(ref_words)) * 100.0, 2)
    accuracy = round(max(0.0, 100.0 - wer), 2)

    # Backtrack alignment to separate substitutions, deletions, and insertions
    i = len(ref_words)
    j = len(hyp_words)
    subs = 0
    dels = 0
    ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == (d[i - 1][j - 1] + (0 if ref_words[i - 1] == hyp_words[j - 1] else 1)):
            if ref_words[i - 1] != hyp_words[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            break

    return {
        "wer_pct": wer,
        "accuracy_pct": accuracy,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "errors": errors,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins
    }

def get_audio_duration(wav_path: str) -> float:
    """Get duration in seconds using ffprobe."""
    try:
        res = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            text=True
        ).strip()
        return float(res)
    except Exception:
        return 0.0

def sync_handy_symlinks():
    """Automatically sync all models in project models/ to Handy store for seamless zero-copy detection."""
    handy_dir = os.path.expanduser("~/.local/share/com.pais.handy/models")
    if not (shutil.which("handy") or os.path.isdir(handy_dir)):
        return

    os.makedirs(handy_dir, exist_ok=True)
    # 1. Symlink top-level model folders and files
    if os.path.exists(MODELS_DIR):
        for entry in os.listdir(MODELS_DIR):
            src = os.path.join(MODELS_DIR, entry)
            dst = os.path.join(handy_dir, entry)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except Exception:
                    pass

        # 2. Symlink all individual .gguf and .bin files from subdirectories
        for root, _, files in os.walk(MODELS_DIR):
            for f in files:
                if f == "tokenizer.bin":
                    continue
                if f.endswith(".gguf") or f.endswith(".bin"):
                    src = os.path.join(root, f)
                    dst = os.path.join(handy_dir, f)
                    if not os.path.exists(dst):
                        try:
                            os.symlink(src, dst)
                        except Exception:
                            pass

        # 3. Root whisper medium alias
        nested_bin = os.path.join(MODELS_DIR, "whisper-medium", "whisper-medium-q4_1.bin")
        root_bin = os.path.join(MODELS_DIR, "whisper-medium-q4_1.bin")
        if os.path.exists(nested_bin) and not os.path.exists(root_bin):
            os.symlink(nested_bin, root_bin)
        elif os.path.exists(root_bin) and not os.path.exists(nested_bin):
            os.makedirs(os.path.dirname(nested_bin), exist_ok=True)
            os.symlink(root_bin, nested_bin)

def check_handy_model_available(model_id: str) -> bool:
    """Check if Handy model is available, linking it if found in project models directory."""
    handy_dir = os.path.expanduser("~/.local/share/com.pais.handy/models")
    if os.path.exists(os.path.join(handy_dir, model_id)):
        return True

    # Extract base filename if model_id is a HuggingFace hub path
    base_file = os.path.basename(model_id)
    if os.path.exists(os.path.join(handy_dir, base_file)):
        return True

    candidates = [
        os.path.join(MODELS_DIR, model_id),
        os.path.join(MODELS_DIR, f"{model_id}-int8"),
        os.path.join(MODELS_DIR, "whisper-medium", "whisper-medium-q4_1.bin") if model_id == "medium" else None,
        os.path.join(MODELS_DIR, "whisper-medium-q4_1.bin") if model_id == "medium" else None,
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                os.makedirs(handy_dir, exist_ok=True)
                dest = os.path.join(handy_dir, model_id)
                if not os.path.exists(dest):
                    os.symlink(os.path.abspath(c), dest)
                return True
            except Exception:
                return True
    return False

def run_handy(model_id: str, wav_path: str) -> dict:
    """Run transcription via handy CLI on CPU (forced to CPU via device-index 1)."""
    cmd = [
        "handy",
        "-f", wav_path,
        "--model", model_id,
        "--device-index", "1",
        "--json"
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    wall_sec = t1 - t0

    transcript = ""
    json_time = None
    if proc.returncode == 0:
        match = re.search(r'\{.*\}', proc.stdout, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                transcript = data.get("text", "")
                if "best_ms" in data:
                    json_time = data["best_ms"] / 1000.0
            except Exception:
                transcript = proc.stdout.strip()
        else:
            transcript = proc.stdout.strip()
    return {
        "text": transcript,
        "wall_sec": round(json_time if json_time else wall_sec, 2),
        "return_code": proc.returncode,
        "error": proc.stderr if proc.returncode != 0 else None
    }

def save_model_inference_log(
    base_log_dir: str,
    model_info: dict,
    dataset_name: str,
    slice_name: str,
    audio_path: str,
    duration_sec: float,
    wall_sec: float,
    rtf_speedup: float,
    metrics: dict,
    reference_text: str,
    detected_text: str
) -> dict:
    """
    Write individual model logs immediately after inference finishes.
    Creates a dedicated folder for each model and quantization scheme.
    Writes:
      1. <dataset>_<slice>_transcript.txt (detected text)
      2. <dataset>_<slice>_reference.txt (ground-truth reference text)
      3. <dataset>_<slice>_metrics.json (performance and precision numbers)
      4. results.json (cumulative list of all slice results for this model)
    """
    clean_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', model_info["name"])
    clean_fmt = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', model_info["format"])
    clean_qnt = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', model_info["quant"])
    model_dir_name = f"{clean_name}_{clean_fmt}_{clean_qnt}"
    model_dir = os.path.join(base_log_dir, model_dir_name)
    os.makedirs(model_dir, exist_ok=True)

    prefix = f"{dataset_name}_{slice_name}"

    # 1. Write detected transcript text file
    transcript_file = os.path.join(model_dir, f"{prefix}_transcript.txt")
    with open(transcript_file, "w") as f:
        f.write(detected_text + "\n")

    # 2. Write ground truth reference text file
    reference_file = os.path.join(model_dir, f"{prefix}_reference.txt")
    with open(reference_file, "w") as f:
        f.write(reference_text + "\n")

    # 3. Write individual slice JSON metrics
    slice_record = {
        "model": model_info["name"],
        "family": model_info["family"],
        "quant": model_info["quant"],
        "format": model_info["format"],
        "size_mb": model_info["size_mb"],
        "model_id": model_info.get("model_id"),
        "dataset": dataset_name,
        "slice": slice_name,
        "audio_file": os.path.abspath(audio_path),
        "duration_sec": round(duration_sec, 2),
        "performance": {
            "wall_sec": round(wall_sec, 2),
            "rtf_speedup": round(rtf_speedup, 2),
            "latency_per_sec_audio": round(wall_sec / duration_sec, 4) if duration_sec > 0 else 0.0
        },
        "precision": {
            "wer_pct": metrics["wer_pct"],
            "accuracy_pct": metrics["accuracy_pct"],
            "ref_words": metrics["ref_words"],
            "hyp_words": metrics["hyp_words"],
            "errors": metrics["errors"],
            "substitutions": metrics["substitutions"],
            "deletions": metrics["deletions"],
            "insertions": metrics["insertions"]
        },
        "transcript_path": os.path.abspath(transcript_file),
        "reference_path": os.path.abspath(reference_file)
    }

    metrics_file = os.path.join(model_dir, f"{prefix}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(slice_record, f, indent=2)

    # 4. Update cumulative results.json inside this model's folder
    cumulative_file = os.path.join(model_dir, "results.json")
    model_history = []
    if os.path.exists(cumulative_file):
        try:
            with open(cumulative_file, "r") as f:
                model_history = json.load(f)
        except Exception:
            model_history = []
    model_history = [item for item in model_history if not (item.get("dataset") == dataset_name and item.get("slice") == slice_name)]
    model_history.append(slice_record)
    with open(cumulative_file, "w") as f:
        json.dump(model_history, f, indent=2)

    return slice_record

def update_master_summary(base_log_dir: str, cpu_desc: str, all_evaluations: list):
    """
    Refresh master summary files immediately after each inference finishes
    so partial runs are never lost.
    """
    # 1. Update master benchmark_results.json
    json_path = os.path.join(base_log_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "last_updated": datetime.now().isoformat(),
            "cpu": cpu_desc,
            "engine": "Handy CLI (CPU AVX2 SIMD)",
            "evaluations": all_evaluations
        }, f, indent=2)

    # 2. Update master BENCHMARK_SUMMARY.md
    md_path = os.path.join(base_log_dir, "BENCHMARK_SUMMARY.md")
    with open(md_path, "w") as f:
        f.write("# Speech-to-Text Benchmark Results (Intel CPU)\n\n")
        f.write(f"- **CPU:** {cpu_desc}\n")
        f.write(f"- **Execution Engine:** Handy CLI (CPU AVX2 SIMD)\n")
        f.write(f"- **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for ev in all_evaluations:
            f.write(f"### Dataset: {ev['script']} | Slice: {ev['slice']}\n\n")
            f.write("| Model | Format / Quant | Size | Latency (s) | Speedup (xRT) | WER (%) | Accuracy (%) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in ev["results"]:
                f.write(f"| **{r['model']}** | {r['format']} {r['quant']} | {r['size_mb']} MB | {r['wall_sec']}s | {r['rtf_speedup']}x | {r['wer']}% | {r['accuracy']}% |\n")
            f.write("\n")

def evaluate_audio_file(
    audio_path: str,
    reference_text: str,
    script_name: str,
    slice_name: str,
    base_log_dir: str,
    cpu_desc: str,
    all_evaluations: list,
    model_filters: list = None
) -> list:
    """Run Handy model matrix against a single audio slice, saving logs after each model."""
    duration = get_audio_duration(audio_path)
    print(f"\n=======================================================")
    print(f">> Evaluating: {script_name} | Slice: {slice_name}")
    print(f"   Audio Path: {audio_path}")
    print(f"   Duration  : {duration:.2f}s ({duration/60:.2f} min)")
    print(f"   Ref Words : {len(reference_text.split())} words")
    print(f"=======================================================")

    current_slice_results = []
    # Create or find entry in all_evaluations
    ev_entry = None
    for ev in all_evaluations:
        if ev["script"] == script_name and ev["slice"] == slice_name:
            ev_entry = ev
            break
    if not ev_entry:
        ev_entry = {"script": script_name, "slice": slice_name, "results": current_slice_results}
        all_evaluations.append(ev_entry)
    else:
        current_slice_results = ev_entry["results"]

    for m in MODELS:
        if model_filters:
            search_blob = f"{m['name']} {m['family']} {m['model_id']} {m['quant']} {m['format']}".lower()
            if not any(f.lower() in search_blob for f in model_filters):
                continue

        print(f"  Testing [{m['format']} {m['quant']}] {m['name']} ({m['size_mb']}MB)... ", end="", flush=True)
        if not shutil.which("handy"):
            print("SKIPPED (handy binary not located)")
            continue
        if not check_handy_model_available(m["model_id"]):
            print("SKIPPED (Model not installed in Handy)")
            continue

        res = run_handy(m["model_id"], audio_path)

        if res["return_code"] != 0:
            print(f"FAILED (Code {res['return_code']})")
            continue

        hyp_text = res["text"]
        wall_sec = res["wall_sec"]
        rtf = round(duration / wall_sec, 2) if wall_sec > 0 else 0.0
        metrics = calculate_wer(reference_text, hyp_text)

        # WRITE LOGS IMMEDIATELY AFTER THIS MODEL INFERENCE
        slice_record = save_model_inference_log(
            base_log_dir=base_log_dir,
            model_info=m,
            dataset_name=script_name,
            slice_name=slice_name,
            audio_path=audio_path,
            duration_sec=duration,
            wall_sec=wall_sec,
            rtf_speedup=rtf,
            metrics=metrics,
            reference_text=reference_text,
            detected_text=hyp_text
        )

        model_summary_entry = {
            "model": m["name"],
            "family": m["family"],
            "quant": m["quant"],
            "format": m["format"],
            "size_mb": m["size_mb"],
            "duration_sec": round(duration, 2),
            "wall_sec": wall_sec,
            "rtf_speedup": rtf,
            "wer": metrics["wer_pct"],
            "accuracy": metrics["accuracy_pct"],
            "errors": metrics["errors"],
            "ref_words": metrics["ref_words"],
            "hyp_words": metrics["hyp_words"],
            "transcript": hyp_text
        }
        current_slice_results.append(model_summary_entry)

        # Update master summary files immediately
        update_master_summary(base_log_dir, cpu_desc, all_evaluations)

        print(f"DONE in {wall_sec}s ({rtf}x RT) | WER: {metrics['wer_pct']}% (Acc: {metrics['accuracy_pct']}%) [Logged]")

    return current_slice_results

def main():
    parser = argparse.ArgumentParser(description="STT CPU Matrix Benchmark Suite (Handy-Only, Real-Time Logging)")
    parser.add_argument("--script", choices=["all", "non_technical", "technical"], default="all")
    parser.add_argument("--slices", nargs="+", default=["slice_30s", "slice_60s", "slice_120s", "slice_180s"])
    parser.add_argument("--models", nargs="+", help="Filter models by substring (e.g. --models canary whisper parakeet)")
    parser.add_argument("--audio", help="Direct test audio wav file override")
    parser.add_argument("--ref", help="Direct reference text file override")
    args = parser.parse_args()

    if not shutil.which("handy"):
        print("[ERROR] 'handy' executable was not found in PATH.")
        print("Please install Handy first: https://github.com/cjpais/handy/releases")
        sys.exit(1)

    # Sync local model directory symlinks into Handy data directory
    sync_handy_symlinks()

    cpu_desc = get_cpu_info()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(LOGS_ROOT, f"matrix_eval_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Point logs/latest to this evaluation run directory
    latest_symlink = os.path.join(LOGS_ROOT, "latest")
    try:
        if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
            os.remove(latest_symlink)
        os.symlink(os.path.abspath(out_dir), latest_symlink)
    except Exception:
        pass

    print(f"=======================================================")
    print(f" STT CPU Benchmark Runner (Handy-Only)")
    print(f" Host CPU       : {cpu_desc}")
    print(f" Engine         : Handy CLI ({shutil.which('handy')})")
    print(f" Log Directory  : {out_dir}")
    print(f" Latest Symlink : {latest_symlink}")
    if args.models:
        print(f" Filter Models  : {args.models}")
    print(f"=======================================================")

    all_evaluations = []

    if args.audio and args.ref:
        with open(args.ref, "r") as f:
            ref_text = f.read().strip()
        evaluate_audio_file(
            args.audio, ref_text, "custom_run", os.path.basename(args.audio),
            out_dir, cpu_desc, all_evaluations, model_filters=args.models
        )
    else:
        scripts = ["non_technical", "technical"] if args.script == "all" else [args.script]
        for s in scripts:
            s_dir = os.path.join(DATASET_DIR, s)
            if not os.path.exists(s_dir):
                print(f"[!] Dataset directory {s_dir} not found. Run 'python3 record_dataset.py' first.")
                continue
            for sl in args.slices:
                wav_path = os.path.join(s_dir, f"{sl}.wav")
                txt_path = os.path.join(s_dir, f"{sl}.txt")
                if not os.path.exists(wav_path) or not os.path.exists(txt_path):
                    print(f"[-] Missing slice: {wav_path}")
                    continue
                with open(txt_path, "r") as f:
                    ref_text = f.read().strip()
                evaluate_audio_file(
                    wav_path, ref_text, s, sl,
                    out_dir, cpu_desc, all_evaluations, model_filters=args.models
                )

    if not all_evaluations:
        print("\nNo benchmarks executed. Please record the audio dataset first:")
        print("    python3 record_dataset.py")
        sys.exit(0)

    md_path = os.path.join(out_dir, "BENCHMARK_SUMMARY.md")
    json_path = os.path.join(out_dir, "benchmark_results.json")

    print(f"\n=======================================================")
    print(f">> Benchmark complete!")
    print(f"   Summary Report: {md_path}")
    print(f"   Raw JSON Data : {json_path}")
    print(f"   Per-Model Logs: {out_dir}/<Model>_<Format>_<Quant>/")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
