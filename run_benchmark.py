#!/usr/bin/env python3
"""
STT CPU Matrix Benchmark Runner (Handy-Native)
Evaluates speech-to-text models on CPU across Q8_0, Int8, and BIN formats exclusively via Handy.
Forces CPU execution via Handy CLI (--device-index 1 / persisted settings).
Measures latency, Real-Time Factor (RTF), Word Error Rate (WER), and accuracy.
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

# Evaluated models via Handy execution engine
MODELS = [
    {
        "name": "Parakeet TDT 0.6B v2",
        "family": "FastConformer (TDT)",
        "quant": "Int8",
        "format": "ONNX",
        "size_mb": 451,
        "model_id": "parakeet-tdt-0.6b-v2"
    },
    {
        "name": "Parakeet TDT 0.6B v3",
        "family": "FastConformer (TDT)",
        "quant": "Int8",
        "format": "ONNX",
        "size_mb": 456,
        "model_id": "parakeet-tdt-0.6b-v3"
    },
    {
        "name": "Canary 180M Flash",
        "family": "Conformer-AED",
        "quant": "FP32/Int8",
        "format": "ONNX",
        "size_mb": 146,
        "model_id": "canary-180m-flash"
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
        "name": "Parakeet TDT CTC 110M",
        "family": "Conformer-CTC",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 129,
        "model_id": "handy-computer/parakeet-tdt_ctc-110m-gguf/parakeet-tdt_ctc-110m-Q8_0.gguf"
    },
    {
        "name": "SenseVoice Small",
        "family": "SenseVoice (CTC)",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 240,
        "model_id": "handy-computer/SenseVoiceSmall-gguf/SenseVoiceSmall-Q8_0.gguf"
    },
    {
        "name": "Whisper Small.en",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 257,
        "model_id": "handy-computer/whisper-small.en-gguf/whisper-small.en-Q8_0.gguf"
    },
    {
        "name": "Whisper Medium",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q4_1",
        "format": "BIN",
        "size_mb": 469,
        "model_id": "medium"
    },
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

def calculate_wer(reference: str, hypothesis: str):
    """Calculate Word Error Rate (WER) using Levenshtein distance."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    if not ref_words:
        return 0.0, 100.0, 0, 0, len(hyp_words)

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost   # substitution
            )

    errors = d[len(ref_words)][len(hyp_words)]
    wer = (errors / len(ref_words)) * 100.0
    accuracy = max(0.0, 100.0 - wer)
    return round(wer, 2), round(accuracy, 2), len(ref_words), len(hyp_words), errors

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

def evaluate_audio_file(audio_path: str, reference_text: str, script_name: str, slice_name: str):
    """Run Handy model matrix against a single audio slice."""
    duration = get_audio_duration(audio_path)
    print(f"\n=======================================================")
    print(f">> Evaluating: {script_name} | Slice: {slice_name}")
    print(f"   Audio Path: {audio_path}")
    print(f"   Duration  : {duration:.2f}s ({duration/60:.2f} min)")
    print(f"   Ref Words : {len(reference_text.split())} words")
    print(f"=======================================================")

    results = []
    for m in MODELS:
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
        wer, acc, ref_w, hyp_w, errs = calculate_wer(reference_text, hyp_text)

        print(f"DONE in {wall_sec}s ({rtf}x RT) | WER: {wer}% (Acc: {acc}%)")
        results.append({
            "model": m["name"],
            "family": m["family"],
            "quant": m["quant"],
            "format": m["format"],
            "size_mb": m["size_mb"],
            "duration_sec": round(duration, 2),
            "wall_sec": wall_sec,
            "rtf_speedup": rtf,
            "wer": wer,
            "accuracy": acc,
            "errors": errs,
            "ref_words": ref_w,
            "hyp_words": hyp_w,
            "transcript": hyp_text
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="STT CPU Matrix Benchmark Suite (Handy-Only)")
    parser.add_argument("--script", choices=["all", "non_technical", "technical"], default="all")
    parser.add_argument("--slices", nargs="+", default=["slice_30s", "slice_60s", "slice_120s", "slice_180s"])
    parser.add_argument("--audio", help="Direct test audio wav file override")
    parser.add_argument("--ref", help="Direct reference text file override")
    args = parser.parse_args()

    if not shutil.which("handy"):
        print("[ERROR] 'handy' executable was not found in PATH.")
        print("Please install Handy first: https://github.com/cjpais/handy/releases")
        sys.exit(1)

    cpu_desc = get_cpu_info()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_DIR, "logs", f"matrix_eval_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"=======================================================")
    print(f" STT CPU Benchmark Runner (Handy-Only)")
    print(f" Host CPU       : {cpu_desc}")
    print(f" Engine         : Handy CLI ({shutil.which('handy')})")
    print(f" Log Directory  : {out_dir}")
    print(f"=======================================================")

    all_evaluations = []

    if args.audio and args.ref:
        with open(args.ref, "r") as f:
            ref_text = f.read().strip()
        res = evaluate_audio_file(args.audio, ref_text, "custom_run", os.path.basename(args.audio))
        all_evaluations.append({"script": "custom", "slice": os.path.basename(args.audio), "results": res})
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
                res = evaluate_audio_file(wav_path, ref_text, s, sl)
                all_evaluations.append({"script": s, "slice": sl, "results": res})

    if not all_evaluations:
        print("\nNo benchmarks executed. Please record the audio dataset first:")
        print("    python3 record_dataset.py")
        sys.exit(0)

    # Save complete JSON results
    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "cpu": cpu_desc,
            "evaluations": all_evaluations
        }, f, indent=2)

    # Generate Markdown Summary Report
    md_path = os.path.join(out_dir, "BENCHMARK_SUMMARY.md")
    with open(md_path, "w") as f:
        f.write(f"# Speech-to-Text Benchmark Results (Intel CPU)\n\n")
        f.write(f"- **CPU:** {cpu_desc}\n")
        f.write(f"- **Execution Engine:** Handy CLI (CPU AVX2 SIMD)\n")
        f.write(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for ev in all_evaluations:
            f.write(f"### Dataset: {ev['script']} | Slice: {ev['slice']}\n\n")
            f.write("| Model | Format / Quant | Size | Latency (s) | Speedup (xRT) | WER (%) | Accuracy (%) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in ev["results"]:
                f.write(f"| **{r['model']}** | {r['format']} {r['quant']} | {r['size_mb']} MB | {r['wall_sec']}s | {r['rtf_speedup']}x | {r['wer']}% | {r['accuracy']}% |\n")
            f.write("\n")

    print(f"\n=======================================================")
    print(f">> Benchmark complete!")
    print(f"   Summary Report: {md_path}")
    print(f"   Raw JSON Data : {json_path}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
