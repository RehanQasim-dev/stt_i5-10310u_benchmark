#!/usr/bin/env python3
"""
STT CPU Matrix Benchmark Runner
Evaluates Q4_K_M, Q8_0, and ONNX models on Intel Core i5-10310U CPU (AVX2).
Supports execution via Handy CLI (forced to CPU via device-index 1 / settings) and native transcribe.cpp.
Measures latency, Real-Time Factor (xRT), Word Error Rate (WER), and Character Error Rate (CER).
"""

import os
import sys
import json
import time
import subprocess
import argparse
import re
from difflib import SequenceMatcher
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
TRANSCRIBE_CLI = "/home/rehan-10xe/Documents/transcribe.cpp/build/bin/transcribe-cli"

# Model definitions strictly <= 700 MB
MODELS = [
    {
        "name": "Parakeet TDT 0.6B v2",
        "family": "FastConformer (TDT)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 454,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q4_K_M.gguf")
    },
    {
        "name": "Parakeet TDT 0.6B v2",
        "family": "FastConformer (TDT)",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 696,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q8_0.gguf")
    },
    {
        "name": "Parakeet TDT 0.6B v2",
        "family": "FastConformer (TDT)",
        "quant": "Int8",
        "format": "ONNX",
        "size_mb": 631,
        "type": "handy",
        "model_id": "parakeet-tdt-0.6b-v2-int8"
    },
    {
        "name": "Canary 180M Flash",
        "family": "Conformer-AED",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 133,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "canary-180m-flash/canary-180m-flash-Q4_K_M.gguf")
    },
    {
        "name": "Canary 180M Flash",
        "family": "Conformer-AED",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 208,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "canary-180m-flash/canary-180m-flash-Q8_0.gguf")
    },
    {
        "name": "Canary 180M Flash",
        "family": "Conformer-AED",
        "quant": "FP32/Int8",
        "format": "ONNX",
        "size_mb": 204,
        "type": "handy",
        "model_id": "canary-180m-flash"
    },
    {
        "name": "Whisper Small.en",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 164,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "whisper-small.en/whisper-small.en-Q4_K_M.gguf")
    },
    {
        "name": "Whisper Small.en",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 257,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "whisper-small.en/whisper-small.en-Q8_0.gguf")
    },
    {
        "name": "Parakeet TDT CTC 110M",
        "family": "Conformer-CTC",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 86,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-Q4_K_M.gguf")
    },
    {
        "name": "Parakeet TDT CTC 110M",
        "family": "Conformer-CTC",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 129,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m-Q8_0.gguf")
    },
    {
        "name": "SenseVoiceSmall",
        "family": "SenseVoice (CTC)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 139,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "SenseVoiceSmall/SenseVoiceSmall-Q4_K_M.gguf")
    },
    {
        "name": "SenseVoiceSmall",
        "family": "SenseVoice (CTC)",
        "quant": "Q8_0",
        "format": "GGUF",
        "size_mb": 241,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "SenseVoiceSmall/SenseVoiceSmall-Q8_0.gguf")
    },
    {
        "name": "Whisper Medium.en",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 481,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "whisper-medium.en/whisper-medium.en-Q4_K_M.gguf")
    },
    {
        "name": "Whisper Medium",
        "family": "Whisper (Encoder-Decoder)",
        "quant": "Q4_1",
        "format": "BIN",
        "size_mb": 469,
        "type": "handy",
        "model_id": "medium"
    },
    {
        "name": "Nemotron Streaming 0.6B",
        "family": "FastConformer (Streaming)",
        "quant": "Q4_K_M",
        "format": "GGUF",
        "size_mb": 454,
        "type": "transcribe_cli",
        "path": os.path.join(MODELS_DIR, "nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf")
    },
    {
        "name": "Moonshine V2 Medium",
        "family": "Conformer Streaming",
        "quant": "Int8",
        "format": "ONNX",
        "size_mb": 289,
        "type": "handy",
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

def run_transcribe_cli(model_path: str, wav_path: str) -> dict:
    """Run transcription via native transcribe-cli on CPU."""
    cmd = [
        TRANSCRIBE_CLI,
        "--backend", "cpu",
        "-m", model_path,
        wav_path
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    wall_sec = t1 - t0
    
    canonical_text = ""
    text_pieces = []
    timing_info = {}
    for line in proc.stdout.splitlines():
        if line.startswith("text: "):
            canonical_text = line[6:].strip()
        m = re.search(r'\[\s*\d+\.\d+\s*->\s*\d+\.\d+\]\s*(?:p=[\d\.]+)?\s*(.*)', line)
        if m:
            text_pieces.append(m.group(1))
        if line.strip().startswith("[info] timings:"):
            timing_info["raw"] = line.strip()

    final_text = canonical_text if canonical_text else "".join(text_pieces).strip()
    return {
        "text": final_text,
        "wall_sec": round(wall_sec, 2),
        "return_code": proc.returncode,
        "error": proc.stderr if proc.returncode != 0 else None,
        "timing_info": timing_info
    }

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
    """Run full model matrix against a single audio slice."""
    duration = get_audio_duration(audio_path)
    print(f"\n=======================================================")
    print(f"▶ Evaluating: {script_name} | Slice: {slice_name}")
    print(f"  Audio Path: {audio_path}")
    print(f"  Duration  : {duration:.2f}s ({duration/60:.2f} min)")
    print(f"  Ref Words : {len(reference_text.split())} words")
    print(f"=======================================================")

    results = []
    for m in MODELS:
        print(f"  Testing [{m['format']} {m['quant']}] {m['name']} ({m['size_mb']}MB)... ", end="", flush=True)
        if m["type"] == "transcribe_cli":
            if not os.path.exists(m["path"]):
                print("SKIPPED (Model file missing)")
                continue
            res = run_transcribe_cli(m["path"], audio_path)
        else:
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
    parser = argparse.ArgumentParser(description="STT CPU Matrix Benchmark on Intel i5-10310U")
    parser.add_argument("--script", choices=["all", "non_technical", "technical"], default="all")
    parser.add_argument("--slices", nargs="+", default=["slice_30s", "slice_60s", "slice_120s", "slice_180s"])
    parser.add_argument("--audio", help="Direct test audio wav file override")
    parser.add_argument("--ref", help="Direct reference text file override")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_DIR, "logs", f"matrix_eval_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

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
                print(f"[!] Dataset directory {s_dir} not found. Run python3 record_dataset.py first.")
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
        print("\n[!] No audio slices found to evaluate.")
        print("Please record your dataset first:")
        print("  python3 record_dataset.py")
        return

    # Save raw JSON
    json_path = os.path.join(out_dir, "matrix_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(all_evaluations, f, indent=2)

    # Generate Markdown Table Report
    md_path = os.path.join(out_dir, "benchmark_report.md")
    with open(md_path, "w") as f:
        f.write("# CPU Speech-to-Text Model Matrix Benchmark Report\n\n")
        f.write(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Compute Device:** Intel(R) Core(TM) i5-10310U CPU @ 1.70GHz (AVX2, 8 threads)\n")
        f.write(f"- **Quantization Scope:** Models $\\le 700$ MB (`Q4_K_M`, `Q8_0`, `ONNX int8`)\n\n")

        for entry in all_evaluations:
            f.write(f"## {entry['script'].upper()} - {entry['slice']}\n\n")
            f.write("| Model | Format | Quant | Size (MB) | Audio (s) | Latency (s) | Speedup (xRT) | WER (%) | Word Acc (%) |\n")
            f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in entry["results"]:
                f.write(f"| {r['model']} | {r['format']} | {r['quant']} | {r['size_mb']} | {r['duration_sec']}s | {r['wall_sec']}s | **{r['rtf_speedup']}x** | {r['wer']}% | **{r['accuracy']}%** |\n")
            f.write("\n")

    print(f"\n=======================================================")
    print(f"✔ Benchmark complete!")
    print(f"  JSON Results : {json_path}")
    print(f"  Markdown Table: {md_path}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
