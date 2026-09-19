#!/usr/bin/env python3
"""
Handy Hardware & Domain-Specific Model Benchmarker
Evaluates speech-to-text models on technical domain terminology (llama.cpp, GPU architectures, GGUF),
records each attempt into an isolated log directory with audio, golden reference, and per-model transcripts.
"""

import subprocess
import json
import os
import sys
import time
import signal
import re
from datetime import datetime
from difflib import SequenceMatcher

# Technical Golden Reference:
# Stress-tests GPU microarchitecture terms, llama.cpp kernel execution, memory hierarchies, and acronyms.
DEFAULT_GOLDEN_REFERENCE = (
    "We are profiling llama.cpp kernel execution to diagnose microarchitectural bottlenecks across different "
    "GPU architectures. When evaluating fused FlashAttention and quantized GEMM kernels, warp divergence and shared "
    "memory bank conflicts can degrade compute throughput. Does the register file pressure per Streaming "
    "Multiprocessor limit theoretical occupancy during KV cache attention tiling? Let's analyze whether uncoalesced "
    "global memory transactions or L2 cache thrashing cause pipeline stalls on the memory controller, and measure "
    "the Roofline arithmetic intensity across systolic tensor cores and SIMT execution units."
)

def get_system_hardware():
    """Detect CPU and GPU for environment-aware benchmarking."""
    info = {"cpu": "Unknown CPU", "gpu": "Unknown GPU"}
    try:
        cpu = subprocess.check_output("lscpu | grep 'Model name'", shell=True, text=True)
        info["cpu"] = cpu.split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        gpu = subprocess.check_output("lspci | grep -E -i 'vga|3d|display'", shell=True, text=True)
        info["gpu"] = gpu.split(":", 2)[-1].strip()
    except Exception:
        pass
    return info

def get_installed_models():
    """Fetch currently downloaded Handy models."""
    try:
        proc = subprocess.run(
            ["handy", "--json", "--list-models"],
            capture_output=True,
            text=True,
            check=True
        )
        match = re.search(r'\[\s*\{.*\}\s*\]', proc.stdout, re.DOTALL)
        if not match:
            return []
        models = json.loads(match.group(0))
        return [m for m in models if m.get("is_downloaded")]
    except Exception as e:
        print(f"Error checking models: {e}")
        return []

def wait_for_enter(prompt=""):
    """Wait for Enter keypress, automatically healing terminal TTY state if corrupted."""
    print(prompt, end="", flush=True)
    os.system("stty sane 2>/dev/null")
    try:
        sys.stdin.readline()
    except Exception:
        input()

def record_audio(output_wav, golden_ref):
    """Record 16 kHz mono WAV directly from default microphone."""
    # Ensure terminal canonical mode is sane
    os.system("stty sane 2>/dev/null")

    print("\n" + "=" * 88)
    print("[TARGET] TECHNICAL GOLDEN REFERENCE (Read this aloud clearly at your natural pace):")
    print("=" * 88)
    print(f"\n\"{golden_ref}\"\n")
    print("=" * 88)
    wait_for_enter("Press [ENTER] to start recording...")

    cmd = [
        "ffmpeg", "-y",
        "-nostdin",
        "-loglevel", "quiet",
        "-f", "pulse",
        "-i", "default",
        "-ar", "16000",
        "-ac", "1",
        output_wav
    ]

    # Explicitly redirect stdin to DEVNULL so ffmpeg cannot hijack keyboard input
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    start_time = time.time()
    print("\n[REC] RECORDING IN PROGRESS... (Speak into your microphone now)")
    
    try:
        wait_for_enter("-> Press [ENTER] to STOP recording.")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        os.system("stty sane 2>/dev/null")

    duration = time.time() - start_time
    print(f" Recorded {duration:.2f}s of audio -> saved to '{os.path.basename(output_wav)}'.")
    return duration

def benchmark_model(model_id, wav_file):
    """Run Handy in headless batch mode for a model."""
    cmd = ["handy", "-f", wav_file, "--model", model_id, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    stdout_lines = proc.stdout.strip().splitlines()
    json_line = ""
    for line in reversed(stdout_lines):
        if line.strip().startswith("{") and line.strip().endswith("}"):
            json_line = line.strip()
            break

    if not json_line:
        return {"error": "Failed to parse JSON result", "raw": proc.stderr or proc.stdout}

    try:
        return json.loads(json_line)
    except Exception as e:
        return {"error": f"JSON parse error: {e}", "raw": json_line}

def tokenize_words(text):
    """Extract lowercase words stripped of punctuation for pure speech recognition WER."""
    clean = re.sub(r'[^\w\s]', '', text.lower())
    return clean.split()

def compute_wer(ref_words, hyp_words):
    """Compute Word Error Rate (WER) via Levenshtein dynamic programming."""
    r_len, h_len = len(ref_words), len(hyp_words)
    if r_len == 0:
        return 0, 0.0, 100.0

    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    edits = dp[r_len][h_len]
    wer = (edits / r_len) * 100.0
    accuracy = max(0.0, 100.0 - wer)
    return edits, wer, accuracy

def sanitize_filename(name):
    """Sanitize string for filename use."""
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)

def main():
    os.system("stty sane 2>/dev/null")
    hw = get_system_hardware()
    print("=" * 88)
    print("Handy Speech-to-Text Technical Benchmark Tool")
    print(f"CPU: {hw['cpu']}")
    print(f"GPU: {hw['gpu']}")
    print("=" * 88)

    # 1. Discover models
    models = get_installed_models()
    if not models:
        print("No downloaded models found in Handy.")
        sys.exit(1)

    print(f"\nInstalled candidate models ready for testing:")
    for i, m in enumerate(models, 1):
        print(f"  [{i}] {m['name']} ({m['id']}) - {m.get('size_mb', '?')} MB")

    # 2. Setup Per-Attempt Log Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    attempt_dir = os.path.join(script_dir, "logs", f"attempt_{timestamp}")
    transcripts_dir = os.path.join(attempt_dir, "transcripts")
    os.makedirs(transcripts_dir, exist_ok=True)

    print(f"\n[DIR] Experiment folder created: {attempt_dir}")

    golden_ref = DEFAULT_GOLDEN_REFERENCE
    
    # Save reference transcript
    ref_file = os.path.join(attempt_dir, "reference.txt")
    with open(ref_file, "w", encoding="utf-8") as f:
        f.write(golden_ref + "\n")

    # 3. Record Audio
    wav_file = os.path.join(attempt_dir, "audio.wav")
    audio_duration = record_audio(wav_file, golden_ref)

    # 4. Run Benchmarks
    print("\n" + "=" * 88)
    print("BENCHMARKING MODELS ON RECORDED AUDIO...")
    print("=" * 88)

    ref_words = tokenize_words(golden_ref)
    results = []

    for m in models:
        mid = m["id"]
        mname = m["name"]
        print(f"Evaluating: {mname:<28} ... ", end="", flush=True)

        res = benchmark_model(mid, wav_file)
        if "error" in res:
            print(f"[FAILED: {res.get('error')}]")
            continue

        hyp_text = res.get("text", "").strip()
        best_ms = res.get("best_ms", 0)
        rtf = res.get("rtf", 0.0)
        bound_backend = res.get("bound_backend", "unknown")

        # Save individual detected transcript to attempt folder
        safe_name = sanitize_filename(f"{mname}_{mid}")
        transcript_path = os.path.join(transcripts_dir, f"{safe_name}.txt")
        with open(transcript_path, "w", encoding="utf-8") as tf:
            tf.write(f"Model: {mname}\n")
            tf.write(f"Model ID: {mid}\n")
            tf.write(f"Backend: {bound_backend}\n")
            tf.write(f"Latency: {best_ms} ms\n")
            tf.write(f"Speed (RTF): {rtf:.2f}x\n")
            tf.write("-" * 50 + "\n")
            tf.write(hyp_text + "\n")

        # Accuracy & WER metrics
        hyp_words = tokenize_words(hyp_text)
        edits, wer, word_acc = compute_wer(ref_words, hyp_words)
        format_match = SequenceMatcher(None, golden_ref, hyp_text).ratio() * 100.0

        # Suitability score (harmonic mean between word accuracy and speed)
        speed_score = min(100.0, rtf * 10.0)
        suitability = (2 * word_acc * speed_score) / (word_acc + speed_score) if (word_acc + speed_score) > 0 else 0

        print(f"Done in {best_ms}ms ({rtf:.1f}x RT) | Word Acc: {word_acc:.1f}%")

        results.append({
            "name": mname,
            "id": mid,
            "backend": bound_backend,
            "text": hyp_text,
            "best_ms": best_ms,
            "rtf": rtf,
            "word_acc": word_acc,
            "wer": wer,
            "format_match": format_match,
            "suitability": suitability,
            "file": os.path.basename(transcript_path)
        })

    if not results:
        print("No models completed successfully.")
        return

    # 5. Build Report & Summary Table
    report_lines = []
    report_lines.append("=" * 105)
    report_lines.append(f"BENCHMARK REPORT - ATTEMPT {timestamp}")
    report_lines.append(f"Hardware: {hw['cpu']} | GPU: {hw['gpu']}")
    report_lines.append(f"Audio Duration: {audio_duration:.2f}s | Location: {wav_file}")
    report_lines.append("=" * 105)
    report_lines.append(f"{'Model':<24} | {'Backend':<7} | {'Latency':<9} | {'Speed (RTF)':<11} | {'Word Acc':<9} | {'Punct/Fmt':<9} | {'Suitability'}")
    report_lines.append("-" * 105)

    sorted_results = sorted(results, key=lambda x: x["suitability"], reverse=True)
    for r in sorted_results:
        report_lines.append(
            f"{r['name']:<24} | "
            f"{r['backend']:<7} | "
            f"{r['best_ms']:>5} ms  | "
            f"{r['rtf']:>5.1f}x RT   | "
            f"{r['word_acc']:>6.1f}%  | "
            f"{r['format_match']:>6.1f}%   | "
            f"* {r['suitability']:>4.1f}/100"
        )
    report_lines.append("=" * 105)

    report_lines.append("\nTRANSCRIPT COMPARISON:")
    report_lines.append("-" * 105)
    report_lines.append(f"[GOLDEN REFERENCE]:\n  \"{golden_ref}\"\n")
    for r in sorted_results:
        report_lines.append(f"[{r['name']} - {r['backend']}]:")
        report_lines.append(f"  \"{r['text']}\"")
        report_lines.append(f"  Word Accuracy: {r['word_acc']:.1f}% | WER: {r['wer']:.1f}%\n")

    report_content = "\n".join(report_lines)

    # Save summary report & json
    with open(os.path.join(attempt_dir, "report.txt"), "w", encoding="utf-8") as rf:
        rf.write(report_content)

    with open(os.path.join(attempt_dir, "results.json"), "w", encoding="utf-8") as jf:
        json.dump({
            "timestamp": timestamp,
            "hardware": hw,
            "audio_duration_seconds": audio_duration,
            "golden_reference": golden_ref,
            "results": sorted_results
        }, jf, indent=2)

    # Print Report to Terminal
    print("\n" + report_content)
    print(f"\n[OK] All artifacts preserved in: {attempt_dir}")
    print(f"   ├── audio.wav              (your voice sample)")
    print(f"   ├── reference.txt          (golden technical text)")
    print(f"   ├── report.txt             (complete comparative report)")
    print(f"   ├── results.json           (machine-readable metrics)")
    print(f"   └── transcripts/           (individual outputs per model)")

if __name__ == "__main__":
    main()
