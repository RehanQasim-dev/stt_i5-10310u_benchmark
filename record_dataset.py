#!/usr/bin/env python3
"""
Interactive Dataset Recorder & Deterministic Slicer
Records Master Audio in 4 calibrated blocks and automatically generates
perfectly aligned slices (30s, 60s, 120s, 180s) with 100% exact ground-truth text.
"""

import os
import sys
import subprocess
import signal
import time

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

SCRIPTS = {
    "non_technical": {
        "title": "Script A: Non-Technical (Everyday English & Narrative)",
        "blocks": [
            {
                "id": "block1",
                "label": "Block 1 (~30s, ~70 words)",
                "text": (
                    "It was early on a Tuesday morning, September 15th, when Sarah decided to pack her bags for an "
                    "impromptu road trip across the Pacific Northwest. She had been staring at spreadsheets and "
                    "calendar invites for forty-two consecutive days, and the idea of quiet mountain roads felt like an "
                    "absolute necessity. She double-checked her tire pressure, packed two thermoses of dark roast coffee, "
                    "and loaded a playlist of acoustic songs."
                )
            },
            {
                "id": "block2",
                "label": "Block 2 (~30s, ~70 words)",
                "text": (
                    "As she drove through the Columbia River Gorge, the morning fog began to lift, revealing towering "
                    "pine trees and sheer basalt cliffs along the highway. She stopped at a small roadside diner where the "
                    "owner, an elderly man named Arthur, recommended the blueberry pancakes with homemade maple syrup. "
                    "'You won't find a better breakfast within fifty miles,' he insisted with a warm grin."
                )
            },
            {
                "id": "block3",
                "label": "Block 3 (~60s, ~140 words)",
                "text": (
                    "While waiting for the bill, which came out to precisely seventeen dollars and eighty-five cents, she "
                    "jotted down a few thoughts in her travel journal. Why do we wait until burnout before allowing ourselves "
                    "to pause? Wouldn't life be significantly more fulfilling if spontaneous journeys were a weekly habit rather "
                    "than an annual escape? The waitress smiled as she refilled her coffee cup, asking whether she was headed "
                    "toward the snowy mountain peaks or the rugged ocean coast. Sarah paused for a moment, realizing she genuinely "
                    "had no fixed itinerary, and that freedom felt exhilarating. Outside, seagulls circled above the river, "
                    "and the crisp autumn air smelled faintly of cedar and rain."
                )
            },
            {
                "id": "block4",
                "label": "Block 4 (~60s, ~140 words)",
                "text": (
                    "Leaving the cafe, the afternoon sun broke through the clouds, warming the steering wheel. She decided to "
                    "take the scenic backroads instead of the busy interstate highway, letting curiosity guide her toward the coast "
                    "without checking digital navigation once. By late afternoon, she arrived at a secluded viewpoint overlooking "
                    "the Pacific Ocean, where waves crashed violently against jagged sea stacks. She pulled over to watch the "
                    "sunset paint the horizon in shades of vibrant amber and deep violet. In that quiet moment, watching the tide "
                    "roll in, she understood that stepping away from daily routines wasn't an act of avoidance, but an essential reset."
                )
            }
        ]
    },
    "technical": {
        "title": "Script B: Deep Technical (GPU Microarchitecture & llama.cpp)",
        "blocks": [
            {
                "id": "block1",
                "label": "Block 1 (~30s, ~70 words)",
                "text": (
                    "We are profiling llama.cpp kernel execution to diagnose microarchitectural bottlenecks across diverse GPU "
                    "architectures. When evaluating fused FlashAttention and quantized GEMM kernels, warp divergence and shared "
                    "memory bank conflicts severely degrade compute throughput. Does the register file pressure per Streaming "
                    "Multiprocessor limit theoretical occupancy during KV cache attention tiling? Let's analyze whether uncoalesced "
                    "global memory transactions or L2 cache thrashing cause pipeline stalls on the memory controller."
                )
            },
            {
                "id": "block2",
                "label": "Block 2 (~30s, ~70 words)",
                "text": (
                    "Next, we measure the Roofline arithmetic intensity across systolic tensor cores and SIMT execution units. "
                    "During the prompt prefill phase, memory bandwidth saturation throttles large matrix-matrix contractions "
                    "unless intermediate activations remain resident in L1 data cache. In the auto-regressive token generation loop, "
                    "decoding collapses into memory-bound matrix-vector operations, where memory bus latency dominates over peak compute "
                    "floating-point capabilities."
                )
            },
            {
                "id": "block3",
                "label": "Block 3 (~60s, ~140 words)",
                "text": (
                    "When offloading transformer layers to the Vulkan or Metal backend, does unified memory eliminate PCIe serialization "
                    "bottlenecks, or does host-device synchronization introduce pipeline bubbles into the command queue? To investigate "
                    "this, we instrumented GPU hardware performance counters, measuring instruction issue rates, thread block dispatch "
                    "schedules, and branch predictor accuracy. As batch sizes scale up, shared memory bank conflicts multiply across "
                    "concurrent warps, causing warp schedulers to stall while waiting for operand collector registers to resolve dependencies. "
                    "Furthermore, when the KV cache footprint exceeds the L2 cache capacity, continuous cache line evictions force "
                    "high-latency DRAM round-trips that degrade real-time token streaming throughput."
                )
            },
            {
                "id": "block4",
                "label": "Block 4 (~60s, ~140 words)",
                "text": (
                    "By quantizing model weights from full FP16 precision down to Q4_K_M and Q8_0 block representations, we drastically "
                    "compress the KV cache footprint and minimize DRAM traffic across memory channels. However, sub-byte quantization "
                    "introduces dequantization overhead inside thread execution units before fused multiply-accumulate operations can execute. "
                    "We must carefully balance quantization block sizes against tensor core precision to ensure numerical perplexity does not "
                    "degrade during long-context generation. Finally, compiler-level loop unrolling and explicit register reuse patterns allow "
                    "custom compute shaders to bypass local memory spilling, unlocking maximum sustainable bandwidth on constrained client-grade hardware."
                )
            }
        ]
    }
}

def wait_for_enter(prompt=""):
    print(prompt, end="", flush=True)
    os.system("stty sane 2>/dev/null")
    try:
        sys.stdin.readline()
    except Exception:
        input()

def record_block(output_wav, label, text):
    os.system("stty sane 2>/dev/null")
    print("\n" + "=" * 90)
    print(f"📖 {label.upper()}")
    print("=" * 90)
    print(f"\n\"{text}\"\n")
    print("=" * 90)
    wait_for_enter("Press [ENTER] to start recording this block...")

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

    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    t0 = time.time()
    print("\n🔴 RECORDING... Speak now at your natural pace.")
    
    try:
        wait_for_enter("👉 Press [ENTER] to STOP recording when finished reading.")
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

    duration = time.time() - t0
    print(f"✅ Saved block audio: {duration:.1f}s -> {os.path.basename(output_wav)}")
    return duration

def concatenate_wavs(wav_list, output_wav):
    """Losslessly concatenate 16 kHz WAVs using ffmpeg concat demuxer."""
    concat_txt = output_wav + ".concat.txt"
    with open(concat_txt, "w") as f:
        for w in wav_list:
            f.write(f"file '{os.path.abspath(w)}'\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-nostdin",
        "-loglevel", "quiet",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        output_wav
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(concat_txt):
        os.remove(concat_txt)

def assemble_slices(script_key, out_dir):
    """Combine recorded blocks into cumulative 30s, 60s, 120s, 180s slices with reference texts."""
    cfg = SCRIPTS[script_key]
    blocks = cfg["blocks"]
    
    slices_def = [
        ("slice_30s",  [0]),
        ("slice_60s",  [0, 1]),
        ("slice_120s", [0, 1, 2]),
        ("slice_180s", [0, 1, 2, 3])
    ]

    print("\n" + "=" * 90)
    print(f"📦 ASSEMBLING CUMULATIVE DATASET SLICES FOR: {cfg['title']}")
    print("=" * 90)

    for slice_name, block_indices in slices_def:
        slice_wav = os.path.join(out_dir, f"{slice_name}.wav")
        slice_txt = os.path.join(out_dir, f"{slice_name}.txt")

        # Concatenate audio
        wavs_to_cat = [os.path.join(out_dir, f"{blocks[i]['id']}.wav") for i in block_indices]
        concatenate_wavs(wavs_to_cat, slice_wav)

        # Concatenate reference text
        combined_text = " ".join(blocks[i]["text"] for i in block_indices)
        with open(slice_txt, "w", encoding="utf-8") as f:
            f.write(combined_text + "\n")

        # Measure duration
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", slice_wav],
            text=True
        ).strip()
        dur = float(probe)
        wc = len(combined_text.split())
        print(f"  • {slice_name:<11} : {dur:>5.1f}s | {wc:>3} words | WAV: {os.path.basename(slice_wav)} | TXT: {os.path.basename(slice_txt)}")

    print("=" * 90)
    print("✅ All slices created with 100% mathematically synchronized ground truth!")

def run_script_recording(script_key):
    cfg = SCRIPTS[script_key]
    out_dir = os.path.join(DATASET_DIR, script_key)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"🎙️ RECORDING WORKFLOW: {cfg['title']}")
    print(f"Folder: {out_dir}")
    print("=" * 90)

    for i, b in enumerate(cfg["blocks"], 1):
        block_wav = os.path.join(out_dir, f"{b['id']}.wav")
        block_txt = os.path.join(out_dir, f"{b['id']}.txt")
        
        # Save block text
        with open(block_txt, "w", encoding="utf-8") as f:
            f.write(b["text"] + "\n")

        if os.path.exists(block_wav):
            choice = input(f"\n[{i}/4] Existing recording for '{b['label']}' found. Re-record? [y/N]: ").strip().lower()
            if choice not in ['y', 'yes']:
                print(f"Skipping {b['label']}, using existing recording.")
                continue

        print(f"\n--- STEP {i} OF 4 ---")
        record_block(block_wav, b["label"], b["text"])

    assemble_slices(script_key, out_dir)

def main():
    os.system("stty sane 2>/dev/null")
    print("=" * 90)
    print("SPEECH-TO-TEXT MASTER DATASET BUILDER (Zero-Model-Bias Cumulative Slicing)")
    print("=" * 90)
    print("Select an option:")
    print("  1. Record Script A: Non-Technical (Everyday English & Narrative)")
    print("  2. Record Script B: Deep Technical (GPU Microarchitecture & llama.cpp)")
    print("  3. Re-assemble slices from existing recorded blocks")
    print("  4. Exit")
    
    choice = input("\nEnter choice [1-4]: ").strip()
    if choice == "1":
        run_script_recording("non_technical")
    elif choice == "2":
        run_script_recording("technical")
    elif choice == "3":
        for k in ["non_technical", "technical"]:
            p = os.path.join(DATASET_DIR, k)
            if os.path.exists(p):
                assemble_slices(k, p)
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()
