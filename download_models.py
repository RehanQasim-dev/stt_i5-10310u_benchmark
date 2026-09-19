#!/usr/bin/env python3
"""
download_models.py - Automated Model Downloader for STT CPU Benchmark Matrix

Downloads all evaluated models (GGUF Q4_K_M & Q8_0, ONNX bundles, and GGML BIN)
directly into the local project 'models/' folder.
Requires no arguments or custom settings - just run:
    python3 download_models.py
"""

import os
import sys
import time
import urllib.request
import tarfile
import shutil

# Resolve directories relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
HANDY_MODELS_DIR = os.path.expanduser(
    os.environ.get("XDG_DATA_HOME", "~/.local/share") + "/com.pais.handy/models"
)

# User-Agent header to bypass CDN bot protection (HuggingFace / Cloudflare)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ------------------------------------------------------------------------------
# Model Matrix Definitions
# ------------------------------------------------------------------------------

GGUF_MODELS = [
    # Parakeet TDT 0.6B v2
    {
        "subdir": "parakeet-tdt-0.6b-v2",
        "filename": "parakeet-tdt-0.6b-v2-Q4_K_M.gguf",
        "url": "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q4_K_M.gguf",
    },
    {
        "subdir": "parakeet-tdt-0.6b-v2",
        "filename": "parakeet-tdt-0.6b-v2-Q8_0.gguf",
        "url": "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q8_0.gguf",
    },
    # Parakeet TDT 0.6B v3
    {
        "subdir": "parakeet-tdt-0.6b-v3",
        "filename": "parakeet-tdt-0.6b-v3-Q4_K_M.gguf",
        "url": "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q4_K_M.gguf",
    },
    {
        "subdir": "parakeet-tdt-0.6b-v3",
        "filename": "parakeet-tdt-0.6b-v3-Q8_0.gguf",
        "url": "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf",
    },
    # Canary 180M Flash
    {
        "subdir": "canary-180m-flash",
        "filename": "canary-180m-flash-Q4_K_M.gguf",
        "url": "https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q4_K_M.gguf",
    },
    {
        "subdir": "canary-180m-flash",
        "filename": "canary-180m-flash-Q8_0.gguf",
        "url": "https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q8_0.gguf",
    },
    # Whisper Small (English)
    {
        "subdir": "whisper-small.en",
        "filename": "whisper-small.en-Q4_K_M.gguf",
        "url": "https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q4_K_M.gguf",
    },
    {
        "subdir": "whisper-small.en",
        "filename": "whisper-small.en-Q8_0.gguf",
        "url": "https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q8_0.gguf",
    },
    # Whisper Medium (English)
    {
        "subdir": "whisper-medium.en",
        "filename": "whisper-medium.en-Q4_K_M.gguf",
        "url": "https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q4_K_M.gguf",
    },
    {
        "subdir": "whisper-medium.en",
        "filename": "whisper-medium.en-Q8_0.gguf",
        "url": "https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q8_0.gguf",
    },
    # Parakeet TDT CTC 110M
    {
        "subdir": "parakeet-tdt_ctc-110m",
        "filename": "parakeet-tdt_ctc-110m-Q4_K_M.gguf",
        "url": "https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q4_K_M.gguf",
    },
    {
        "subdir": "parakeet-tdt_ctc-110m",
        "filename": "parakeet-tdt_ctc-110m-Q8_0.gguf",
        "url": "https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q8_0.gguf",
    },
    # SenseVoice Small
    {
        "subdir": "SenseVoiceSmall",
        "filename": "SenseVoiceSmall-Q4_K_M.gguf",
        "url": "https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q4_K_M.gguf",
    },
    {
        "subdir": "SenseVoiceSmall",
        "filename": "SenseVoiceSmall-Q8_0.gguf",
        "url": "https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q8_0.gguf",
    },
    # Nemotron Speech Streaming 0.6B
    {
        "subdir": "nemotron-speech-streaming-en-0.6b",
        "filename": "nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf",
        "url": "https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf",
    },
    {
        "subdir": "nemotron-speech-streaming-en-0.6b",
        "filename": "nemotron-speech-streaming-en-0.6b-Q8_0.gguf",
        "url": "https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q8_0.gguf",
    },
]

ONNX_BUNDLES = [
    # Parakeet TDT 0.6B v2 (Int8 ONNX)
    {
        "url": "https://blob.handy.computer/parakeet-v2-int8.tar.gz",
        "check_file": os.path.join(MODELS_DIR, "parakeet-tdt-0.6b-v2-int8", "encoder-model.int8.onnx"),
        "name": "Parakeet TDT 0.6B v2 (ONNX Int8)",
    },
    # Parakeet TDT 0.6B v3 (Int8 ONNX)
    {
        "url": "https://blob.handy.computer/parakeet-v3-int8.tar.gz",
        "check_file": os.path.join(MODELS_DIR, "parakeet-tdt-0.6b-v3-int8", "encoder-model.int8.onnx"),
        "name": "Parakeet TDT 0.6B v3 (ONNX Int8)",
    },
    # Canary 180M Flash (ONNX Int8)
    {
        "url": "https://blob.handy.computer/canary-180m-flash.tar.gz",
        "check_file": os.path.join(MODELS_DIR, "canary-180m-flash", "encoder-model.int8.onnx"),
        "name": "Canary 180M Flash (ONNX Int8)",
    },
    # Moonshine V2 Medium Streaming (ONNX Int8)
    {
        "url": "https://blob.handy.computer/moonshine-medium-streaming-en.tar.gz",
        "check_file": os.path.join(MODELS_DIR, "moonshine-medium-streaming-en", "encoder.ort"),
        "name": "Moonshine V2 Medium Streaming (ONNX Int8)",
    },
]

BIN_MODELS = [
    # Whisper Medium (q4_1 legacy GGML binary)
    {
        "subdir": "whisper-medium",
        "filename": "whisper-medium-q4_1.bin",
        "url": "https://blob.handy.computer/whisper-medium-q4_1.bin",
    },
]

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def format_bytes(size_bytes: int) -> str:
    """Format byte counts into human-readable strings."""
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"

def download_file(url: str, dest_path: str, display_name: str = ""):
    """
    Stream download a file with progress reporting and atomic renaming.
    Avoids corrupt files by downloading to .part first.
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[OK] Already exists: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    part_path = f"{dest_path}.part"

    label = display_name or os.path.basename(dest_path)
    print(f"\n[DOWNLOAD] Fetching {label}...")
    print(f"           Source: {url}")

    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    start_time = time.time()

    try:
        with urllib.request.urlopen(req) as resp, open(part_path, "wb") as out_file:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunk

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                elapsed = time.time() - start_time
                speed = (downloaded / elapsed) if elapsed > 0 else 0
                speed_str = f"{format_bytes(speed)}/s"

                if total_size > 0:
                    percent = (downloaded / total_size) * 100.0
                    status_line = (
                        f"\r  -> [{percent:5.1f}%] {format_bytes(downloaded)} / "
                        f"{format_bytes(total_size)} ({speed_str})"
                    )
                else:
                    status_line = f"\r  -> {format_bytes(downloaded)} downloaded ({speed_str})"

                sys.stdout.write(status_line)
                sys.stdout.flush()

        sys.stdout.write("\n")
        # Atomic rename once completely downloaded
        os.rename(part_path, dest_path)
        print(f"[OK] Completed: {dest_path}")

    except Exception as e:
        if os.path.exists(part_path):
            os.remove(part_path)
        print(f"\n[ERROR] Failed to download {label}: {e}")
        raise

def download_tarball(url: str, extract_root: str, check_file: str, display_name: str = ""):
    """Download and extract a tar.gz archive into extract_root, omitting macOS metadata."""
    if os.path.exists(check_file) and os.path.getsize(check_file) > 0:
        print(f"[OK] Already exists: {check_file}")
        return

    os.makedirs(extract_root, exist_ok=True)
    archive_name = os.path.basename(url)
    part_archive = os.path.join(extract_root, f"{archive_name}.part")

    download_file(url, part_archive, display_name=display_name or archive_name)

    print(f"[EXTRACT] Unpacking {archive_name} into {extract_root}...")
    try:
        with tarfile.open(part_archive, "r:gz") as tar:
            # Filter out macOS hidden metadata files
            members = [m for m in tar.getmembers() if not os.path.basename(m.name).startswith("._")]
            tar.extractall(path=extract_root, members=members)
        print(f"[OK] Extracted successfully: {check_file}")
    finally:
        if os.path.exists(part_archive):
            os.remove(part_archive)

def sync_handy_symlinks():
    """
    Ensure all models in project models/ are symlinked into Handy's store
    at ~/.local/share/com.pais.handy/models/ for seamless zero-copy detection.
    """
    if not (shutil.which("handy") or os.path.isdir(HANDY_MODELS_DIR)):
        return

    os.makedirs(HANDY_MODELS_DIR, exist_ok=True)
    print("\n[INFO] Synchronizing model symlinks with Handy data directory...")

    # Symlink top-level model folders and files
    for entry in os.listdir(MODELS_DIR):
        src = os.path.join(MODELS_DIR, entry)
        dst = os.path.join(HANDY_MODELS_DIR, entry)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
                print(f"[SYMLINK] Linked {entry} -> Handy store")
            except Exception as e:
                print(f"[WARN] Failed to link {entry}: {e}")

    # Also make sure root-level whisper-medium-q4_1.bin exists if nested
    nested_bin = os.path.join(MODELS_DIR, "whisper-medium", "whisper-medium-q4_1.bin")
    root_bin = os.path.join(MODELS_DIR, "whisper-medium-q4_1.bin")
    if os.path.exists(nested_bin) and not os.path.exists(root_bin):
        os.symlink(nested_bin, root_bin)
    elif os.path.exists(root_bin) and not os.path.exists(nested_bin):
        os.makedirs(os.path.dirname(nested_bin), exist_ok=True)
        os.symlink(root_bin, nested_bin)

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("STT Benchmark Suite - Automated Model Downloader")
    print(f"Target Directory: {MODELS_DIR}")
    print("=" * 80)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. GGUF Quantized Models (transcribe.cpp GGML)
    print("\n--- 1. GGUF Models (transcribe.cpp / GGML C++) ---")
    for item in GGUF_MODELS:
        target_dir = os.path.join(MODELS_DIR, item["subdir"])
        target_path = os.path.join(target_dir, item["filename"])
        download_file(item["url"], target_path, display_name=item["filename"])

    # 2. ONNX Runtime CPU Bundles
    print("\n--- 2. ONNX Runtime Models (CPU Bundles) ---")
    for item in ONNX_BUNDLES:
        download_tarball(
            url=item["url"],
            extract_root=MODELS_DIR,
            check_file=item["check_file"],
            display_name=item["name"],
        )

    # 3. Legacy GGML Binary Models
    print("\n--- 3. Legacy GGML Binary Models ---")
    for item in BIN_MODELS:
        target_dir = os.path.join(MODELS_DIR, item["subdir"])
        target_path = os.path.join(target_dir, item["filename"])
        download_file(item["url"], target_path, display_name=item["filename"])

    # 4. Sync Handy Symlinks
    sync_handy_symlinks()

    print("\n" + "=" * 80)
    print(f"All models successfully verified in {MODELS_DIR}")
    print("Ready to run benchmark: python3 run_benchmark.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
