#!/usr/bin/env bash
# ==============================================================================
# download_models.sh - Automated Downloader for STT CPU Benchmark Matrix
# Fetches optimized GGUF (Q4_K_M, Q8_0) and ONNX models from handy-computer HuggingFace
# ==============================================================================

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
mkdir -p "$MODELS_DIR"

DL_TOOL="curl -L -C - -o"
if command -v aria2c &>/dev/null; then
    DL_TOOL="aria2c -x 4 -s 4 -c -d"
fi

download_file() {
    local target_dir="$1"
    local filename="$2"
    local url="$3"

    mkdir -p "$target_dir"
    local dest="$target_dir/$filename"

    if [ -f "$dest" ]; then
        echo "[OK] Already exists: $dest"
        return 0
    fi

    echo "[*] Downloading $filename..."
    if command -v aria2c &>/dev/null; then
        aria2c -x 4 -s 4 -c -d "$target_dir" -o "$filename" "$url"
    else
        curl -L -C - -o "$dest" "$url"
    fi
}

echo "=== Downloading STT CPU Benchmark Model Matrix ==="

# 1. Parakeet TDT 0.6B v2
download_file "$MODELS_DIR/parakeet-tdt-0.6b-v2" "parakeet-tdt-0.6b-v2-Q4_K_M.gguf" \
    "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q4_K_M.gguf"
download_file "$MODELS_DIR/parakeet-tdt-0.6b-v2" "parakeet-tdt-0.6b-v2-Q8_0.gguf" \
    "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q8_0.gguf"

# 2. Canary 180M Flash
download_file "$MODELS_DIR/canary-180m-flash" "canary-180m-flash-Q4_K_M.gguf" \
    "https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q4_K_M.gguf"
download_file "$MODELS_DIR/canary-180m-flash" "canary-180m-flash-Q8_0.gguf" \
    "https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q8_0.gguf"

# 3. Whisper Small (English)
download_file "$MODELS_DIR/whisper-small.en" "whisper-small.en-Q4_K_M.gguf" \
    "https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q4_K_M.gguf"
download_file "$MODELS_DIR/whisper-small.en" "whisper-small.en-Q8_0.gguf" \
    "https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q8_0.gguf"

# 4. Parakeet TDT CTC 110M
download_file "$MODELS_DIR/parakeet-tdt_ctc-110m" "parakeet-tdt_ctc-110m-Q4_K_M.gguf" \
    "https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q4_K_M.gguf"
download_file "$MODELS_DIR/parakeet-tdt_ctc-110m" "parakeet-tdt_ctc-110m-Q8_0.gguf" \
    "https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q8_0.gguf"

# 5. SenseVoiceSmall
download_file "$MODELS_DIR/SenseVoiceSmall" "SenseVoiceSmall-Q4_K_M.gguf" \
    "https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q4_K_M.gguf"
download_file "$MODELS_DIR/SenseVoiceSmall" "SenseVoiceSmall-Q8_0.gguf" \
    "https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q8_0.gguf"

# 6. Whisper Medium (English)
download_file "$MODELS_DIR/whisper-medium.en" "whisper-medium.en-Q4_K_M.gguf" \
    "https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q4_K_M.gguf"

# 7. Nemotron Speech Streaming 0.6B
download_file "$MODELS_DIR/nemotron-speech-streaming-en-0.6b" "nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf" \
    "https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf"

echo "=== All models ready in $MODELS_DIR ==="
