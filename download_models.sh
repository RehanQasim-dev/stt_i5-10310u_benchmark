#!/usr/bin/env bash
# ==============================================================================
# download_models.sh - Automated Downloader for STT CPU Benchmark Matrix
# Downloads GGUF (Q4_K_M, Q8_0), ONNX Runtime directories, and GGML BIN models
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
HANDY_MODELS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/com.pais.handy/models"

mkdir -p "$MODELS_DIR"

USER_AGENT="curl/7.81.0"

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

    echo "[DOWNLOAD] Fetching $filename..."
    if command -v aria2c &>/dev/null; then
        aria2c --user-agent="$USER_AGENT" -x 4 -s 4 -c -d "$target_dir" -o "$filename" "$url"
    else
        curl -H "User-Agent: $USER_AGENT" -L -C - -o "$dest" "$url"
    fi
}

download_tarball() {
    local extract_root="$1"
    local check_path="$2"
    local url="$3"
    local archive_name
    archive_name="$(basename "$url")"

    if [ -e "$check_path" ]; then
        echo "[OK] Already exists: $check_path"
        return 0
    fi

    echo "[DOWNLOAD] Fetching archive $archive_name..."
    local tmp_archive="$extract_root/$archive_name.part"
    if command -v aria2c &>/dev/null; then
        aria2c --user-agent="$USER_AGENT" -x 4 -s 4 -c -d "$extract_root" -o "$archive_name.part" "$url"
    else
        curl -H "User-Agent: $USER_AGENT" -L -C - -o "$tmp_archive" "$url"
    fi

    echo "[EXTRACT] Extracting $archive_name into $extract_root..."
    tar --exclude="._*" -xzf "$tmp_archive" -C "$extract_root"
    rm -f "$tmp_archive"
    echo "[OK] Extracted: $check_path"
}

symlink_to_handy() {
    local source_path="$1"
    local link_name="$2"

    if [ -d "$HANDY_MODELS_DIR" ]; then
        local dest_link="$HANDY_MODELS_DIR/$link_name"
        if [ ! -e "$dest_link" ]; then
            ln -snf "$source_path" "$dest_link"
            echo "[SYMLINK] Linked to Handy: $dest_link -> $source_path"
        fi
    fi
}

sync_all_handy_symlinks() {
    if command -v handy &>/dev/null || [ -d "$HANDY_MODELS_DIR" ]; then
        mkdir -p "$HANDY_MODELS_DIR"
        echo "[INFO] Syncing symlinks with Handy data store at $HANDY_MODELS_DIR..."
        
        # ONNX model directories
        [ -d "$MODELS_DIR/parakeet-tdt-0.6b-v2-int8" ] && symlink_to_handy "$MODELS_DIR/parakeet-tdt-0.6b-v2-int8" "parakeet-tdt-0.6b-v2-int8"
        [ -d "$MODELS_DIR/parakeet-tdt-0.6b-v3-int8" ] && symlink_to_handy "$MODELS_DIR/parakeet-tdt-0.6b-v3-int8" "parakeet-tdt-0.6b-v3-int8"
        [ -d "$MODELS_DIR/canary-180m-flash" ] && symlink_to_handy "$MODELS_DIR/canary-180m-flash" "canary-180m-flash"
        [ -d "$MODELS_DIR/moonshine-medium-streaming-en" ] && symlink_to_handy "$MODELS_DIR/moonshine-medium-streaming-en" "moonshine-medium-streaming-en"
        
        # BIN models
        if [ -f "$MODELS_DIR/whisper-medium-q4_1.bin" ]; then
            symlink_to_handy "$MODELS_DIR/whisper-medium-q4_1.bin" "whisper-medium-q4_1.bin"
        elif [ -f "$MODELS_DIR/whisper-medium/whisper-medium-q4_1.bin" ]; then
            symlink_to_handy "$MODELS_DIR/whisper-medium/whisper-medium-q4_1.bin" "whisper-medium-q4_1.bin"
        fi

        # GGUF model directories for direct Handy catalog discovery
        for d in "$MODELS_DIR"/*; do
            if [ -d "$d" ]; then
                local base_name
                base_name="$(basename "$d")"
                symlink_to_handy "$d" "$base_name"
            fi
        done
    fi
}

download_gguf_models() {
    echo "--- 1. GGUF Models (transcribe.cpp / GGML C++) ---"

    # Parakeet TDT 0.6B v2
    download_file "$MODELS_DIR/parakeet-tdt-0.6b-v2" "parakeet-tdt-0.6b-v2-Q4_K_M.gguf" \
        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q4_K_M.gguf"
    download_file "$MODELS_DIR/parakeet-tdt-0.6b-v2" "parakeet-tdt-0.6b-v2-Q8_0.gguf" \
        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q8_0.gguf"

    # Parakeet TDT 0.6B v3
    download_file "$MODELS_DIR/parakeet-tdt-0.6b-v3" "parakeet-tdt-0.6b-v3-Q4_K_M.gguf" \
        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q4_K_M.gguf"
    download_file "$MODELS_DIR/parakeet-tdt-0.6b-v3" "parakeet-tdt-0.6b-v3-Q8_0.gguf" \
        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf"

    # Canary 180M Flash
    download_file "$MODELS_DIR/canary-180m-flash" "canary-180m-flash-Q4_K_M.gguf" \
        "https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q4_K_M.gguf"
    download_file "$MODELS_DIR/canary-180m-flash" "canary-180m-flash-Q8_0.gguf" \
        "https://huggingface.co/handy-computer/canary-180m-flash-gguf/resolve/main/canary-180m-flash-Q8_0.gguf"

    # Whisper Small (English)
    download_file "$MODELS_DIR/whisper-small.en" "whisper-small.en-Q4_K_M.gguf" \
        "https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q4_K_M.gguf"
    download_file "$MODELS_DIR/whisper-small.en" "whisper-small.en-Q8_0.gguf" \
        "https://huggingface.co/handy-computer/whisper-small.en-gguf/resolve/main/whisper-small.en-Q8_0.gguf"

    # Whisper Medium (English)
    download_file "$MODELS_DIR/whisper-medium.en" "whisper-medium.en-Q4_K_M.gguf" \
        "https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q4_K_M.gguf"
    download_file "$MODELS_DIR/whisper-medium.en" "whisper-medium.en-Q8_0.gguf" \
        "https://huggingface.co/handy-computer/whisper-medium.en-gguf/resolve/main/whisper-medium.en-Q8_0.gguf"

    # Parakeet TDT CTC 110M
    download_file "$MODELS_DIR/parakeet-tdt_ctc-110m" "parakeet-tdt_ctc-110m-Q4_K_M.gguf" \
        "https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q4_K_M.gguf"
    download_file "$MODELS_DIR/parakeet-tdt_ctc-110m" "parakeet-tdt_ctc-110m-Q8_0.gguf" \
        "https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/main/parakeet-tdt_ctc-110m-Q8_0.gguf"

    # SenseVoice Small
    download_file "$MODELS_DIR/SenseVoiceSmall" "SenseVoiceSmall-Q4_K_M.gguf" \
        "https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q4_K_M.gguf"
    download_file "$MODELS_DIR/SenseVoiceSmall" "SenseVoiceSmall-Q8_0.gguf" \
        "https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q8_0.gguf"

    # Nemotron Speech Streaming 0.6B
    download_file "$MODELS_DIR/nemotron-speech-streaming-en-0.6b" "nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf" \
        "https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q4_K_M.gguf"
    download_file "$MODELS_DIR/nemotron-speech-streaming-en-0.6b" "nemotron-speech-streaming-en-0.6b-Q8_0.gguf" \
        "https://huggingface.co/handy-computer/nemotron-speech-streaming-en-0.6b-gguf/resolve/main/nemotron-speech-streaming-en-0.6b-Q8_0.gguf"
}

download_onnx_models() {
    echo "--- 2. ONNX Runtime Models (CPU Bundles) ---"

    # Parakeet TDT 0.6B v2 (ONNX Int8)
    download_tarball "$MODELS_DIR" "$MODELS_DIR/parakeet-tdt-0.6b-v2-int8/encoder-model.int8.onnx" \
        "https://blob.handy.computer/parakeet-v2-int8.tar.gz"

    # Parakeet TDT 0.6B v3 (ONNX Int8)
    download_tarball "$MODELS_DIR" "$MODELS_DIR/parakeet-tdt-0.6b-v3-int8/encoder-model.int8.onnx" \
        "https://blob.handy.computer/parakeet-v3-int8.tar.gz"

    # Canary 180M Flash (ONNX Int8)
    download_tarball "$MODELS_DIR" "$MODELS_DIR/canary-180m-flash/encoder-model.int8.onnx" \
        "https://blob.handy.computer/canary-180m-flash.tar.gz"

    # Moonshine V2 Medium Streaming (ONNX Int8)
    download_tarball "$MODELS_DIR" "$MODELS_DIR/moonshine-medium-streaming-en/encoder.ort" \
        "https://blob.handy.computer/moonshine-medium-streaming-en.tar.gz"
}

download_bin_models() {
    echo "--- 3. Legacy GGML Binary Models ---"

    # Whisper Medium (q4_1 BIN)
    download_file "$MODELS_DIR/whisper-medium" "whisper-medium-q4_1.bin" \
        "https://blob.handy.computer/whisper-medium-q4_1.bin"
    # Ensure root-level symlink inside models/ if needed by Handy
    if [ ! -e "$MODELS_DIR/whisper-medium-q4_1.bin" ]; then
        ln -snf "$MODELS_DIR/whisper-medium/whisper-medium-q4_1.bin" "$MODELS_DIR/whisper-medium-q4_1.bin"
    fi
}

list_models() {
    echo "=== STT CPU Benchmark Model Matrix ==="
    echo "GGUF Models (transcribe.cpp):"
    echo "  - Parakeet TDT 0.6B v2       [Q4_K_M (454 MB), Q8_0 (696 MB)]"
    echo "  - Parakeet TDT 0.6B v3       [Q4_K_M (454 MB), Q8_0 (696 MB)]"
    echo "  - Canary 180M Flash          [Q4_K_M (133 MB), Q8_0 (208 MB)]"
    echo "  - Whisper Small (English)    [Q4_K_M (164 MB), Q8_0 (257 MB)]"
    echo "  - Whisper Medium (English)   [Q4_K_M (481 MB), Q8_0 (831 MB)]"
    echo "  - Parakeet TDT CTC 110M      [Q4_K_M (86 MB),  Q8_0 (129 MB)]"
    echo "  - SenseVoice Small           [Q4_K_M (139 MB), Q8_0 (241 MB)]"
    echo "  - Nemotron Streaming 0.6B    [Q4_K_M (454 MB), Q8_0 (696 MB)]"
    echo ""
    echo "ONNX Models (ONNX Runtime CPU):"
    echo "  - Parakeet TDT 0.6B v2       [Int8 directory (631 MB)]"
    echo "  - Parakeet TDT 0.6B v3       [Int8 directory (639 MB)]"
    echo "  - Canary 180M Flash          [FP32/Int8 directory (204 MB)]"
    echo "  - Moonshine V2 Medium        [Int8 streaming directory (289 MB)]"
    echo ""
    echo "Binary Models (transcribe.cpp):"
    echo "  - Whisper Medium             [q4_1 GGML binary (469 MB)]"
}

show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Options:"
    echo "  --all       Download entire benchmark matrix (default)"
    echo "  --gguf      Download only GGUF quantized models"
    echo "  --onnx      Download only ONNX runtime model bundles"
    echo "  --bin       Download only legacy binary models"
    echo "  --list      List all models in the benchmark matrix"
    echo "  --sync      Synchronize symlinks with Handy data directory"
    echo "  --help      Show this help message"
}

MODE="${1:---all}"

case "$MODE" in
    --list)
        list_models
        exit 0
        ;;
    --help|-h)
        show_help
        exit 0
        ;;
    --gguf)
        echo "=== Downloading GGUF Models ==="
        download_gguf_models
        sync_all_handy_symlinks
        ;;
    --onnx)
        echo "=== Downloading ONNX Models ==="
        download_onnx_models
        sync_all_handy_symlinks
        ;;
    --bin)
        echo "=== Downloading Legacy Binary Models ==="
        download_bin_models
        sync_all_handy_symlinks
        ;;
    --sync)
        sync_all_handy_symlinks
        ;;
    --all)
        echo "=== Downloading Full STT CPU Benchmark Model Matrix ==="
        download_gguf_models
        download_onnx_models
        download_bin_models
        sync_all_handy_symlinks
        ;;
    *)
        echo "[ERROR] Unknown option: $MODE"
        show_help
        exit 1
        ;;
esac

echo "=== All requested models are ready in $MODELS_DIR ==="
