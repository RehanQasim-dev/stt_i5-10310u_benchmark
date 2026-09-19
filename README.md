# Speech-to-Text Benchmark Suite (Intel Core i5-10310U)

A reproducible, end-to-end speech-to-text (STT) benchmark suite evaluating models across **Q4_K_M**, **Q8_0**, and **ONNX** formats on an Intel Core i5-10310U CPU.

---

## 1. System Target & Architecture

- **CPU:** Intel(R) Core(TM) i5-10310U CPU @ 1.70GHz (Comet Lake, 4 Cores / 8 Threads, AVX2, FMA3)
- **iGPU:** Intel(R) UHD Graphics (CML GT2)
- **RAM:** 16 GB DDR4
- **OS:** Linux (Ubuntu/Debian-based x86_64)
- **Primary Engine:** [Handy](https://github.com/cjpais/handy)

### Internal Dual-Backend Architecture of Handy
Handy acts as the unified STT execution engine, managing two distinct runtime backends under the hood:
1. **transcribe.cpp (GGML C++):** Used automatically for all `.gguf` and legacy `.bin` models (Parakeet, Canary, Whisper, SenseVoice, Nemotron).
2. **ONNX Runtime (CPU):** Used automatically for directory-based `.onnx` models (Parakeet V2/V3 int8, Canary 180M, Moonshine Streaming).

---

## 2. Configuring Handy to Use CPU

When running GGUF models via `transcribe.cpp`, Handy defaults to the integrated GPU (`Vulkan0`), which is slower than the CPU on this hardware, so we configure Handy to use the CPU.

Because modern versions of Handy do not expose a hardware dropdown in the settings GUI, you can set CPU execution through either of the following methods:

### Method 1: Permanent Config Update (Recommended)
Edit Handy's persisted configuration file:
1. Stop Handy if it is currently running:
   ```bash
   killall -9 handy
   ```
2. Update `transcribe_accelerator` from `"auto"` to `"cpu"` in `~/.local/share/com.pais.handy/settings_store.json`:
   ```bash
   sed -i 's/"transcribe_accelerator": "auto"/"transcribe_accelerator": "cpu"/' ~/.local/share/com.pais.handy/settings_store.json
   ```
3. Relaunch Handy. The logs will confirm CPU binding:
   ```text
   [handy_app_lib::managers::transcription][INFO] transcribe.cpp accelerator preference: Cpu
   [transcribe_cpp][INFO] whisper: using cpu backend (strict)
   [handy_app_lib::managers::transcription][INFO] Loaded whisper model ... bound backend 'CPU', bound device 'Intel(R) Core(TM) i5-10310U CPU @ 1.70GHz'
   ```

### Method 2: Headless CLI Override
Pass `--device-index 1` directly to Handy when executing transcriptions:
```bash
handy -f audio.wav --model medium --device-index 1
```

---

## 3. Model Benchmark Matrix

| Model Family | Variant | Quantization | Size (MB) | Handy Internal Backend | Architecture |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Parakeet TDT 0.6B v2** | Flagship | Q4_K_M | 454 MB | transcribe.cpp | FastConformer Non-Autoregressive |
| **Parakeet TDT 0.6B v2** | Flagship | Q8_0 | 696 MB | transcribe.cpp | FastConformer Non-Autoregressive |
| **Parakeet TDT 0.6B v2** | Flagship | Int8 | 631 MB | ONNX Runtime (CPU) | FastConformer Non-Autoregressive |
| **Parakeet TDT 0.6B v3** | Multilingual | Q4_K_M | 454 MB | transcribe.cpp | FastConformer Non-Autoregressive |
| **Parakeet TDT 0.6B v3** | Multilingual | Q8_0 | 696 MB | transcribe.cpp | FastConformer Non-Autoregressive |
| **Parakeet TDT 0.6B v3** | Multilingual | Int8 | 639 MB | ONNX Runtime (CPU) | FastConformer Non-Autoregressive |
| **Canary 180M Flash** | Multilingual | Q4_K_M | 133 MB | transcribe.cpp | Conformer-AED (Autoregressive) |
| **Canary 180M Flash** | Multilingual | Q8_0 | 208 MB | transcribe.cpp | Conformer-AED (Autoregressive) |
| **Canary 180M Flash** | Multilingual | FP32/Int8 | 204 MB | ONNX Runtime (CPU) | Conformer-AED (Autoregressive) |
| **Whisper Small.en** | English | Q4_K_M | 164 MB | transcribe.cpp | Whisper Enc-Dec (Autoregressive) |
| **Whisper Small.en** | English | Q8_0 | 257 MB | transcribe.cpp | Whisper Enc-Dec (Autoregressive) |
| **Parakeet TDT CTC 110M**| Lightweight | Q4_K_M | 86 MB | transcribe.cpp | Conformer-CTC |
| **Parakeet TDT CTC 110M**| Lightweight | Q8_0 | 129 MB | transcribe.cpp | Conformer-CTC |
| **SenseVoice Small** | Multilingual | Q4_K_M | 139 MB | transcribe.cpp | SenseVoice Non-Autoregressive CTC |
| **SenseVoice Small** | Multilingual | Q8_0 | 241 MB | transcribe.cpp | SenseVoice Non-Autoregressive CTC |
| **Whisper Medium.en** | English | Q4_K_M | 481 MB | transcribe.cpp | Whisper Enc-Dec (Autoregressive) |
| **Whisper Medium.en** | English | Q8_0 | 831 MB | transcribe.cpp | Whisper Enc-Dec (Autoregressive) |
| **Whisper Medium** | Legacy | q4_1 | 469 MB | transcribe.cpp | Whisper Enc-Dec (Autoregressive) |
| **Nemotron Streaming 0.6B**| Streaming | Q4_K_M | 454 MB | transcribe.cpp | FastConformer Buffered Streaming |
| **Nemotron Streaming 0.6B**| Streaming | Q8_0 | 696 MB | transcribe.cpp | FastConformer Buffered Streaming |
| **Moonshine V2 Medium** | Streaming | Int8 | 289 MB | ONNX Runtime (CPU) | Conformer Streaming |

---

## 4. Deterministic Slicing Dataset Methodology

To benchmark models across varying durations (30s, 60s, 120s, 180s) without slicing through middle-of-word phonemes:

Traditional time-based slicing cuts through words and vowels, which artificially spikes deletions and substitutions in Word Error Rate (WER) scoring.

`record_dataset.py` implements **Cumulative Calibration Recording**:
1. You record **4 natural speech blocks**:
   - Block 1: ~30s (~70 words)
   - Block 2: ~30s (~70 words)
   - Block 3: ~60s (~140 words)
   - Block 4: ~60s (~140 words)
2. Slices are concatenated losslessly using ffmpeg:
   - `slice_30s`: Block 1
   - `slice_60s`: Block 1 + Block 2
   - `slice_120s`: Block 1 + Block 2 + Block 3
   - `slice_180s`: Block 1 + Block 2 + Block 3 + Block 4
3. Corresponding reference `.txt` files are compiled automatically, guaranteeing 100% mathematically exact ground truth with 0 cut-off words.

Two scripts are provided:
- **Script A:** Everyday English narrative.
- **Script B:** Deep technical domain covering GPU microarchitectures, llama.cpp kernels, FlashAttention, and memory hierarchies.

---

## 5. Quickstart

### Step 1: Download Model Weights
Download the model weights into the `models/` directory:
```bash
./download_models.sh
```
Or selectively download subsets:
```bash
./download_models.sh --list   # View complete model matrix with file sizes
./download_models.sh --gguf   # Download only GGUF quantized models
./download_models.sh --onnx   # Download only ONNX runtime model bundles
./download_models.sh --bin    # Download only legacy binary models
```
The script automatically synchronizes symlinks with Handy's local model store (`~/.local/share/com.pais.handy/models/`).

### Step 2: Record Calibrated Audio Dataset
```bash
python3 record_dataset.py
```
Follow the interactive prompts to record Script A and/or Script B.

### Step 3: Run the Benchmark
```bash
python3 run_benchmark.py
```
This executes all model configurations via Handy on CPU, measures execution latency, computes Word Error Rate (WER), and outputs Markdown and JSON reports under `logs/`.
---
