# Speech-to-Text Model Matrix Benchmark (Intel Core i5-10310U)

A reproducible, end-to-end speech-to-text (STT) benchmark suite comparing state-of-the-art speech models strictly under **700 MB** across **`Q4_K_M`**, **`Q8_0`**, and **`ONNX`** formats on modern low-power x86 CPU hardware.

---

## 1. System Hardware Target

- **CPU:** Intel(R) Core(TM) i5-10310U CPU @ 1.70GHz (Comet Lake, 4 Cores / 8 Threads, AVX2, FMA3)
- **iGPU:** Intel(R) UHD Graphics (CML GT2, Mesa Vulkan)
- **RAM:** 16 GB DDR4
- **OS:** Linux (Ubuntu/Debian-based x86_64)
- **Engines:** [Handy](https://github.com/cjpais/handy), [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (GGML C++), and ONNX Runtime (CPU)

---

## 2. The Handy & transcribe.cpp GPU Issue & The CPU Fix

### The Trap: Why Handy Defaults to the Slow GPU
When Handy launches with its default `"transcribe_accelerator": "auto"`, it probes hardware compute devices:
```text
transcribe-cpp compute devices:
  index=0 kind=vulkan name=Intel(R) UHD Graphics (CML GT2) (vram=11437MB)
  index=1 kind=cpu    name=Intel(R) Core(TM) i5-10310U CPU @ 1.70GHz (vram=15250MB)
```
In `transcribe.cpp` and GGML, any discovered GPU accelerator takes priority over the CPU when `"auto"` is selected. As a result, Handy silently binds to **Device Index 0 (`Vulkan0`)**:

```text
[transcribe_cpp][INFO] whisper: using vulkan backend: Vulkan0
[handy_app_lib::managers::transcription][INFO] Loaded whisper model ... bound backend 'Vulkan0', bound device 'Intel(R) UHD Graphics (CML GT2)'
```

### The Problem: Why Vulkan is Slower on Intel Integrated GPUs
Integrated Intel graphics share system RAM, have low compute density, lack matrix hardware cores, and incur heavy Vulkan driver pipeline/shader compilation overhead. 

On this Core i5 laptop, **the CPU with AVX2 SIMD kernels (`libggml-cpu-haswell.so`) is significantly faster and more stable than Vulkan**:
- **Whisper Medium (Vulkan0 iGPU):** 49.8s (1.0× Real-Time)
- **Whisper Medium (CPU AVX2):** 43.8s (1.1× Real-Time)
- **Parakeet TDT 0.6B (CPU AVX2):** 10.2s (5.0× Real-Time)

### The Permanent Fix
In current Handy versions (v0.9.x), the accelerator toggle was removed from the UI. To permanently force CPU execution:

1. Stop the running Handy process:
   ```bash
   killall -9 handy
   ```
2. Update `~/.local/share/com.pais.handy/settings_store.json`:
   ```bash
   sed -i 's/"transcribe_accelerator": "auto"/"transcribe_accelerator": "cpu"/' ~/.local/share/com.pais.handy/settings_store.json
   ```
3. Relaunch Handy. The logs will confirm CPU binding:
   ```text
   [handy_app_lib::managers::transcription][INFO] transcribe.cpp accelerator preference: Cpu
   [transcribe_cpp][INFO] whisper: using cpu backend (strict)
   [handy_app_lib::managers::transcription][INFO] Loaded whisper model ... bound backend 'CPU', bound device 'Intel(R) Core(TM) i5-10310U CPU @ 1.70GHz'
   ```

*Alternative CLI Override:* Pass `--device-index 1` to `handy -f <audio.wav>` or use `transcribe-cli --backend cpu`.

---

## 3. Model Benchmark Matrix (< 700 MB)

All models are constrained to $\le 700$ MB to ensure instant loading and minimal RAM usage on edge devices.

| Model Family | Variant | Quantization | Size (MB) | Engine | Architecture |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Parakeet TDT 0.6B v2** | Flagship | `Q4_K_M` | 454 MB | `transcribe.cpp` | FastConformer Non-Autoregressive |
| **Parakeet TDT 0.6B v2** | Flagship | `Q8_0` | 696 MB | `transcribe.cpp` | FastConformer Non-Autoregressive |
| **Parakeet TDT 0.6B v2** | Flagship | `Int8` | 631 MB | `handy` (ONNX) | FastConformer Non-Autoregressive |
| **Canary 180M Flash** | Multilingual | `Q4_K_M` | 133 MB | `transcribe.cpp` | Conformer-AED (Autoregressive) |
| **Canary 180M Flash** | Multilingual | `Q8_0` | 208 MB | `transcribe.cpp` | Conformer-AED (Autoregressive) |
| **Canary 180M Flash** | Multilingual | `FP32/Int8` | 204 MB | `handy` (ONNX) | Conformer-AED (Autoregressive) |
| **Whisper Small.en** | English | `Q4_K_M` | 164 MB | `transcribe.cpp` | Whisper Enc-Dec (Autoregressive) |
| **Whisper Small.en** | English | `Q8_0` | 257 MB | `transcribe.cpp` | Whisper Enc-Dec (Autoregressive) |
| **Parakeet TDT CTC 110M**| Lightweight | `Q4_K_M` | 86 MB | `transcribe.cpp` | Conformer-CTC |
| **Parakeet TDT CTC 110M**| Lightweight | `Q8_0` | 129 MB | `transcribe.cpp` | Conformer-CTC |
| **SenseVoice Small** | Multilingual | `Q4_K_M` | 139 MB | `transcribe.cpp` | SenseVoice Non-Autoregressive CTC |
| **SenseVoice Small** | Multilingual | `Q8_0` | 241 MB | `transcribe.cpp` | SenseVoice Non-Autoregressive CTC |
| **Whisper Medium.en** | English | `Q4_K_M` | 481 MB | `transcribe.cpp` | Whisper Enc-Dec (Autoregressive) |
| **Whisper Medium** | Legacy | `q4_1` | 469 MB | `handy` (BIN) | Whisper Enc-Dec (Autoregressive) |
| **Nemotron Streaming 0.6B**| Streaming | `Q4_K_M` | 454 MB | `transcribe.cpp` | FastConformer Buffered Streaming |
| **Moonshine V2 Medium** | Streaming | `Int8` | 289 MB | `handy` (ONNX) | Conformer Streaming |

*(Note: Whisper Medium in `Q8_0` is 793 MB and Parakeet V3 in `Q8_0` is 705 MB, so they are excluded to respect the 700 MB boundary).*

### Why Per-Tensor Bucket Quantization Matters
Standard `Q4_K_M` quants in `transcribe.cpp` use standard GGML vector kernels, but apply an ASR-tuned per-tensor routing policy:
- **`Linear`** GEMMs: `Q4_K` / `Q8_0`
- **`Linear Fallback`** (non-256 divisible shapes like Parakeet `ne0=640` or Whisper `ne0=384`): Kept at `Q8_0` instead of degrading to `Q4_1` or inflating to `F16`.
- **`Attention Outputs`**: Bumped to `Q8_0` to preserve attention alignment.
- **`Token Embeddings`**: Bumped to `Q6_K` to prevent subword vocabulary collapse.
- **`Convolutions & Norms`**: Preserved at `F16`/`F32` to prevent acoustic frontend degradation.

---

## 4. Deterministic Slicing Dataset Methodology

To benchmark models across various audio lengths (**30s, 60s, 120s, 180s**) without word-boundary artifacts:

Traditional arbitrary audio slicing cuts through vowels and syllables mid-word, inducing false deletion and substitution penalties in Word Error Rate (WER) scoring.

`record_dataset.py` solves this via **Cumulative Calibration Recording**:
1. You record **4 natural pauses / blocks**:
   - Block 1: ~30s (~70 words)
   - Block 2: ~30s (~70 words)
   - Block 3: ~60s (~140 words)
   - Block 4: ~60s (~140 words)
2. Slices are concatenated losslessly using `ffmpeg`:
   - `slice_30s`: Block 1
   - `slice_60s`: Block 1 + Block 2
   - `slice_120s`: Block 1 + Block 2 + Block 3
   - `slice_180s`: Block 1 + Block 2 + Block 3 + Block 4
3. Matching golden `.txt` files are assembled automatically, guaranteeing **100% mathematically exact ground-truth text** with **zero cut-off words**.

Two evaluation domains are included:
- **Script A:** Everyday English narrative & natural rhythm.
- **Script B:** Deep technical domain stress-testing GPU microarchitectures, `llama.cpp` kernels, FlashAttention, and memory hierarchies.

---

## 5. Quickstart

### Step 1: Download Model Weights
Run the automated downloader (skips files already downloaded):
```bash
./download_models.sh
```

### Step 2: Record Your Calibrated Audio Dataset
```bash
python3 record_dataset.py
```
Follow the interactive CLI prompts to record Script A (Non-technical) and/or Script B (Technical).

### Step 3: Run the Matrix Benchmark
```bash
python3 run_benchmark.py
```
This executes all 16 model configurations on CPU, measures latency, calculates Word Error Rate (WER), and generates a structured Markdown report under `logs/matrix_eval_<timestamp>/benchmark_report.md`.

---

## 6. Empirical Preview (49.5s Technical Sample on i5-10310U)

| Model | Format | Quant | Size | Latency | Speedup (xRT) | Word Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SenseVoice Small** | GGUF | `Q4_K_M` | 139 MB | 7.13s | **7.0×** | Good (Best for $\le 30$s) |
| **Parakeet TDT 0.6B v2** | GGUF | `Q4_K_M` | 454 MB | 10.28s | **5.0×** | **83.1%** (Best overall) |
| **Whisper Small.en** | GGUF | `Q4_K_M` | 164 MB | 12.18s | **4.0×** | 76.5% |
| **Canary 180M Flash** | GGUF | `Q8_0` | 208 MB | 9.32s | **5.3×** | 80.2% |
| **Whisper Medium.en** | GGUF | `Q4_K_M` | 481 MB | 50.58s | **1.0×** | **85.2%** (Highest precision) |

---

## 7. License

Apache 2.0 / CC-BY-4.0 depending on upstream model weights. See respective model cards in `models/`.
