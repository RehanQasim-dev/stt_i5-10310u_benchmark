# Project Handoff: STT CPU Benchmark (Intel Core i5-10310U)

## 1. Project Identity & Location
- **Workspace Location:** `/home/rehan-10xe/Documents/chats/stt_i5-10310u_benchmark`
- **GitHub Repository:** [https://github.com/RehanQasim-dev/stt_i5-10310u_benchmark](https://github.com/RehanQasim-dev/stt_i5-10310u_benchmark)
- **Branch:** `main` (clean working directory, in sync with origin)
- **Host Hardware:** Intel(R) Core(TM) i5-10310U CPU @ 1.70GHz (4 Cores / 8 Threads, AVX2 SIMD), 16 GB DDR4 RAM, Intel UHD Graphics (CML GT2)

---

## 2. Core Architecture & Dispatch Mechanics
The benchmark uses **Handy** as the primary speech-to-text engine. Handy internally abstracts two distinct execution runtimes:
1. **transcribe.cpp (GGML C++):** Automatically processes `.gguf` and legacy `.bin` models (Parakeet, Canary, Whisper, SenseVoice, Nemotron).
2. **ONNX Runtime (CPU):** Automatically processes multi-file directory `.onnx` models (Parakeet V2/V3 int8, Canary 180M, Moonshine Streaming).

### The Vulkan vs CPU Hardware Nuance
By default (`"transcribe_accelerator": "auto"`), `transcribe.cpp` binds to discovered GPU hardware before CPU. On Intel integrated graphics (`Vulkan0`), shader pipeline compilation and shared system memory bandwidth cause significant latency spikes compared to the CPU AVX2 SIMD backend (`libggml-cpu-haswell.so`).

Handy has been explicitly configured to prioritize CPU:
- **Persisted Config:** `"transcribe_accelerator": "cpu"` in `~/.local/share/com.pais.handy/settings_store.json`.
- **CLI Invocations:** Forced via `--device-index 1` or `--backend cpu`.

---

## 3. Current File Structure & State

| File / Directory | Purpose & Status |
| :--- | :--- |
| `download_models.sh` | Automated model downloader. Supports `--all`, `--gguf`, `--onnx`, `--bin`, `--list`, and `--sync`. Sets `curl` user-agent to bypass CDN 403 blocks and strips macOS `._*` metadata. Auto-symlinks downloaded models into Handy. |
| `record_dataset.py` | Interactive cumulative calibration recorder. Records 4 natural speech blocks and losslessly concatenates them into `slice_30s`, `slice_60s`, `slice_120s`, and `slice_180s` with exact reference text files. |
| `run_benchmark.py` | Automated benchmarking harness. Dynamically detects CPU specs via `/proc/cpuinfo`, dynamically locates `transcribe-cli`, checks and auto-links Handy models, computes Word Error Rate (WER) and Real-Time Factor (RTF), and formats markdown/JSON reports. |
| `README.md` | Comprehensive user guide, architecture documentation, model matrix, and quickstart instructions (zero emojis, no hardcoded paths). |
| `.gitignore` | Excludes all heavy binary weights (`*.gguf`, `*.bin`, `*.onnx`, `*.ort`, `*.tar.gz`), audio recordings (`*.wav`, `dataset/`), and output run logs (`logs/`). |
| `models/` | Local storage directory for all downloaded models. Symlinked backward-compatibly with `~/.local/share/com.pais.handy/models/` and `~/Documents/transcribe.cpp/models/` consuming 0 extra disk space. |

---

## 4. Evaluated Model Matrix

| Model | Variant / Arch | Formats Available | Quantizations |
| :--- | :--- | :---: | :---: |
| **Parakeet TDT 0.6B v2** | FastConformer TDT | GGUF / ONNX | Q4_K_M, Q8_0, Int8 |
| **Parakeet TDT 0.6B v3** | FastConformer TDT | GGUF / ONNX | Q4_K_M, Q8_0, Int8 |
| **Canary 180M Flash** | Conformer-AED | GGUF / ONNX | Q4_K_M, Q8_0, FP32/Int8 |
| **Whisper Small.en** | Whisper Enc-Dec | GGUF | Q4_K_M, Q8_0 |
| **Whisper Medium.en** | Whisper Enc-Dec | GGUF | Q4_K_M, Q8_0 |
| **Whisper Medium** | Whisper Enc-Dec | GGML BIN | q4_1 |
| **Parakeet TDT CTC 110M** | Conformer-CTC | GGUF | Q4_K_M, Q8_0 |
| **SenseVoice Small** | SenseVoice Non-Autoregressive | GGUF | Q4_K_M, Q8_0 |
| **Nemotron Streaming 0.6B** | FastConformer Streaming | GGUF | Q4_K_M, Q8_0 |
| **Moonshine V2 Medium** | Conformer Streaming | ONNX | Int8 |

---

## 5. Immediate Next Steps for Next Session

1. **Record the Audio Dataset:**
   Run the interactive recorder to generate deterministic audio slices:
   ```bash
   python3 record_dataset.py
   ```
   Follow the prompts to record either Script A (everyday speech) or Script B (technical compiler/SIMD text). The script will output sliced files to `dataset/` along with matching reference `.txt` files.

2. **Execute the Benchmark Matrix:**
   Once audio slices are present in `dataset/`, run the benchmark:
   ```bash
   python3 run_benchmark.py
   ```
   The harness will execute each model against each slice, measure wall-clock latency, calculate RTF and WER, and emit JSON and markdown logs under `logs/`.

3. **Generate Empirical Markdown Report:**
   After benchmarking completes:
   - Extract the generated summary table from `logs/`.
   - Create a dedicated markdown file (e.g., `BENCHMARK_RESULTS.md`) in the repository documenting the actual latency, RTF, and accuracy numbers across Q4_K_M vs Q8_0 vs ONNX Int8.
   - Commit and push `BENCHMARK_RESULTS.md` to GitHub.

---

## 6. Essential Operational Commands

Check model catalog and download status:
```bash
./download_models.sh --list
```

Verify Handy recognized models headlessly:
```bash
handy --list-models --json | grep -E '"id":|"name":|"is_downloaded":'
```

Verify Handy compute devices:
```bash
handy --list-devices
```

Commit new findings or updates to GitHub:
```bash
git status
git add <files>
git commit -m "<descriptive message>"
git push origin main
```
