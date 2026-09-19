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
| `download_models.py` | Automated model downloader. Fetches all benchmark models (GGUF Q4_K_M & Q8_0, ONNX bundles, GGML BIN) directly into `./models` with no flags or settings needed. Auto-symlinks into Handy store. |
| `download_models.sh` | Shell-based alternative model downloader. Supports `--all`, `--gguf`, `--onnx`, `--bin`, `--list`, and `--sync`. |
| `record_dataset.py` | Interactive cumulative calibration recorder. Records 4 natural speech blocks and losslessly concatenates them into `slice_30s`, `slice_60s`, `slice_120s`, and `slice_180s` with exact reference text files. |
| `run_benchmark.py` | Automated benchmarking harness (Handy-only). Features Levenshtein error breakdown, real-time per-model logging into `logs/latest/<Model>_<Format>_<Quant>/`, dynamic `BENCHMARK_SUMMARY.md` updates after every single inference, and automated symlink maintenance. |
| `README.md` | Comprehensive user guide, architecture documentation, model matrix, and quickstart instructions (zero emojis, no hardcoded paths). |
| `.gitignore` | Excludes all heavy binary weights (`*.gguf`, `*.bin`, `*.onnx`, `*.ort`, `*.tar.gz`), audio recordings (`*.wav`, `dataset/`), and output run logs (`logs/`). |
| `models/` | Local storage directory for all downloaded models (6.9 GB ready-to-run). Physical source of truth cut/moved from external caches, symlinked into `~/.local/share/com.pais.handy/models/`. |

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

1. **Record the Final Audio Dataset:**
   When ready to record your real dataset, run the interactive recorder:
   ```bash
   python3 record_dataset.py
   ```
   Follow the prompts to record Script A (everyday narrative) and/or Script B (technical compiler/SIMD text). The script automatically outputs `slice_30s`, `slice_60s`, `slice_120s`, and `slice_180s` along with exact matching reference text files.

2. **Execute the Full Benchmark Matrix:**
   Once your audio slices are recorded:
   ```bash
   python3 run_benchmark.py --script all
   ```
   Or evaluate a specific domain/slice:
   ```bash
   python3 run_benchmark.py --script non_technical --slices slice_30s slice_60s slice_120s slice_180s
   ```
   As each model finishes inference, the harness immediately writes:
   - `logs/latest/<Model>_<Format>_<Quant>/<dataset>_<slice>_transcript.txt`
   - `logs/latest/<Model>_<Format>_<Quant>/<dataset>_<slice>_reference.txt`
   - `logs/latest/<Model>_<Format>_<Quant>/<dataset>_<slice>_metrics.json` (WER, accuracy, substitutions, deletions, insertions, latency, RTF)
   - `logs/latest/<Model>_<Format>_<Quant>/results.json`
   - Live updates to `logs/latest/BENCHMARK_SUMMARY.md` and `logs/latest/benchmark_results.json`

3. **Verification Status:**
   - The end-to-end pipeline was rigorously tested on a sample recording across all 9 models.
   - All 9 models executed without errors, exits, or missing dependencies.
   - Real-time incremental file writes, Levenshtein error breakdowns, and report formatting were 100% verified.

---

## 6. Essential Operational Commands

Download all benchmark models directly into `models/`:
```bash
python3 download_models.py
```

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
