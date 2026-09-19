# Speech-to-Text Model Benchmark & Handy Catalog Scrutiny

**Date:** September 19, 2026  
**Environment:** Intel Core i5-10310U CPU @ 1.70GHz (4 cores / 8 threads, AVX2) | Intel UHD Graphics 620 (CometLake-U GT2, 24 EUs)  
**Audio Sample:** 49.78s continuous reading of GPU microarchitecture & `llama.cpp` kernel optimization passage  
**Audio File:** [`logs/attempt_20260919_111852/audio.wav`](file:///home/rehan-10xe/Documents/chats/logs/attempt_20260919_111852/audio.wav)

---

## 1. Executive Summary & Benchmark Matrix

This benchmark evaluated candidate models headlessly against a ground-truth **Golden Reference** passage. Accuracy was measured using industry-standard **Word Error Rate (WER)** via Levenshtein edit distance:
$$\text{Word Accuracy} = 100\% - \text{WER}$$

Speed was evaluated via **Real-Time Factor (RTF)**:
$$\text{RTF} = \frac{\text{Audio Duration}}{\text{Processing Time}}$$

| Model | Size | Engine / Backend | Handy Claimed Accuracy | Empirical Word Accuracy | Empirical WER | Handy Claimed Speed | Empirical Speed (RTF) | Empirical Latency | Suitability Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parakeet TDT-CTC 110M** | **129 MB** | TranscribeCpp (Vulkan0) | 0.85 | 61.7% | 38.3% | **0.98** | **9.5x RT** | **5.2s** | *** 74.9 / 100** [FAST] |
| **Parakeet V2** | 451 MB | ONNX (CPU AVX2) | **0.85** | **82.7%** | **17.3%** | 0.85 | **5.9x RT** | **8.4s** | *** 68.8 / 100** [#1] |
| **Canary 180M Flash (GGUF)** | **208 MB** | TranscribeCpp (Vulkan0) | **0.88** | **80.2%** | **19.8%** | **0.98** | **5.3x RT** | **9.3s** | *** 64.0 / 100** [#2] |
| **Parakeet V3** | 456 MB | ONNX (CPU AVX2) | 0.80 | 70.4% | 29.6% | 0.85 | 4.6x RT | 10.8s | * 55.4 / 100 |
| **Whisper Small (English)** | **257 MB** | TranscribeCpp (Vulkan0) | 0.81 | 72.8% | 27.2% | 0.80 | 2.3x RT | 21.8s | * 34.6 / 100 |
| **Canary 180M Flash (ONNX)** | 146 MB | ONNX (CPU AVX2) | 0.75 | 63.0% | 37.0% | 0.85 | 1.8x RT | 26.8s | * 28.6 / 100 |
| **Moonshine V2 Med** | 192 MB | ONNX (Streaming) | 0.75 | 55.6% | 44.4% | 0.80 | 1.4x RT | 34.3s | * 22.9 / 100 |
| **Whisper Medium** | 469 MB | TranscribeCpp (CPU) | 0.75 | **85.2%** | **14.8%** | 0.60 | 1.1x RT | 43.8s | * 20.1 / 100 |
| **Whisper Medium** | 469 MB | TranscribeCpp (Vulkan0) | 0.75 | **85.2%** | **14.8%** | 0.60 | 1.0x RT | 49.8s | * 17.8 / 100 |

---

## 2. Technical Vocabulary Scorecard

How each model transcribed deep GPU microarchitecture and `llama.cpp` terms:

| Target Term | Golden Reference | Parakeet V2 (ONNX) | Whisper Medium | Canary 180M Flash |
| :--- | :--- | :--- | :--- | :--- |
| **LLM Framework** | `llama.cpp` | *"Lamador CPP"* | *"Lama.CBP"* | *"Lamardot's CVP"* |
| **Attention Kernel** | `fused FlashAttention` | *"fused flash attention"* | *"fused flash retention"* | *"fused flash retention"* |
| **Matrix Multiply** | `quantized GEMM` | *"quantized gem"* | *"quantized gem"* | *"quantized gem"* |
| **Branch Divergence** | `warp divergence` | *"WARC divergence"* | *"Warp divergence"* [OK] | *"work divergence"* |
| **Core Hardware Unit** | `Streaming Multiprocessor` | *"stream multi processor"* | *"streaming multiprocessor"* [OK] | *"stream multiprocessor"* |
| **Memory Access** | `uncoalesced` | *"uncoalescent"* | *"uncollapsed"* | *"uncoalesced"* [OK] |
| **Cache Behavior** | `L2 cache thrashing` | *"L2 cache thrashing"* [OK] | *"L2 cache thrashing"* [OK] | *"L two cash thrashing"* |
| **Compute Engines** | `systolic tensor cores` | *"stolic tensor cores"* | *"systolic tensor cores"* [OK] | *(omitted)* |
| **Execution Model** | `SIMT execution units` | *"SIMT execution units"* [OK] | *"SIE empty execution units"* | *"SIMT execution units"* [OK] |

---

## 3. Discrepancies: Handy Catalog Claims vs. Physical Hardware

### A. The Canary 180M Speed Inflation (3.3× Slower than Claimed)
* **Handy Claim:** Speed score of `0.85` (tied with Parakeet).
* **Reality:** Canary took **26.8s–27.9s (1.8x RT)** vs. Parakeet’s **8.4s (5.9x RT)**.
* **Root Cause:** Canary is an **Autoregressive Encoder-Decoder**. For a 50-second passage (~130 subword tokens), it must execute 130 sequential ONNX sessions one token at a time. On CPU, single-token generation collapses into GEMV operations, which are strictly bottlenecked by DDR4 memory bandwidth (~35 GB/s) and dispatch overhead.

### B. Parakeet V3 vs. V2 Paradox
* **Handy Claim:** Parakeet V3 is flagged as `Recommended`, while V2 is unflagged.
* **Reality:** **Parakeet V2 outperformed V3 across all metrics**:
  * Word Accuracy: V2 achieved **82.7%** vs. V3's **70.4%**.
  * Speed: V2 achieved **8.4s (5.9x RT)** vs. V3's **10.8s (4.6x RT)**.
* **Root Cause:** Parakeet V3 has altered token duration predictions that caused phonetic hallucinations on technical words (`gem cards` instead of `gem kernels`, `uncle as the` instead of `uncoalesced`).

### C. Whisper Medium Accuracy Underestimation
* **Handy Claim:** Accuracy score of `0.75` (tied for lowest).
* **Reality:** Whisper Medium achieved the **highest word recognition accuracy (85.2%)**, flawlessly transcribing *"systolic tensor cores"* and *"streaming multiprocessor"*.
* **Trade-off:** Whisper ran at only **1.0x–1.1x real-time (43.8s–49.8s)**, making it too sluggish for real-time dictation on integrated Intel graphics.

---

## 4. Top 10 High-Accuracy Candidates in Handy Catalog (To Test Next)

Querying Handy's complete catalog of 68 English models reveals the top models by catalog accuracy (`accuracy_score`):

| Rank | Model Name | Catalog ID | Claimed Accuracy | Claimed Speed | Size | Architecture / Engine | Why It Is Worth Testing |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | **Cohere Transcribe** | `handy-computer/cohere-transcribe-03-2026-gguf/...` | **0.92** | 0.63 | 1.68 GB | TranscribeCpp (GGUF) | Highest catalog accuracy; state-of-the-art multilingual vocabulary. |
| **2** | **Granite Speech 4.1 2B NAR** | `handy-computer/granite-speech-4.1-2b-nar-gguf/...` | **0.92** | 0.37 | 1.70 GB | TranscribeCpp (Non-Autoregressive) | IBM's Non-Autoregressive architecture; avoids step-by-step token latency. |
| **3** | **Granite Speech 4.1 2B** | `handy-computer/granite-speech-4.1-2b-gguf/...` | **0.92** | 0.30 | 1.74 GB | TranscribeCpp | IBM's standard autoregressive foundation speech model. |
| **4** | **Parakeet TDT 1.1B** | `handy-computer/parakeet-tdt-1.1b-gguf/...` | **0.91** | **0.76** | 892 MB | TranscribeCpp (TDT) | **Top Recommendation:** Scaled-up version of Parakeet V2. Maintains fast TDT inference while doubling parameter count for accuracy. |
| **5** | **Parakeet RNN-T 1.1B** | `handy-computer/parakeet-rnnt-1.1b-gguf/...` | **0.91** | **0.75** | 892 MB | TranscribeCpp (RNN-T) | Classic RNN-Transducer variant for comparison against TDT. |
| **6** | **Granite Speech 4.0 1B** | `handy-computer/granite-4.0-1b-speech-gguf/...` | **0.91** | 0.31 | 1.74 GB | TranscribeCpp | Compact 1B parameter Granite model. |
| **7** | **Parakeet RNN-T 0.6B** | `handy-computer/parakeet-rnnt-0.6b-gguf/...` | **0.90** | **0.84** | 695 MB | TranscribeCpp (GGUF Q8_0) | GGUF-quantized Parakeet running in `transcribe-cpp` rather than ONNX. |
| **8** | **Canary 1B Flash** | `handy-computer/canary-1b-flash-gguf/...` | **0.90** | **0.83** | 733 MB | TranscribeCpp | 1B parameter Canary running in quantized GGUF format. |
| **9** | **Whisper Large v3 Turbo** | `handy-computer/whisper-large-v3-turbo-gguf/...` | **0.88** | 0.35 | 845 MB | TranscribeCpp (GGUF Q8_0) | **Top Recommendation:** Cuts Whisper decoder from 32 layers to 4 layers while keeping the Large encoder. Expected to beat Whisper Medium's accuracy with lower latency. |
| **10** | **Parakeet TDT 0.6B v2 (GGUF)** | `handy-computer/parakeet-tdt-0.6b-v2-gguf/...` | **0.89** | **0.85** | 695 MB | TranscribeCpp (GGUF Q8_0) | Direct test of our winning Parakeet V2 architecture running under GGML instead of ONNX. |

---

## 5. Curated Recommendations to Test Next

For your specific hardware (Intel Core i5-10310U with integrated graphics), the three most high-yield models to download and benchmark next are:

1. **`Parakeet TDT 1.1B` (892 MB)**  
   * *Hypothesis:* Will likely surpass Parakeet V2's 82.7% word accuracy while preserving the 3x–5x real-time speed advantage of the TDT architecture.
2. **`Whisper Large v3 Turbo` (845 MB)**  
   * *Hypothesis:* Retains Whisper's superior domain vocabulary recognition (systolic tensor cores, streaming multiprocessor) while running significantly faster than Whisper Medium due to its 4-layer pruned decoder.
3. **`Cohere Transcribe` (1.68 GB)**  
   * *Hypothesis:* Tests the absolute highest accuracy model in Handy's catalog (`0.92`) to determine if its precision warrants the ~1.7 GB memory footprint.
