# LLM Inference Optimization Pipeline 🚀

> End-to-end inference optimization of transformer models using 
> ONNX Runtime and INT8 quantization — achieving 1.94x speedup 
> over FP32 GPU baseline while reducing model size by 74.9%

## Results

| Method | Latency (mean) | P95 Latency | Model Size | vs Baseline |
|--------|---------------|-------------|------------|-------------|
| PyTorch FP32 GPU | 6.510ms | 7.485ms | 267.9MB | baseline |
| ONNX Runtime CPU | 8.001ms | 8.104ms | 267.9MB | 1.23x slower |
| **INT8 Quantized** | **3.349ms** | **3.589ms** | **67.3MB** | **1.94x faster** ✅ |

**Hardware:** NVIDIA L4 GPU (23.7GB) | **Model:** DistilBERT (66.9M params)

## Key Achievements
- ⚡ **1.94x latency improvement** over FP32 GPU baseline
- 📦 **74.9% model size reduction** (267.9MB → 67.3MB)
- 🖥️ **Optimized CPU outperforms unoptimized GPU**
- 💰 Lower inference cost — no GPU required after optimization

## Optimization Pipeline

### 1. Baseline Profiling
Profiled PyTorch FP32 model using `torch.profiler`:
- Identified `aten::addmm` (matrix multiply) = **84.89% of CUDA time**
- 380 linear layer calls dominating inference
- Established rigorous benchmark: 100 runs + 10 warmup, P50/P95/P99

### 2. FP16 Half Precision
- Result: **Slower** (6.51ms → 6.31ms)
- Insight: FP16 conversion overhead outweighs bandwidth savings for small models
- Real engineering lesson: **always profile, never assume**

### 3. ONNX Export
- Exported via `torch.onnx` with opset 18
- Applied constant folding and graph optimization
- Eliminated Python runtime overhead entirely

### 4. INT8 Dynamic Quantization
- Converted FP32 weights (32-bit) → INT8 (8-bit)
- 4x memory reduction
- Final result: **3.349ms — 1.94x faster than GPU baseline**

## What I Learned
1. Always profile before optimizing — never guess the bottleneck
2. Hardware-specific optimizations don't always generalize
3. INT8 quantization on CPU can outperform FP32 on GPU
4. Model size reduction directly impacts inference cost at scale
5. P95/P99 latency matters as much as mean in production

## Tech Stack
```
PyTorch 2.10 · ONNX Runtime 1.24 · Hugging Face Transformers
CUDA 12.8 · torch.profiler · INT8 Quantization
Google Colab L4 GPU · Python 3.12
```

## Part of ML Systems Optimization Suite
This is **Module 1** of an ongoing series:
- ✅ Module 1 — Inference Optimization (ONNX + Quantization)
- 🔜 Module 2 — CUDA Kernel Optimization (Triton + Custom Kernels)
- 🔜 Module 3 — Distributed Training (FSDP + NCCL)
- 🔜 Module 4 — TensorRT Optimization
- 🔜 Module 5 — Agentic AI Systems

## Author
**Piyush Kunjilwar**
MS Information Systems — Northeastern University (May 2026)
[LinkedIn](https://linkedin.com/in/piyush-kunjilwar) · 
[GitHub](https://github.com/piyush12kunjilwar) · 
[Portfolio](https://piyush12kunjilwar.github.io)# LLM-Inference-Optimization
