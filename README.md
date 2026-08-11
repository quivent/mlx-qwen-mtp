<div align="center">

```
 __  __ _  __  __  
|  \/  | | \ \/ /  
| |\/| | |  \  /   
| |  | | |__/  \   
|_|  |_|____/_/\_\ 
  Q W E N - M T P
```

**First MTP inference implementation for Qwen3.5 in Python.**

*1.45x speedup on Apple Silicon via speculative decoding and fused Metal kernels.*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS-lightgrey.svg?style=for-the-badge&logo=apple)](https://apple.com)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## 📑 Table of Contents
- [🎯 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [⚡ Performance & Kernel Fusion](#-performance--kernel-fusion)
- [🔮 Future Optimizations](#-future-optimizations)
- [📄 License](#-license)

---

## 🎯 Overview

Every other framework strips MTP weights on load. We reverse-engineered the architecture and built working inference with speculative decoding in Python using MLX. 

> [!IMPORTANT]
> **Requirements**: Python >= 3.10, mlx >= 0.30, mlx-lm >= 0.20, Apple Silicon Mac (M1+).

---

## 🚀 Quick Start

### 1. Extract MTP weights
The weights are in the HF checkpoint but ignored by `mlx-lm`. 

```python
from src.extract_weights import extract_mtp_weights

extract_mtp_weights(
    model_path="mlx-community/Qwen3.5-27B-4bit",
    output_path="src/mtp_weights.safetensors",
)
```

### 2. Patch and Generate
```python
import mlx_lm
from src import patch_model, mtp_generate, load_mtp

# Load base model
model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-27B-4bit")

# Patch GatedDeltaNet layers with fused Metal kernels
patch_model(model)

# Load MTP head
mtp_head = load_mtp(model, weights_path="src/mtp_weights.safetensors")

# Generate
output = mtp_generate(
    model, tokenizer,
    prompt="Explain quantum computing in simple terms.",
    max_tokens=256,
    mtp_head=mtp_head,
)
print(output)
```

---

## 🏗️ Architecture

The MTP head is a single transformer layer predicting token t+2 from the hidden state at t and embedding at t+1.

### Split-Recurrence Rollback
Qwen3.5 is a hybrid architecture. Speculative decoding requires rollback on draft rejection:
- **DeltaNet layers**: Recurrent state saved before speculation and restored on reject.
- **Attention layers**: KV cache offset decremented by 1.

The generation loop overlaps MTP draft computation with verification, making the accept path add near-zero latency.

---

## ⚡ Performance & Kernel Fusion

Measured on M4 Max (128GB, 546 GB/s) with Qwen3.5-27B-4bit:

| Configuration | tok/s | Speedup |
|---|---|---|
| Baseline | 29.5 | 1.00x |
| + Fused Metal kernels | 30.0 | 1.02x |
| + MTP spec decoding | 42.7 | 1.45x |
| + Fused rms_norm into matmul | **~45** | **~1.52x** |

> [!TIP]
> Fusing RMS norm and quantized matmul kernels eliminates dispatch barriers, saving 8.6ms per forward pass.

Two custom Metal kernels accelerate DeltaNet layers: `fused_conv1d_silu` and `fused_gdn_step`.

---

## 🔮 Future Optimizations

The theoretical ceiling is **O(1) per token**. 

Path to greater speed:
- Reduce step overhead (compile MTP head into main model's graph).
- Eliminate eval sync + Python loop (async pipelines, batch steps).
- Improve MTP acceptance rate (fine-tuning or distillation).

---

## 📄 License

Apache-2.0
