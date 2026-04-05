# mlx-qwen-mtp: Multi-Token Prediction for Qwen3.5 on Apple Silicon

First MTP inference implementation for Qwen3.5 in Python. Every other framework strips MTP weights on load. We reverse-engineered the architecture and built working inference with speculative decoding.

## Performance

| Configuration | tok/s | Speedup |
|---|---|---|
| Baseline (stock mlx-lm) | 29.5 | 1.00x |
| + Fused Metal kernels | 30.0 | 1.02x |
| + MTP speculative decoding | 42.7 | 1.45x |
| + Fused rms_norm into matmul | **~45** | **~1.52x** |

Measured on M4 Max (128GB, 546 GB/s bandwidth) with Qwen3.5-27B-4bit.

### Kernel Fusion: rms_norm + quantized_matmul

We discovered that **8.6ms per forward pass** is spent on dispatch barriers between RMS norm and quantized matmul kernels. By fusing the norm computation into the matmul kernel's input loading (two-pass: compute RMS, then normalized dot product), we eliminate these barriers. Full MLX metallib integration [saves 10.4ms (30%)](https://github.com/quivent/mlx-fused-qmv) on the matmul pipeline — projecting ~75 tok/s when stacked with MTP.

## Quick Start

### 1. Extract MTP weights

The MTP head weights are present in the HuggingFace checkpoint but ignored by mlx-lm on load. Extract them first:

```python
from src.extract_weights import extract_mtp_weights

extract_mtp_weights(
    model_path="mlx-community/Qwen3.5-27B-4bit",
    output_path="src/mtp_weights.safetensors",
)
```

### 2. Patch the model and generate

```python
import mlx_lm
from src import patch_model, mtp_generate, load_mtp

# Load the base model
model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-27B-4bit")

# Patch GatedDeltaNet layers with fused Metal kernels
patch_model(model)

# Load MTP head
mtp_head = load_mtp(model, weights_path="src/mtp_weights.safetensors")

# Generate with MTP speculative decoding
output = mtp_generate(
    model, tokenizer,
    prompt="Explain quantum computing in simple terms.",
    max_tokens=256,
    mtp_head=mtp_head,
)
print(output)
```

## Architecture

### MTP Head (15 tensors)

The MTP head is a single transformer layer that predicts token t+2 from:
- The main model's last hidden state at position t
- The embedding of the token at position t+1

Architecture:
1. RMSNorm hidden + RMSNorm embedding
2. Concatenate **[embed, hidden]** (not [hidden, embed] -- matches GGUF eh_proj naming)
3. FC projection (10240 -> 5120)
4. One gated-attention transformer layer (same arch as Qwen3.5 attention layers)
5. RMSNorm -> shared lm_head -> logits

Weight tensors:
- `pre_fc_norm_hidden.weight`, `pre_fc_norm_embedding.weight`
- `fc.weight` (10240 -> 5120)
- `input_layernorm.weight`
- `q_proj.weight` (5120 -> 12288), `k_proj.weight` (5120 -> 1024), `v_proj.weight` (5120 -> 1024), `o_proj.weight` (6144 -> 5120)
- `q_norm.weight`, `k_norm.weight`
- `post_attention_layernorm.weight`
- `gate_proj.weight` (5120 -> 17408), `up_proj.weight` (5120 -> 17408), `down_proj.weight` (17408 -> 5120)
- `norm.weight` (final, before shared lm_head)

### How It Works: Split-Recurrence Rollback

Qwen3.5 is a hybrid architecture: 48 DeltaNet (recurrent) layers + 16 attention layers. Speculative decoding requires rollback on draft rejection, which works differently for each:

- **DeltaNet layers**: Recurrent state is saved before speculation and restored on reject (zero-copy -- MLX arrays are immutable)
- **Attention layers**: KV cache offset is decremented by 1 to "un-see" the rejected token

The generation loop:
1. Draft token t+2 via MTP head (cheap: one transformer layer)
2. Verify by running T=2 forward pass with [token_t+1, draft_t+2]
3. **Accept**: keep both tokens (2 tokens per step), pipeline next MTP draft
4. **Reject**: rollback DeltaNet states, trim KV offsets, re-run T=1 with correct token

The async eval pipeline overlaps MTP draft computation with verification, so the accept path adds near-zero latency.

### Where the Time Goes

```
T=1 baseline:         34ms → 1 token    → 29.5 tok/s
T=2 split-recurrence: 38ms → 1.8 tokens → 42.7 tok/s (1.45x)

Breakdown of the 38ms:
  34ms  weight reads (13.7 GB at 546 GB/s — same as T=1, bandwidth bound)
   3ms  MTP head forward (one transformer layer)
   1ms  split-recurrence overhead (48 extra GDN kernel dispatches)

Theoretical with 100% acceptance and zero overhead:
  34ms → 2.0 tokens → 58.8 tok/s (2.0x)

Achieved: 1.45x out of 2.0x maximum
```

Gap between 1.45x and 2.0x (0.55x lost to):
- 4ms step overhead (MTP head + split dispatches): ~15%
- 21% rejection (1 token instead of 2): ~12%
- eval sync + Python loop per step: ~5%

### Future Optimization Potential

**Reduce the 38ms/34ms overhead ratio (currently 1.12x)**:
- Compile the MTP head into the main model's mx.compile graph (eliminate 3ms MTP dispatch)
- Fuse the GDN split into the existing Metal kernels (eliminate 1ms split overhead)
- Target: 34.5ms per step → 1.8/0.0345 = 52.2 tok/s

**Eliminate eval sync + Python loop overhead**:
- Async pipeline: start MTP draft while GPU finishes verification
- Batch multiple steps into one mx.eval call
- Move the accept/reject decision to GPU (argmax + compare as GPU ops)
- Target: save 1-2ms per step

**Improve MTP acceptance rate beyond 79%**:
- The current MTP head is a single transformer layer (265 MB)
- Fine-tuning on the target model's own outputs would improve acceptance significantly
- Training cost is minimal: freeze the main model, train only the MTP head (15 tensors, ~800M params) on next-token prediction from the main model's hidden states
- Even distillation from the main model's logits (no labeled data needed) should push acceptance to 85-90%
- At 90% acceptance: 1.9 tokens/step → ~48 tok/s
- At 95% acceptance: 1.95 tokens/step → ~50 tok/s

**The theoretical ceiling is not 2x — it's O(1) per token**:
- With N draft tokens and 100% acceptance, one T=N+1 forward pass produces N+1 tokens
- Weight reads are bandwidth-bound: T=N costs ~1x (same 13.7 GB, just more output)
- DeltaNet adds N × ~0.02ms per layer = N × 1ms
- At N=8: 34ms + 8ms = 42ms for 9 tokens = 4.7ms/tok = 213 tok/s
- The only limit is MTP accuracy at depth — which degrades because a single MTP layer can't predict 8 tokens ahead
- Multi-layer MTP heads (Qwen3.5 only has 1) would sustain accuracy at greater depth

### Fused Metal Kernels

Two custom Metal kernels accelerate the DeltaNet layers:

- **fused_conv1d_silu**: Combines conv1d + SiLU activation in a single GPU dispatch
- **fused_gdn_step**: Combines RMS norm + scale + gating + beta + state update in one kernel (replaces 6+ separate dispatches)

Additionally, the 4 input projection matmuls per DeltaNet layer are concatenated into 1 fused matmul.

## Requirements

- Python >= 3.10
- mlx >= 0.30
- mlx-lm >= 0.20
- Apple Silicon Mac (M1 or later)

## License

Apache-2.0
