# Flash Streaming

Run models that don't fit in RAM on a 16 GB Mac. Full 4-bit quality, no mmap thrashing.

## Measured Results

All numbers measured on a Mac mini M4 (16 GB). Nothing estimated.

| Model | Type | Total Size | RAM Used | Speed | Status |
|-------|------|-----------|---------|-------|--------|
| Qwen3-32B | Dense | 18.4 GB | 4.5 GB | 0.15 tok/s | Working, coherent |
| Qwen3.5-27B | Dense hybrid (SSM/attention) | 16.1 GB | 5.5 GB | 0.18 tok/s | Working, coherent |
| Qwen3.5-35B-A3B | MoE (256 experts) | 22 GB | 1.42 GB | 1.54 tok/s | Working, interactive agent |
| Qwen3.5-35B-A3B | MoE batched verify (K=8) | 22 GB | 1.42 GB | 5.1 tok/s | Proven, research prototype |

For comparison: llama.cpp with mmap on the 18.4 GB model gets 0.017 tok/s (swap thrashing). The 2-bit IQ2_M gets 6 tok/s but degrades after ~60 tokens.

## The Idea

LLMs have two kinds of weights:
- **Attention** (~23% of params): needed for every token, small enough to fit in RAM
- **FFN** (~77% of params): needed once per layer per token, can be streamed

Pin attention in RAM. Stream FFN from SSD. Discard after each layer. Memory never grows.

For MoE models, only 8 of 256 experts activate per token. Instead of streaming the full FFN layer (~460 MB), stream only the 8 active experts (~14 MB). That's why MoE is 10x faster.

## How to Run

### 35B MoE Agent (1.54 tok/s)

Interactive agent with web search, shell commands, and reasoning. 22 GB model, 1.42 GB RAM.

**Requires pre-built stream files.** You need the 35B model split into pinned + per-layer expert files. To build them:

```bash
# Download the MLX model (~22 GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3.5-35B-A3B-4bit', local_dir='\$HOME/models/qwen35-35b-a3b-mlx-4bit')
"

# Split into streaming format
python3 split_mlx_model.py

# Run the agent
python3 moe_agent.py
```

### 27B Dense (0.18 tok/s)

```bash
pip3 install mlx-lm transformers --break-system-packages

# Download (~16 GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3.5-27B-4bit', local_dir='\$HOME/models/qwen35-27b-mlx-4bit')
"

# Split into streaming format
python3 split_dense_27b.py

# Run
python3 flash_stream_27b.py
```

### 32B Dense (0.15 tok/s) — the original proof

This was the first model we proved the method on. Requires GGUF conversion (more complex).

```bash
# Requires Qwen3-32B GGUF (Q4_K_M) already downloaded
python3 convert_split.py       # GGUF → split safetensors
python3 convert_aligned.py     # safetensors → 16KB-aligned binary (for v2)
python3 flash_stream.py        # v1: mmap streaming
python3 flash_stream_v2.py     # v2: F_NOCACHE direct I/O (faster)
```

### Batched Union-of-Experts (5.1 tok/s)

Research prototype. Verifies 8 draft tokens in one forward pass by computing the set union of active experts across all positions (~27 unique experts per layer, not 64). Not interactive — this is verification speed for speculative decoding.

```bash
python3 batched_moe.py
```

## Research Timeline

### Phase 1: Dense streaming (32B)

**Problem:** Qwen3-32B at MLX 4-bit is 18.4 GB. Doesn't fit in 16 GB. llama.cpp mmap thrashes at 0.017 tok/s.

**v1 — mmap streaming** (`flash_stream.py`): Split model, load FFN per layer via `mx.load()`. 0.12 tok/s. Proved the architecture works.

**v2 — F_NOCACHE direct I/O** (`flash_stream_v2.py`, `direct_io.py`): Bypass macOS Unified Buffer Cache with `fcntl(F_NOCACHE)` + `os.pread()`. 16KB-aligned binary format for DART IOMMU. 0.152 tok/s — 27% faster than mmap.

**Batched eval experiment** (`flash_stream_batched.py`): Tested 8-layer batches (16 evals vs 128). Zero speedup. Proved the bottleneck is SSD I/O, not GPU sync overhead.

### Phase 2: MoE Expert Sniper (35B)

**Insight:** MoE models activate only 8 of 256 experts per token. Instead of streaming the full FFN layer, read only the active experts from SSD.

**Expert I/O** (`expert_io.py`): 8-thread `F_NOCACHE` + `pread` reader. Lazy FD open, 16KB alignment, bfloat16 handling. Saturates NVMe queue depth.

**MoE engine** (`flash_moe.py`): Router predicts active experts → `expert_io` loads them → `gather_qmm` computes the weighted expert output in one fused call.

**Working agent** (`moe_agent.py`): Full interactive agent — web search via DuckDuckGo, shell commands, chain-of-thought. 1.54 tok/s at 1.42 GB RAM. **This is the main deliverable.**

### Phase 3: Batched MoE (5.1 tok/s)

**Union-of-Experts** (`batched_moe.py`): Given K=8 draft tokens, compute which experts each token needs across all layers. Take the set union — typically ~27 unique experts per layer, not 64. Load those 27 once, run `gather_qmm` for all K positions.

Result: 5.1 tok/s for K=8 token verification. This is the path to fast speculative decoding — if we can get a good draft model (the 0.8B draft gave α=0.31, too low).

### Phase 4: 27B Dense hybrid (proving generality)

**Different architecture:** Qwen3.5-27B is a dense hybrid model with 64 layers, 3:1 linear_attention:full_attention ratio, and GatedDeltaNet SSM layers. No MoE.

**Same technique works** (`flash_stream_27b.py`, `split_dense_27b.py`): Pin attention + SSM weights (~5.5 GB), stream dense FFN per layer (~165 MB × 64). 0.18 tok/s — proves Flash Streaming is not architecture-specific.

## Key Discoveries

Things we learned the hard way. Each one cost hours of debugging.

1. **GGUF is column-major.** The correct reshape is `flat.reshape(ne[1], ne[0])`, not `flat.reshape(ne[0], ne[1]).T`. The latter gives correct shapes but garbage output — we got "зезезе" for hours before finding this. (`dequant_gguf.py`, `convert_split.py`)

2. **MLX 4-bit is 15% larger than you'd expect.** Scales + biases at group_size=64 add overhead: 0.156 bytes/param, not 0.125. A 32B model is 18.4 GB, not 16 GB. This is why the model doesn't fit in 16 GB RAM even at 4-bit. (Measured during conversion)

3. **`nn.quantize()` silently skips MoE experts.** `SwitchLinear` is not a subclass of `nn.Linear`. You must pass a `class_predicate` that explicitly includes it. Without this, expert weights stay in float16 and produce garbage. (`moe_agent.py`)

4. **`gather_qmm` eliminates accumulator divergence.** Running 8 separate `quantized_matmul` calls compounds rounding errors across 40 layers. One batched `gather_qmm` call matches the reference model. (`batched_moe.py`, `flash_moe.py`)

5. **F_NOCACHE is 27% faster than mmap** for sequential streaming. macOS Unified Buffer Cache adds overhead for read-once workloads. `fcntl(F_NOCACHE)` + `os.pread()` with 16KB alignment bypasses it. (`direct_io.py`, `expert_io.py`)

6. **`setattr` on `nn.Module` leaks memory.** Injecting FFN weights into the model tree via `setattr` prevents garbage collection — memory grew 3.6 GB per 16 layers. Fix: use `mx.quantized_matmul` directly on loaded arrays, never touch the model tree. (`flash_stream.py`)

7. **Batching layers doesn't help dense streaming.** Tested 8-layer batches (16 evals vs 128). Zero speedup. The bottleneck is SSD I/O latency, not GPU eval overhead. (`flash_stream_batched.py`)

8. **Speculative decoding needs a matched draft.** 0.8B draft for 35B target gives α=0.31 (31% acceptance). Architecture works but not worth it without α > 0.70. Same-model-different-quant (IQ2_M draft → Q4_K_M target) is the promising direction.

## Known Limitations

- **Dense streaming is slow.** 0.15-0.18 tok/s. Each token reads 10-14 GB of FFN from SSD. Even at 5 GB/s NVMe, theoretical max is ~0.35 tok/s. Useful for proving the method, not for interactive use.
- **MoE streaming is usable but not fast.** 1.54 tok/s is fine for thinking tasks, not for rapid interaction.
- **Batched MoE is verification only.** 5.1 tok/s requires draft tokens — you can't generate at this speed without a good draft model.
- **Prefill is slow.** Every prompt token runs the full streaming pipeline. A 20-token prompt takes 30-60 seconds.
- **No neuron-level sparsity.** We load entire FFN layers for dense models. The Apple "LLM in a Flash" paper predicts active neurons to load 10-20%. We haven't implemented this.

## Files

| File | What it does |
|------|-------------|
| **Agents & engines** | |
| `moe_agent.py` | Working 35B MoE interactive agent (1.54 tok/s) |
| `flash_moe.py` | MoE streaming engine with gather_qmm fusion |
| `flash_stream.py` | v1 dense streaming engine (mmap, 0.12 tok/s) |
| `flash_stream_v2.py` | v2 dense streaming engine (F_NOCACHE, 0.15 tok/s) |
| `flash_stream_27b.py` | 27B dense hybrid streaming (0.18 tok/s) |
| `flash_agent.py` | 32B dense streaming agent (early version) |
| **I/O** | |
| `expert_io.py` | 8-thread F_NOCACHE expert reader for MoE |
| `direct_io.py` | F_NOCACHE + pread for dense FFN layers |
| **Model splitting** | |
| `split_mlx_model.py` | Split 35B MoE MLX model into pinned + experts |
| `split_dense_27b.py` | Split 27B dense MLX model into pinned + FFN |
| `convert_split.py` | GGUF → split safetensors (for 32B) |
| `convert_aligned.py` | Safetensors → 16KB-aligned binary (for v2) |
| `rebuild_pinned.py` | Rebuild pinned weights from MLX golden model |
| **Research** | |
| `batched_moe.py` | Batched Union-of-Experts verification (5.1 tok/s) |
| `flash_stream_batched.py` | Batched eval experiment (proved eval isn't bottleneck) |
| `dequant_gguf.py` | Custom Q4_K/Q6_K block dequantization (numpy) |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4, 16 GB RAM)
- Python 3.11+
- `mlx`, `mlx-lm`, `transformers`, `rich`
- For MoE: `ddgs` (web search)
- For 32B GGUF conversion: `gguf` package
- 20-40 GB free disk for split model files
