# MLX Backend

MLX-native inference server for Apple Silicon. Drop-in replacement for llama.cpp.

## What Works

- **`mlx_engine.py`** — HTTP server on localhost:8000 with OpenAI-compatible API. Supports Qwen3.5-9B and 35B-A3B models via mlx-lm. Use this when you want native MLX inference instead of llama.cpp.
- **`kv_cache.py`** — Save and load KV cache tensors to disk. Enables resuming conversations without reprocessing the prompt.

## Experiments (not production-ready)

These files explore ideas that haven't been fully validated:

- **`turboquant.py`** — KV cache compression inspired by Google's TurboQuant paper. Implements PolarQuant and group quantization at 2/3/4-bit. Not benchmarked against baseline quality.
- **`paged_inference.py`** — KV cache paging to SSD when GPU memory is full. Checkpoint/resume approach (not true virtual memory). Proof of concept.
- **`tiered_cache.py`** — GPU → SSD → Cloudflare R2 tiered KV cache. The idea: hot context in GPU, warm on SSD, cold on R2. Requires R2 setup, not tested end-to-end.
- **`r2_store.py`** — Cloudflare R2 upload/download for persistent KV cache. Requires R2 credentials. Not tested in production.

## Benchmarks

- **`benchmark.py`** — Compares llama.cpp vs MLX vs MLX+TurboQuant on same prompts.
- **`agent_benchmark.py`** — Compares agent task performance across backends.

These benchmarks exist but we haven't published results. Run them yourself on your hardware.

## Usage

```bash
# Start MLX server (default: 9B model)
python3 mlx/mlx_engine.py

# Use 35B model
python3 mlx/mlx_engine.py --model 35b

# Save context after processing
python3 mlx/mlx_engine.py --save-context my-project

# Resume from saved context
python3 mlx/mlx_engine.py --load-context my-project
```

Then use `agent.py` as normal — it talks to localhost:8000.

## Requirements

- `mlx`, `mlx-lm`, `transformers`
- For R2: `boto3` + Cloudflare R2 credentials
