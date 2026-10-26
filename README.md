# RoPE / YaRN — Attention Geometry

Explore actual query/key vectors from all 16 query heads in layer 0 of Qwen3-0.6B.

**Live site:** https://slinkoigor.github.io/rope-yarn-lab/

- Inspect 64 rotary coordinate pairs, before and after RoPE or YaRN.
- Change token positions, frequency base, YaRN parameters, and temperature scaling live.
- Compare attention probabilities over an editable phrase, with and without temperature correction.
- Try the experimental shared Q = K control and switch attention heads.

## Runtime

All calculations run in browser JavaScript. Tokenizer data and small BF16 checkpoint ranges are fetched directly from [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), pinned to revision `c1899de289a04d12100db370d81485cdf75e47ca`. No API key, GPU, or server is required. The first visit needs internet access to Hugging Face.

The head dimension is 128, using Qwen's 64 split-half rotary pairs. The visualization uses float64 calculations from BF16 weights. The attention panel normalizes over the entered key tokens only, excluding Q's own key and unspecified intervening tokens. YaRN is an experimental geometric transformation applied without fine-tuning.

## Run locally

Serve this folder with any static HTTP server, for example:

```sh
python3 -m http.server 8000
```

Open http://localhost:8000. Opening index.html directly as a local file does not support ES module loading.

## Deployment

GitHub Pages serves the root of the `main` branch. Updating the public files and pushing to `main` publishes a new version.
