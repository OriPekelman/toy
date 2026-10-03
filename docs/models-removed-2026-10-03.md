# Models removed 2026-10-03 (disk reclaim)

28 GGUF files, 194.9 GiB, removed from `data/` to reclaim disk.

All were **unreferenced by every gate, smoke and runner** and unread since
2026-06-27 (relatime, so the dates are real reads). None is regenerable by
committed tooling: only `prep/convert_smollm2_to_gguf.py` and
`convert_distilgpt2_to_gguf.py` exist, so re-acquiring these means fetching the
source below and redoing the conversion with whatever produced them originally.

**15 of them are referenced by `bench/`, `demos/` or `docs/`** — those
benchmarks and demos will not run until the file is re-fetched. They are marked
below. `docs/models.md`'s validated-parity table also rests on several of these;
its claims stand as recorded but cannot be re-verified locally without a
re-download, which that document already notes happening once before
(Qwen3-1.7B Q8).

| file | bytes | GiB | used by | likely source |
|---|---:|---:|---|---|
| `qwen25-7b-f32.gguf` | 30462485856 | 28.37 | nothing | `Qwen/Qwen2.5-7B` |
| `mistral-7b-instruct-v0.2-tok.gguf` | 28968472608 | 26.98 | nothing | `mistralai/Mistral-7B-Instruct-v0.2` |
| `qwen3-4b-tok.gguf` | 16095822400 | 14.99 | nothing | `Qwen/Qwen3-4B` |
| `llama-3.2-3b-f32.gguf` | 12851014912 | 11.97 | nothing | `meta-llama/Llama-3.2-3B` |
| `qwen25-3b-native.gguf` | 12343780064 | 11.50 | bench/demos/docs | `Qwen/Qwen2.5-3B` |
| `qwen25-3b-f32.gguf` | 12343780032 | 11.50 | bench/demos/docs | `Qwen/Qwen2.5-3B` |
| `qwen25-7b-native-q8.gguf` | 9693521280 | 9.03 | bench/demos/docs | `Qwen/Qwen2.5-7B` |
| `qwen25-7b-q8_0.gguf` | 9693521248 | 9.03 | bench/demos/docs | `Qwen/Qwen2.5-7B` |
| `mistral-7b-instruct-v0.2.gguf` | 8080164256 | 7.53 | bench/demos/docs | `mistralai/Mistral-7B-Instruct-v0.2` |
| `OLMoE-1b-7b-0924-Instruct-q8_0.gguf` | 7359943040 | 6.85 | bench/demos/docs | `allenai/OLMoE-1B-7B-0924-Instruct` |
| `qwen3-1.7b-tok.gguf` | 6888244704 | 6.42 | nothing | `Qwen/Qwen3-1.7B` |
| `llama-3.2-1b-tok.gguf` | 4951084576 | 4.61 | bench/demos/docs | `meta-llama/Llama-3.2-1B` |
| `llama-3.2-1b-f32-rs.gguf` | 4943267008 | 4.60 | nothing | `meta-llama/Llama-3.2-1B` |
| `llama-3.2-1b-f32.gguf` | 4943266752 | 4.60 | bench/demos/docs | `meta-llama/Llama-3.2-1B` |
| `tinyllama-1.1b-tok.gguf` | 4401775008 | 4.10 | bench/demos/docs | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `tinyllama-1.1b-native.gguf` | 4401774944 | 4.10 | bench/demos/docs | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `tinyllama-1.1b-f32.gguf` | 4400205952 | 4.10 | bench/demos/docs | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `qwen25-3b-native-q8.gguf` | 4193592032 | 3.91 | nothing | `Qwen/Qwen2.5-3B` |
| `qwen25-3b-q8_0.gguf` | 4193592000 | 3.91 | nothing | `Qwen/Qwen2.5-3B` |
| `gemma-2-2b-it-Q8_0.gguf` | 2784495456 | 2.59 | nothing | `google/gemma-2-2b-it` |
| `qwen3-0.6b-tok.gguf` | 2390144480 | 2.23 | nothing | `Qwen/Qwen3-0.6B` |
| `qwen25-1.5b-native-q8.gguf` | 2326177120 | 2.17 | bench/demos/docs | `Qwen/Qwen2.5-1.5B` |
| `qwen25-1.5b-q8_0.gguf` | 2326177088 | 2.17 | nothing | `Qwen/Qwen2.5-1.5B` |
| `qwen25-0.5b-tok.gguf` | 1982072224 | 1.85 | bench/demos/docs | `Qwen/Qwen2.5-0.5B` |
| `qwen25-0.5b-f32.gguf` | 1976148096 | 1.84 | bench/demos/docs | `Qwen/Qwen2.5-0.5B` |
| `smollm2-360m-native.gguf` | 1449070304 | 1.35 | nothing | `HuggingFaceTB/SmolLM2-360M` |
| `smollm2-360m-f32.gguf` | 1447302240 | 1.35 | nothing | `HuggingFaceTB/SmolLM2-360M` |
| `tinyllama-1.1b-q8_0.gguf` | 1361596544 | 1.27 | bench/demos/docs | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

## What was kept, and why

`data/*.gguf` referenced by `Makefile`/`prep`/`lib`/`bin`, everything under 1 GiB,
and the five **symlinks** into `~/.cache/huggingface` (the MoE arch fixtures:
Qwen3-30B-A3B, DeepSeek-V2-Lite, Qwen1.5-MoE, OLMoE-Q4_K_M, stories15M). Also
kept: the 139-175 byte **header-only GGUF stubs**, which are arch-detection
fixtures the gates read, not truncated downloads.

Verified after removal: `gates-framework`, `gate-prereq`, `gate-consumer`,
`gate-truck-plant`, `gate-truck-lane`, and the model-reading gates
(`deepseek-mla`, `qwen3moe`, `qwen2moe-shexp`, `moe-kquant`, `cpu`).

