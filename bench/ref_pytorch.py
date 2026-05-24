#!/usr/bin/env python3
"""
bench/ref_pytorch.py — the PyTorch "old-stable" reference for toy's
routine self-comparison. NOT trying to be fast or clever; it is the
yardstick we measure toy against to keep design choices honest.

Both workloads use a Llama at SmolLM2-135M dims (random weights — perf
is weight-independent, so no checkpoint/tokenizer needed), matching the
model toy's own gx10 benches use (demos/seq_train_bench_cuda,
demos/qwen25_bench_cuda on smollm2-135m). transformers' LlamaForCausalLM
gives a faithful Llama (RMSNorm/RoPE/GQA/SwiGLU/KV-cache) so we compare
against a real Llama, not an approximation.

Emits `BENCH <metric> <value>` lines (same grammar bench/check*.rb parse):

  train -> BENCH pt_train_ms <ms/step>            full fine-tune step
           (fwd + CE + backward + AdamW), T positions, batch=1, first
           steps warmed up. Matches toy's seq_train_bench Full-FT.

  infer -> BENCH pt_infer_toks_per_sec <tok/s>    KV-cache greedy decode
           BENCH pt_infer_step_ms <ms/tok>        batch=1, one tok/step,
           prefill + warmup excluded. Matches toy's qwen25_bench decode.

Device defaults to cuda (the single-GPU case); cpu|mps work for local
sanity. On gx10 run inside the torch image:
  docker run --rm --gpus all --ipc=host -v "$PWD":/w -w /w \
    gx10/dev-pytorch:latest python3 bench/ref_pytorch.py --workload both
"""
import argparse, time
import torch

# SmolLM2-135M architecture (HF config). Weights are random; we time throughput.
SMOLLM2_135M = dict(vocab_size=49152, hidden_size=576, intermediate_size=1536,
                    num_hidden_layers=30, num_attention_heads=9,
                    num_key_value_heads=3, max_position_embeddings=8192,
                    rope_theta=10000.0, rms_norm_eps=1e-5)


def build_llama(dev):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(**SMOLLM2_135M, tie_word_embeddings=True)
    return cfg, LlamaForCausalLM(cfg).to(dev)


def bench_train(args, dev, sync):
    torch.manual_seed(42)
    cfg, model = build_llama(dev)
    model.train()                                   # full fine-tune: all params trainable
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, cfg.vocab_size, (1, args.train_t), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids).loss
        loss.backward()
        opt.step()

    for _ in range(args.warmup):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    sync()
    print(f"BENCH pt_train_ms {(time.perf_counter() - t0) / args.steps * 1e3}")


def bench_infer(args, dev, sync):
    torch.manual_seed(42)
    cfg, model = build_llama(dev)
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=dev)

    @torch.no_grad()
    def decode(n):
        out = model(prompt, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(-1)
        for _ in range(n):
            out = model(tok, past_key_values=past, use_cache=True)
            past, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)

    decode(args.warmup)                             # prefill + warmup absorbed here
    sync()
    t0 = time.perf_counter()
    decode(args.n_new)
    sync()
    elapsed = time.perf_counter() - t0
    print(f"BENCH pt_infer_toks_per_sec {args.n_new / elapsed}")
    print(f"BENCH pt_infer_step_ms {elapsed * 1e3 / args.n_new}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["train", "infer", "both"], default="both")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_t", type=int, default=4)    # T positions, matches toy seq_train_bench
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--prompt_len", type=int, default=32)  # matches toy qwen25_bench PREFILL_T
    ap.add_argument("--n_new", type=int, default=8)        # matches toy qwen25_bench N_NEW
    args = ap.parse_args()

    dev = torch.device(args.device)
    sync = (torch.cuda.synchronize if dev.type == "cuda"
            else torch.mps.synchronize if dev.type == "mps"
            else (lambda: None))

    if args.workload in ("train", "both"):
        bench_train(args, dev, sync)
    if args.workload in ("infer", "both"):
        bench_infer(args, dev, sync)


if __name__ == "__main__":
    main()
