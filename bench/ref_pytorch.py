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

# Architectures (HF config dicts). Weights are random; perf is weight-
# independent, so we time throughput at the right shape. New arches can
# be added by dropping a dict in ARCHES.
SMOLLM2_135M = dict(vocab_size=49152, hidden_size=576, intermediate_size=1536,
                    num_hidden_layers=30, num_attention_heads=9,
                    num_key_value_heads=3, max_position_embeddings=8192,
                    rope_theta=10000.0, rms_norm_eps=1e-5)

# Qwen2.5-1.5B-Instruct dims (used by heavy LoRA train workload).
QWEN25_1P5B = dict(vocab_size=151936, hidden_size=1536, intermediate_size=8960,
                   num_hidden_layers=28, num_attention_heads=12,
                   num_key_value_heads=2, max_position_embeddings=32768,
                   rope_theta=1000000.0, rms_norm_eps=1e-6)

# Qwen2.5-7B-Instruct dims (used by heavy 7B decode workload).
QWEN25_7B = dict(vocab_size=152064, hidden_size=3584, intermediate_size=18944,
                 num_hidden_layers=28, num_attention_heads=28,
                 num_key_value_heads=4, max_position_embeddings=32768,
                 rope_theta=1000000.0, rms_norm_eps=1e-6)

ARCHES = {
    "smollm2_135m": ("llama",   SMOLLM2_135M, dict(tie_word_embeddings=True)),
    "qwen25_1p5b":  ("qwen2",   QWEN25_1P5B,  dict(tie_word_embeddings=True)),
    "qwen25_7b":    ("qwen2",   QWEN25_7B,    dict(tie_word_embeddings=False)),
}


def build_model(arch, dev, dtype=None):
    kind, dims, extra = ARCHES[arch]
    if kind == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(**dims, **extra)
        model = LlamaForCausalLM(cfg)
    else:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(**dims, **extra)
        model = Qwen2ForCausalLM(cfg)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return cfg, model.to(dev)


class LoRAQ(torch.nn.Module):
    """Hand-rolled LoRA-on-q_proj. Matches toy's enable_lora_q!(r) exactly:
    y = q_proj(x) + (x @ A) @ B, where A is [d_in, r], B is [r, d_out],
    base q_proj is frozen. We avoid peft to sidestep its torchao
    version-pinning (the dev-pytorch image ships an older torchao than
    peft 0.19 wants — and we don't need any peft features here)."""
    def __init__(self, base, r):
        super().__init__()
        self.base = base
        for p in base.parameters(): p.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        dev, dtype = base.weight.device, base.weight.dtype
        self.A = torch.nn.Parameter(torch.zeros(d_in, r, device=dev, dtype=dtype))
        self.B = torch.nn.Parameter(torch.zeros(r, d_out, device=dev, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        return self.base(x) + (x @ self.A) @ self.B


def wrap_lora_q(model, r=8):
    """Replace every q_proj nn.Linear inside the model with LoRAQ(base, r).
    All non-LoRA params get frozen. Matches toy/Spinel-side semantics."""
    for p in model.parameters(): p.requires_grad_(False)
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name == "q_proj" and isinstance(child, torch.nn.Linear):
                setattr(module, name, LoRAQ(child, r))
    return model


def bench_train(args, dev, sync):
    torch.manual_seed(42)
    cfg, model = build_model(args.arch, dev)
    if args.lora:
        model = wrap_lora_q(model, r=args.lora_r)
        # PEFT freezes non-LoRA params; only LoRA params hit the optimizer.
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = list(model.parameters())
    model.train()
    opt = torch.optim.AdamW(trainable, lr=1e-4)
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
    print(f"BENCH {args.metric_prefix}train_ms {(time.perf_counter() - t0) / args.steps * 1e3}")


def bench_infer(args, dev, sync):
    torch.manual_seed(42)
    cfg, model = build_model(args.arch, dev)
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
    print(f"BENCH {args.metric_prefix}infer_toks_per_sec {args.n_new / elapsed}")
    print(f"BENCH {args.metric_prefix}infer_step_ms {elapsed * 1e3 / args.n_new}")


def profile_train(args, dev, sync):
    """torch.profiler over a few train steps. Emits a flat op→ms table
    that's directly comparable to bench/aggregate_trace.rb output."""
    from torch.profiler import profile, record_function, ProfilerActivity
    torch.manual_seed(42)
    cfg, model = build_model(args.arch, dev)
    if args.lora:
        model = wrap_lora_q(model, r=args.lora_r)
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = list(model.parameters())
    model.train()
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    ids = torch.randint(0, cfg.vocab_size, (1, args.train_t), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids).loss
        loss.backward()
        opt.step()

    for _ in range(args.warmup):
        step()
    sync()

    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(args.profile_steps):
            with record_function("step"):
                step()
        sync()

    # Aggregate by op name. For CUDA we use self_device_time_total
    # (kernel-only, excludes children). For CPU we use self_cpu_time_total.
    # Both are in microseconds in this profiler.
    use_cuda = dev.type == "cuda"
    events = prof.key_averages()
    rows = []
    for ev in events:
        if ev.key == "step":
            continue
        total_us = (ev.self_device_time_total if use_cuda else ev.self_cpu_time_total)
        if total_us <= 0:
            continue
        rows.append((ev.key, ev.count, total_us))
    rows.sort(key=lambda r: -r[2])

    total_step_us = sum(t for _, _, t in rows)
    print(f"PROFILE_HEADER op,count,total_us,mean_us,pct")
    for name, count, total_us in rows[:args.profile_top]:
        mean_us = total_us / count if count else 0.0
        pct = 100.0 * total_us / total_step_us if total_step_us else 0.0
        # Escape commas in op names (rare but happens for some aten ops).
        safe = name.replace(",", ";")
        print(f"PROFILE_ROW {safe},{count},{total_us:.0f},{mean_us:.2f},{pct:.2f}")
    print(f"PROFILE_TOTAL steps={args.profile_steps} total_us={total_step_us:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["train", "infer", "both", "profile_train"], default="both")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arch", choices=list(ARCHES.keys()), default="smollm2_135m",
                    help="model arch dims (smollm2_135m | qwen25_1p5b | qwen25_7b)")
    ap.add_argument("--lora", action="store_true",
                    help="wrap model in PEFT LoRA on q_proj (heavy train workload)")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--metric_prefix", default="pt_",
                    help="prefix for BENCH metric names (default pt_; heavy mode uses pt_heavy_)")
    ap.add_argument("--train_t", type=int, default=4)    # T positions, matches toy seq_train_bench
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--prompt_len", type=int, default=32)  # matches toy qwen25_bench PREFILL_T
    ap.add_argument("--n_new", type=int, default=8)        # matches toy qwen25_bench N_NEW
    # Op-mix profiling (used for tracing comparisons, not the routine gate).
    ap.add_argument("--profile_steps", type=int, default=3)
    ap.add_argument("--profile_top", type=int, default=20)
    args = ap.parse_args()

    dev = torch.device(args.device)
    sync = (torch.cuda.synchronize if dev.type == "cuda"
            else torch.mps.synchronize if dev.type == "mps"
            else (lambda: None))

    if args.workload == "profile_train":
        profile_train(args, dev, sync)
        return
    if args.workload in ("train", "both"):
        bench_train(args, dev, sync)
    if args.workload in ("infer", "both"):
        bench_infer(args, dev, sync)


if __name__ == "__main__":
    main()
