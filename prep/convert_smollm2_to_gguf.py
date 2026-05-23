#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub",
#   "safetensors",
#   "gguf>=0.10",
#   "numpy",
#   "tokenizers",
# ]
# ///
#
# Convert HF SmolLM2-135M (or any llama-family model with the same
# architecture: TinyLlama, SmolLM2-360M, Qwen2.5-0.5B with biases off)
# to a project-native GGUF in the layout `Toy::SmolLM2` expects.
#
# HF Llama layout (nn.Linear stores [out, in], so every projection
# needs a transpose to land in our [in, out] convention):
#
#   model.embed_tokens.weight                            [V, D]      (no transpose)
#   model.norm.weight                                    [D]
#   model.layers.N.input_layernorm.weight                [D]
#   model.layers.N.post_attention_layernorm.weight       [D]
#   model.layers.N.self_attn.q_proj.weight  [n_heads * Dh,    D]
#   model.layers.N.self_attn.k_proj.weight  [n_kv    * Dh,    D]
#   model.layers.N.self_attn.v_proj.weight  [n_kv    * Dh,    D]
#   model.layers.N.self_attn.o_proj.weight  [D, D]
#   model.layers.N.mlp.gate_proj.weight     [d_ff, D]
#   model.layers.N.mlp.up_proj.weight       [d_ff, D]
#   model.layers.N.mlp.down_proj.weight     [D, d_ff]
#
# Output GGUF tensor names follow llama.cpp convention:
#   token_embd.weight, output_norm.weight,
#   blk.N.{attn_norm, attn_q, attn_k, attn_v, attn_output,
#          ffn_norm, ffn_gate, ffn_up, ffn_down}.weight
#
# No tokenizer is embedded — tokenization happens host-side
# (prep/encode_smollm2.py) and IDs go through the Ruby model as
# integers. v1 simplification; the converter writes weights only.

import argparse
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import gguf


REPO_ID = "HuggingFaceTB/SmolLM2-135M"
OUT     = Path("data/smollm2-135m-f32.gguf")
CACHE   = Path("prep/_hf_cache")


def is_quantizable(name: str, shape) -> bool:
    """Quantize 2-D weight tensors. Keep embeddings + 1-D norms + 1-D
    biases in f32: embeddings are gathered (no matmul savings to quantize);
    norms are tiny and quantizing hurts more than it saves."""
    if len(shape) < 2:
        return False
    if "embd" in name and "weight" in name:
        return False           # token_embd, position_embd
    if "output_norm" in name or "attn_norm" in name or "ffn_norm" in name:
        return False
    return True


def embed_tokenizer(w: "gguf.GGUFWriter", repo_id: str, cache: str) -> None:
    """Pull the model's tokenizer from HF and embed vocab + merges +
    special-token IDs + chat template into the GGUF.

    Covers tiktoken-style byte-level BPE (Llama-3, Qwen2/3, SmolLM2):
    these all store their vocab + merges in `tokenizer.json` under
    `model.vocab` and `model.merges`. SentencePiece tokenizers
    (Mistral, TinyLlama) use a different on-disk layout; falls back
    to extracting vocab via the HF tokenizers Python API.
    """
    from tokenizers import Tokenizer
    import tempfile

    print(f"  embedding tokenizer from {repo_id}…")
    # Locate the tokenizer.json (downloads on first run, cached after).
    tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json",
                                cache_dir=cache)
    tok = Tokenizer.from_file(tok_path)
    vocab = tok.get_vocab()        # dict[str, int]
    inv_vocab = sorted(vocab.items(), key=lambda kv: kv[1])
    tokens   = [k for k, _ in inv_vocab]
    n_vocab  = len(tokens)

    # Token type tags. 1=NORMAL, 3=CONTROL/SPECIAL. Used by llama.cpp
    # to skip special tokens during text decode.
    added_ids = set()
    if hasattr(tok, "get_added_tokens_decoder"):
        # Returns dict[int, AddedToken] keyed by token id.
        added_ids = set(tok.get_added_tokens_decoder().keys())
    token_type = [3 if i in added_ids else 1 for i in range(n_vocab)]

    # Merges. Try to read from tokenizer.json's model.merges first;
    # fall back to empty if not present (SentencePiece path).
    with open(tok_path) as f:
        tjson = json.load(f)
    raw_merges = tjson.get("model", {}).get("merges", [])
    # Newer HF tokenizers format stores merges as [["a", "b"], ...];
    # older versions as ["a b", ...]. Normalize to "a b" strings.
    merges = []
    for m in raw_merges:
        if isinstance(m, list):
            merges.append(f"{m[0]} {m[1]}")
        else:
            merges.append(m)

    # T1.3: detect SentencePiece vs byte-level BPE. The two algorithms
    # differ in their space marker (▁ vs Ġ) and byte-fallback
    # convention. We look at vocab[3] — for SPM-style vocabs that's
    # the first byte-fallback token "<0x00>"; for GPT-2 byte-level it
    # would be a meaningful merge-derived token.
    is_spm = (n_vocab > 3 and tokens[3] == "<0x00>")
    tokenizer_kind = "llama" if is_spm else "gpt2"
    w.add_tokenizer_model(tokenizer_kind)

    # Pre-tokenizer hint. "llama-bpe" = cl100k_base-style regex
    # (Llama-3 / Qwen / SmolLM2 / Qwen3). "default" = SentencePiece
    # metaspace replacement (Llama-1/2 / Mistral / TinyLlama).
    pre_kind = "default" if is_spm else "llama-bpe"
    w.add_string("tokenizer.ggml.pre", pre_kind)
    print(f"    tokenizer kind = {tokenizer_kind} ({pre_kind})")

    w.add_token_list(tokens)
    w.add_token_types(token_type)
    if merges:
        w.add_token_merges(merges)
    print(f"    vocab={n_vocab} merges={len(merges)}")

    # Special token IDs. tok.token_to_id returns None if absent.
    def get_id(name: str):
        return tok.token_to_id(name)

    # Read from tokenizer config first (canonical source); fall back
    # to common names.
    tcfg_path = None
    try:
        tcfg_path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json",
                                     cache_dir=cache)
    except Exception:
        pass

    bos_id = eos_id = pad_id = unk_id = None
    chat_template = None
    if tcfg_path:
        with open(tcfg_path) as f:
            tcfg = json.load(f)
        chat_template = tcfg.get("chat_template")
        for key, setter in (
            ("bos_token", lambda v: v),
            ("eos_token", lambda v: v),
            ("pad_token", lambda v: v),
            ("unk_token", lambda v: v),
        ):
            tok_obj = tcfg.get(key)
            if isinstance(tok_obj, dict):
                tok_str = tok_obj.get("content")
            else:
                tok_str = tok_obj
            tid = get_id(tok_str) if tok_str else None
            if key == "bos_token": bos_id = tid
            elif key == "eos_token": eos_id = tid
            elif key == "pad_token": pad_id = tid
            elif key == "unk_token": unk_id = tid

    if bos_id is not None: w.add_bos_token_id(bos_id)
    if eos_id is not None: w.add_eos_token_id(eos_id)
    if pad_id is not None: w.add_pad_token_id(pad_id)
    if unk_id is not None: w.add_unk_token_id(unk_id)
    if chat_template:
        w.add_chat_template(chat_template)
    print(f"    bos={bos_id} eos={eos_id} pad={pad_id} unk={unk_id} "
          f"chat_template={'yes' if chat_template else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--out",     default=str(OUT))
    ap.add_argument("--cache",   default=str(CACHE))
    ap.add_argument("--quantize", choices=["", "q8_0", "q4_0", "q5_0"],
                    default="",
                    help="Quantize 2-D weight tensors (legacy quantizers via "
                         "gguf-py). q8_0 (~4× smaller, ~1% noise), q4_0 "
                         "(~8× smaller, ~3-5% noise), q5_0 (in between). "
                         "Embeddings, biases, and norms stay f32. K-quants "
                         "(Q4_K_M etc.) need ggml's C-side quantizer.")
    ap.add_argument("--ggml-native", action="store_true",
                    help="Write 2D linear weights in HF-native [out, in] "
                         "row-major (no transpose). The resulting bytes "
                         "match ggml's column-major ne=[in, out] layout "
                         "directly, enabling zero-copy mmap loading. The "
                         "loader picks this path automatically via the "
                         "'toy.ggml_native' metadata key. NOT compatible "
                         "with the old transposed loader path.")
    ap.add_argument("--with-tokenizer", action="store_true",
                    help="Embed the model's tokenizer (vocab + merges + "
                         "special token IDs + chat template) into the "
                         "GGUF. Required for Ruby-side encoding/decoding "
                         "via lib/tokenizer.rb; without this flag, callers "
                         "must tokenize via prep/llama_tokens.py and pass "
                         "token IDs to the model.")
    args = ap.parse_args()

    qtype = None
    qmap = {"q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0,
            "q5_0": gguf.GGMLQuantizationType.Q5_0}
    if args.quantize:
        qtype = qmap[args.quantize]

    os.makedirs(args.cache,            exist_ok=True)
    os.makedirs(Path(args.out).parent, exist_ok=True)

    print(f"[1/4] downloading {args.repo_id} → {args.cache}")
    cfg_path = hf_hub_download(args.repo_id, "config.json",       cache_dir=args.cache)
    # Larger models (≥3B) shard weights across multiple safetensors
    # with a model.safetensors.index.json that lists which key lives
    # in which shard. Try single-file first, fall back to sharded.
    try:
        sft_path = hf_hub_download(args.repo_id, "model.safetensors",
                                    cache_dir=args.cache)
        shard_paths = [sft_path]
    except Exception:
        idx_path = hf_hub_download(args.repo_id,
                                    "model.safetensors.index.json",
                                    cache_dir=args.cache)
        with open(idx_path) as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        print(f"      sharded: {len(shards)} files ({', '.join(shards)})")
        shard_paths = [hf_hub_download(args.repo_id, s,
                                         cache_dir=args.cache)
                        for s in shards]

    with open(cfg_path) as f:
        cfg = json.load(f)
    n_vocab     = cfg["vocab_size"]
    n_ctx       = cfg["max_position_embeddings"]
    n_embd      = cfg["hidden_size"]
    n_head      = cfg["num_attention_heads"]
    n_kv        = cfg["num_key_value_heads"]
    n_layer     = cfg["num_hidden_layers"]
    n_ff        = cfg["intermediate_size"]
    rope_theta  = float(cfg.get("rope_theta", 10000.0))
    rms_eps     = float(cfg.get("rms_norm_eps", 1e-5))
    # M1.1: prefer explicit head_dim when HF sets it (Qwen3 does:
    # head_dim=128 with hidden_size=1024 / num_heads=16 = 64 — the
    # computed value is half the actual). Fall back to the computed
    # value for models that don't set the field (SmolLM2, Llama-3.x,
    # Qwen2.5 all match hidden_size/num_heads).
    d_head      = int(cfg.get("head_dim", n_embd // n_head))
    tie_embed   = bool(cfg.get("tie_word_embeddings", False))
    # rope_scaling.* propagation. HF config carries one of:
    #   {"rope_type":"llama3", "factor":32.0, "low_freq_factor":1.0,
    #    "high_freq_factor":4.0, "original_max_position_embeddings":8192}
    #   {"rope_type":"linear", "factor":2.0}
    #   {"rope_type":"yarn",   "factor":..., "attn_factor":..., ...}
    # We emit the matching llama.rope.scaling.* GGUF keys so the Ruby
    # loader (SmolLM2ConfigLoader.read_rope_scaling) can dispatch.
    rope_scaling = cfg.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        # HF uses both "type" (older) and "rope_type" (newer Llama-3).
        rs_kind = rope_scaling.get("rope_type") or rope_scaling.get("type")
    else:
        rs_kind = None

    print(f"      vocab={n_vocab} ctx={n_ctx} d={n_embd} d_head={d_head} "
          f"heads={n_head} n_kv={n_kv} layers={n_layer} d_ff={n_ff} "
          f"rope_base={rope_theta} rms_eps={rms_eps} "
          f"rope_scaling={rs_kind or 'none'}")

    # safe_open in numpy framework chokes on bfloat16 (SmolLM2's storage
    # dtype). Read raw bytes, then decode bf16 → f32 manually: bf16 is
    # literally the upper 16 bits of f32, so left-shift uint16 by 16
    # and reinterpret as f32.
    print(f"[2/4] opening {len(shard_paths)} shard(s)")

    def _bf16_to_f32(buf: bytes, shape) -> np.ndarray:
        as_u16 = np.frombuffer(buf, dtype=np.uint16)
        as_u32 = as_u16.astype(np.uint32) << 16
        return as_u32.view(np.float32).reshape(shape)

    from safetensors import deserialize
    blobs = {}
    for sft_path in shard_paths:
        with open(sft_path, "rb") as fh:
            raw_payload = fh.read()
        blobs.update(dict(deserialize(raw_payload)))

    def _load_f32(name: str) -> np.ndarray:
        info = blobs[name]
        dtype, shape, buf = info["dtype"], info["shape"], info["data"]
        if dtype == "BF16":
            arr = _bf16_to_f32(buf, shape)
        elif dtype in ("F32", "F16"):
            np_dt = {"F32": np.float32, "F16": np.float16}[dtype]
            arr = np.frombuffer(buf, dtype=np_dt).reshape(shape).astype(np.float32)
        else:
            raise RuntimeError(f"unsupported dtype: {dtype} for {name}")
        return np.ascontiguousarray(arr)

    def take(name: str) -> np.ndarray:
        return _load_f32(name)

    # nn.Linear convention: weight.shape == [out, in]. The original Mat
    # path needed [in, out], which costs a transpose here. The native
    # path (--ggml-native) preserves [out, in] — ggml's column-major
    # ne=[in, out] reads those bytes directly without fixup, so we can
    # mmap rather than copy.
    def take_T(name: str) -> np.ndarray:
        if args.ggml_native:
            return _load_f32(name)
        return np.ascontiguousarray(_load_f32(name).T)

    print(f"[3/4] writing GGUF → {args.out}")
    w = gguf.GGUFWriter(args.out, "llama")

    w.add_context_length(n_ctx)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_block_count(n_layer)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv)
    w.add_rope_freq_base(rope_theta)
    w.add_rope_dimension_count(d_head)
    # M1.1: explicit head_dim. Mirrors llama.cpp's convention of
    # `llama.attention.key_length` (and `value_length`, equal for
    # all models we target). Qwen3 needs this because head_dim is
    # NOT hidden_size/num_heads.
    w.add_uint32("llama.attention.key_length", d_head)
    w.add_uint32("llama.attention.value_length", d_head)
    w.add_layer_norm_rms_eps(rms_eps)
    # rope_scaling.* emission. Skipped silently when the HF config has
    # no rope_scaling block (SmolLM2, Qwen2.5-short-ctx, etc.).
    if rs_kind in ("linear", "yarn", "llama3"):
        w.add_string("llama.rope.scaling.type", rs_kind)
        rs_factor = float(rope_scaling["factor"])
        w.add_float32("llama.rope.scaling.factor", rs_factor)
        orig_max = rope_scaling.get("original_max_position_embeddings")
        if orig_max is not None:
            w.add_uint32("llama.rope.scaling.original_context_length", int(orig_max))
        if rs_kind == "llama3":
            w.add_float32("llama.rope.scaling.low_freq_factor",
                          float(rope_scaling.get("low_freq_factor", 1.0)))
            w.add_float32("llama.rope.scaling.high_freq_factor",
                          float(rope_scaling.get("high_freq_factor", 4.0)))
        elif rs_kind == "yarn":
            w.add_float32("llama.rope.scaling.attn_factor",
                          float(rope_scaling.get("attn_factor", 1.0)))
            w.add_float32("llama.rope.scaling.beta_fast",
                          float(rope_scaling.get("beta_fast", 32.0)))
            w.add_float32("llama.rope.scaling.beta_slow",
                          float(rope_scaling.get("beta_slow", 1.0)))
    w.add_file_type(gguf.LlamaFileType.ALL_F32)  # file_type metadata is informational
    w.add_uint32("llama.vocab_size", n_vocab)
    # Layout flag. When set, the loader knows 2D linear weights are in
    # HF-native [out, in] row-major and can be mmap'd / memcpy'd into
    # ggml ne=[in, out] without transpose.
    if args.ggml_native:
        w.add_bool("toy.ggml_native", True)

    # Embed tokenizer (vocab + merges + special-token IDs + chat
    # template) when --with-tokenizer. Required for Ruby-side
    # encode/decode via lib/tokenizer.rb.
    if args.with_tokenizer:
        embed_tokenizer(w, args.repo_id, args.cache)

    # Quantize 2-D weight tensors when --quantize is set. add() wraps
    # w.add_tensor so quantization is transparent at the call sites.
    def add(name: str, data: np.ndarray):
        if qtype is not None and is_quantizable(name, data.shape):
            try:
                quantized_bytes = gguf.quantize(data, qtype)
                w.add_tensor(name, quantized_bytes, raw_dtype=qtype)
                return
            except (gguf.quants.QuantError, ValueError) as e:
                print(f"      [skip-quant {name} {data.shape}: {e}]")
        w.add_tensor(name, data)

    # Globals
    add("token_embd.weight",  take("model.embed_tokens.weight"))   # [V, D]
    add("output_norm.weight", take("model.norm.weight"))           # [D]

    # Untied output projection (`lm_head.weight`) when present.
    # SmolLM2 doesn't have it (tied embeddings). TinyLlama and other
    # llama-family models that don't tie do — we transpose to [D, V]
    # to match our [in, out] Mat convention, but actually we keep the
    # HF orientation [V, D] so the Ruby side can do matmul_t against
    # x_final the same way as the tied case (token_embed.weight is
    # also [V, D] and we already matmul_t against it).
    # M1.1: tie_word_embeddings handling. Some models (Qwen3) ship an
    # lm_head.weight tensor in their safetensors for compat with
    # frameworks that don't auto-tie, but the HF config sets
    # tie_word_embeddings=true. Trust the config flag, not the tensor
    # presence — emitting output.weight in tied mode would give us
    # two copies and the inference path would diverge from HF.
    if "lm_head.weight" in blobs and not tie_embed:
        print(f"      lm_head.weight present (untied embeddings)")
        add("output.weight", take("lm_head.weight"))               # [V, D]
    elif tie_embed:
        print(f"      tie_word_embeddings=true (skipping lm_head)")

    # Qwen2 / Qwen2.5 has biases on q_proj / k_proj / v_proj (o_proj
    # has none). Detect once via block 0; the architecture is uniform
    # so all blocks have the same shape.
    has_qkv_bias = (
        f"model.layers.0.self_attn.q_proj.bias" in blobs and
        f"model.layers.0.self_attn.k_proj.bias" in blobs and
        f"model.layers.0.self_attn.v_proj.bias" in blobs
    )
    if has_qkv_bias:
        print("      Q/K/V biases present (Qwen2.x convention)")

    # Per-block
    for li in range(n_layer):
        hf  = f"model.layers.{li}"
        out = f"blk.{li}"

        # RMSNorms (1-D, no transpose)
        add(f"{out}.attn_norm.weight", take(f"{hf}.input_layernorm.weight"))
        add(f"{out}.ffn_norm.weight",  take(f"{hf}.post_attention_layernorm.weight"))

        # Attention projections (all transposed: HF stores [out, in])
        #   q_proj: HF [n_heads*Dh, D]  →  ours [D, n_heads*Dh]   (= [D, D] for SmolLM2)
        #   k_proj: HF [n_kv*Dh,   D]  →  ours [D, n_kv*Dh]
        #   v_proj: HF [n_kv*Dh,   D]  →  ours [D, n_kv*Dh]
        #   o_proj: HF [D, D]          →  ours [D, D]
        add(f"{out}.attn_q.weight",      take_T(f"{hf}.self_attn.q_proj.weight"))
        add(f"{out}.attn_k.weight",      take_T(f"{hf}.self_attn.k_proj.weight"))
        add(f"{out}.attn_v.weight",      take_T(f"{hf}.self_attn.v_proj.weight"))
        add(f"{out}.attn_output.weight", take_T(f"{hf}.self_attn.o_proj.weight"))

        # Qwen2.x: Q/K/V biases — 1-D, no transpose. The loader keeps
        # the f32 storage (biases are tiny next to the weights).
        if has_qkv_bias:
            add(f"{out}.attn_q.bias", take(f"{hf}.self_attn.q_proj.bias"))
            add(f"{out}.attn_k.bias", take(f"{hf}.self_attn.k_proj.bias"))
            add(f"{out}.attn_v.bias", take(f"{hf}.self_attn.v_proj.bias"))

        # M1: Qwen3 added per-head QK-norm (RMSNorm on Q and K with a
        # shared [d_head] gamma applied BEFORE RoPE). HF tensors live
        # at self_attn.q_norm.weight / k_norm.weight; ggml convention
        # is attn_q_norm / attn_k_norm. Only emit when present (Qwen2.x
        # and Llama don't have these).
        if f"{hf}.self_attn.q_norm.weight" in blobs:
            add(f"{out}.attn_q_norm.weight", take(f"{hf}.self_attn.q_norm.weight"))
            add(f"{out}.attn_k_norm.weight", take(f"{hf}.self_attn.k_norm.weight"))

        # SwiGLU FFN (all transposed)
        #   gate_proj / up_proj: HF [d_ff, D]  →  ours [D, d_ff]
        #   down_proj:           HF [D, d_ff] →  ours [d_ff, D]
        add(f"{out}.ffn_gate.weight", take_T(f"{hf}.mlp.gate_proj.weight"))
        add(f"{out}.ffn_up.weight",   take_T(f"{hf}.mlp.up_proj.weight"))
        add(f"{out}.ffn_down.weight", take_T(f"{hf}.mlp.down_proj.weight"))

    print(f"[4/4] finalising")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = os.path.getsize(args.out)
    print(f"done — {args.out} ({sz / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
