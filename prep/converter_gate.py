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
# prep/converter_gate.py — toy#204 regression gate for
# prep/convert_smollm2_to_gguf.py.
#
# Builds a tiny llama-family checkpoint directory (2 layers, d 32, random
# weights) the way transformers 5 saves one, and converts it with
# --model-dir. Nothing is downloaded and no fixture is committed.
#
# Legs:
#   rope5     config with only rope_parameters.rope_theta: the GGUF carries
#             that base (it was silently 10000), and toy-infer loads the
#             file and reports the same base
#   rope4     top-level rope_theta still read
#   norope    neither form: the converter refuses and writes no file
#   override  --rope-theta states the base when the config has none
#   scaling   rope_parameters with a scaling type emits llama.rope.scaling.*;
#             an unsupported type is refused
#   shards    model.safetensors.index.json + shards gives the same file as
#             the single-file checkpoint
#   embd      --quantize q8_0 keeps token_embd F32 and quantizes
#             output.weight; --quantize-embd quantizes token_embd too and
#             toy-infer decodes from it; --quantize-embd alone is refused
#   tokenizer --with-tokenizer reads tokenizer.json and chat_template.jinja
#             from the directory
#
#   uv run --script prep/converter_gate.py     # exit 0 pass, 1 fail, 2 unusable
#
# TOY_GATE_RUNNER=<toy-infer> overrides the runner.

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import gguf
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parent.parent
CONVERTER = ROOT / "prep" / "convert_smollm2_to_gguf.py"
RUNNER = os.environ.get("TOY_GATE_RUNNER") or str(ROOT / "libexec" / "toy-infer")

V, D, HEADS, KV, LAYERS, FF = 64, 32, 4, 2, 2, 48
DH = D // HEADS
THETA = 1000000.0

BASE_CFG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": V,
    "max_position_embeddings": 512,
    "hidden_size": D,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV,
    "num_hidden_layers": LAYERS,
    "intermediate_size": FF,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
}


def weights():
    rng = np.random.default_rng(204)

    def m(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    t = {
        "model.embed_tokens.weight": m(V, D),
        "model.norm.weight": np.ones(D, dtype=np.float32),
        "lm_head.weight": m(V, D),
    }
    for li in range(LAYERS):
        p = f"model.layers.{li}"
        t[f"{p}.input_layernorm.weight"] = np.ones(D, dtype=np.float32)
        t[f"{p}.post_attention_layernorm.weight"] = np.ones(D, dtype=np.float32)
        t[f"{p}.self_attn.q_proj.weight"] = m(HEADS * DH, D)
        t[f"{p}.self_attn.k_proj.weight"] = m(KV * DH, D)
        t[f"{p}.self_attn.v_proj.weight"] = m(KV * DH, D)
        t[f"{p}.self_attn.o_proj.weight"] = m(D, D)
        t[f"{p}.mlp.gate_proj.weight"] = m(FF, D)
        t[f"{p}.mlp.up_proj.weight"] = m(FF, D)
        t[f"{p}.mlp.down_proj.weight"] = m(D, FF)
    return t


def make_dir(root: Path, name: str, rope: dict, shards: bool = False) -> Path:
    d = root / name
    d.mkdir()
    cfg = dict(BASE_CFG)
    cfg.update(rope)
    (d / "config.json").write_text(json.dumps(cfg))
    t = weights()
    if not shards:
        save_file(t, str(d / "model.safetensors"))
    else:
        names = list(t)
        half = len(names) // 2
        parts = {"model-00001-of-00002.safetensors": names[:half],
                 "model-00002-of-00002.safetensors": names[half:]}
        wmap = {}
        for fn, ns in parts.items():
            save_file({n: t[n] for n in ns}, str(d / fn))
            for n in ns:
                wmap[n] = fn
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": wmap}))
    return d


def add_tokenizer(d: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {f"t{i}": i for i in range(V - 2)}
    vocab["<|a|>"] = V - 2
    vocab["<|b|>"] = V - 1
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.add_special_tokens(["<|a|>", "<|b|>"])
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|b|>"}))
    (d / "chat_template.jinja").write_text("TEMPLATE-204 {{ messages }}")


def convert(model_dir: Path, out: Path, *flags):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    p = subprocess.run(
        [sys.executable, str(CONVERTER), "--model-dir", str(model_dir),
         "--ggml-native", "--out", str(out), *flags],
        capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def kv(path: Path, key: str):
    f = gguf.GGUFReader(str(path)).fields.get(key)
    if f is None:
        return None
    return f.contents()


def tensor_types(path: Path) -> dict:
    return {t.name: t.tensor_type.name for t in gguf.GGUFReader(str(path)).tensors}


def infer(path: Path):
    env = dict(os.environ, GGUF=str(path), PROMPT_IDS="1 2 3 4", N_NEW="4")
    p = subprocess.run([RUNNER], capture_output=True, text=True, env=env)
    out = p.stdout + p.stderr
    ids = ""
    for line in out.splitlines():
        if line.startswith("ids: "):
            ids = line[5:].strip()
    return p.returncode, ids, out


failures = []


def ok(leg: str, what: str) -> None:
    print(f"  ok: {leg} — {what}")


def fail(leg: str, what: str) -> None:
    failures.append(f"{leg}: {what}")


def last(out: str) -> str:
    lines = [l for l in out.strip().splitlines() if l.strip()]
    return lines[-1][:160] if lines else "(no output)"


def main() -> int:
    if not os.environ.get("TOY_GATE_RUNNER"):
        b = subprocess.run(["make", "-C", str(ROOT), "libexec/toy-infer"],
                           capture_output=True, text=True)
        if b.returncode != 0:
            print("converter_gate: could not build libexec/toy-infer", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory(prefix="toy204-") as tmp:
        root = Path(tmp)
        v5 = {"rope_parameters": {"rope_theta": THETA, "rope_type": "default"}}

        # ---- rope5 ----
        d5 = make_dir(root, "v5", v5)
        g5 = root / "v5.gguf"
        rc, out = convert(d5, g5)
        if rc != 0:
            fail("rope5", f"conversion failed: {last(out)}")
        else:
            base = kv(g5, "llama.rope.freq_base")
            if base != THETA:
                fail("rope5", f"llama.rope.freq_base is {base}, config says {THETA}")
            else:
                rc, ids, o = infer(g5)
                n_ids = len(ids.split())
                if rc != 0 or n_ids != 8:
                    fail("rope5", f"toy-infer did not decode the file: {last(o)}")
                elif f"rope_base={THETA}" not in o:
                    fail("rope5", f"toy-infer does not report rope_base={THETA}")
                else:
                    ok("rope5", f"rope_parameters.rope_theta reaches the GGUF and the engine ({THETA})")

        # ---- rope4 ----
        d4 = make_dir(root, "v4", {"rope_theta": 250000.0, "rope_scaling": None})
        g4 = root / "v4.gguf"
        rc, out = convert(d4, g4)
        base = kv(g4, "llama.rope.freq_base") if rc == 0 else None
        if base != 250000.0:
            fail("rope4", f"top-level rope_theta 250000 gave {base}: {last(out)}")
        else:
            ok("rope4", "top-level rope_theta still read")

        # ---- norope ----
        dn = make_dir(root, "none", {})
        gn = root / "none.gguf"
        rc, out = convert(dn, gn)
        if rc == 0:
            fail("norope", f"converted with no rope base stated (wrote {kv(gn, 'llama.rope.freq_base')})")
        elif "rope" not in out:
            fail("norope", f"failed for another reason: {last(out)}")
        elif gn.exists():
            fail("norope", "refused but left an output file")
        else:
            ok("norope", "no rope base in the config is refused")

        # ---- override ----
        go = root / "override.gguf"
        rc, out = convert(dn, go, "--rope-theta", "5000")
        base = kv(go, "llama.rope.freq_base") if rc == 0 else None
        if base != 5000.0:
            fail("override", f"--rope-theta 5000 gave {base}: {last(out)}")
        else:
            ok("override", "--rope-theta states the base")

        # ---- scaling ----
        dy = make_dir(root, "yarn", {"rope_parameters": {
            "rope_theta": THETA, "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 128}})
        gy = root / "yarn.gguf"
        rc, out = convert(dy, gy)
        got = (kv(gy, "llama.rope.scaling.type"), kv(gy, "llama.rope.scaling.factor"),
               kv(gy, "llama.rope.scaling.original_context_length")) if rc == 0 else None
        du = make_dir(root, "unk", {"rope_parameters": {
            "rope_theta": THETA, "rope_type": "dynamic", "factor": 2.0}})
        rc_u, out_u = convert(du, root / "unk.gguf")
        if got != ("yarn", 4.0, 128):
            fail("scaling", f"yarn fields in rope_parameters gave {got}: {last(out)}")
        elif kv(g5, "llama.rope.scaling.type") is not None:
            fail("scaling", "rope_type default emitted a scaling type")
        elif rc_u == 0 or "dynamic" not in out_u:
            fail("scaling", f"unsupported rope_type was not refused: {last(out_u)}")
        else:
            ok("scaling", "rope_parameters scaling fields emitted; unsupported type refused")

        # ---- shards ----
        ds = make_dir(root, "shards", v5, shards=True)
        gs = root / "shards.gguf"
        rc, out = convert(ds, gs)
        if rc != 0:
            fail("shards", f"conversion failed: {last(out)}")
        elif not g5.exists() or gs.read_bytes() != g5.read_bytes():
            fail("shards", "sharded checkpoint converts to a different file")
        else:
            ok("shards", "sharded directory gives the single-file bytes")

        # ---- embd ----
        gq = root / "q8.gguf"
        ge = root / "q8e.gguf"
        rc_q, out_q = convert(d5, gq, "--quantize", "q8_0")
        rc_e, out_e = convert(d5, ge, "--quantize", "q8_0", "--quantize-embd")
        rc_x, out_x = convert(d5, root / "x.gguf", "--quantize-embd")
        if rc_q != 0 or rc_e != 0:
            fail("embd", f"conversion failed: {last(out_q if rc_q else out_e)}")
        else:
            tq, te = tensor_types(gq), tensor_types(ge)
            pair_q = (tq["token_embd.weight"], tq["output.weight"])
            pair_e = (te["token_embd.weight"], te["output.weight"])
            rc_i, ids_e, o_i = infer(ge)
            _, ids_q, _ = infer(gq)
            if pair_q != ("F32", "Q8_0"):
                fail("embd", f"--quantize q8_0: token_embd/output are {pair_q}, expected F32/Q8_0")
            elif pair_e != ("Q8_0", "Q8_0"):
                fail("embd", f"--quantize-embd: token_embd/output are {pair_e}, expected Q8_0/Q8_0")
            elif ge.stat().st_size >= gq.stat().st_size:
                fail("embd", "--quantize-embd did not shrink the file")
            elif rc_i != 0 or len(ids_e.split()) != 8:
                fail("embd", f"toy-infer did not decode the quantized-embedding file: {last(o_i)}")
            elif rc_x == 0:
                fail("embd", "--quantize-embd without --quantize was accepted")
            else:
                same = "same ids as F32 embeddings" if ids_e == ids_q else "ids differ from F32 embeddings (random weights)"
                ok("embd", f"token_embd Q8_0 with --quantize-embd, F32 without; decodes ({same})")

        # ---- tokenizer ----
        add_tokenizer(d5)
        gt = root / "tok.gguf"
        rc, out = convert(d5, gt, "--with-tokenizer")
        if rc != 0:
            fail("tokenizer", f"conversion failed: {last(out)}")
        else:
            toks = kv(gt, "tokenizer.ggml.tokens")
            tmpl = kv(gt, "tokenizer.chat_template")
            eos = kv(gt, "tokenizer.ggml.eos_token_id")
            if toks is None or len(toks) != V or toks[V - 1] != "<|b|>":
                fail("tokenizer", "vocabulary not read from the directory's tokenizer.json")
            elif tmpl is None or "TEMPLATE-204" not in tmpl:
                fail("tokenizer", "chat_template.jinja not embedded")
            elif eos != V - 1:
                fail("tokenizer", f"eos id is {eos}, expected {V - 1}")
            else:
                ok("tokenizer", "tokenizer.json and chat_template.jinja read from the directory")

    if not failures:
        print("GATE PASS [converter]: a local checkpoint converts with its own rope base, "
              "and token_embd can be quantized")
        return 0
    for m in failures:
        print(f"  FAIL: {m}")
    print(f"GATE FAIL [converter]: {len(failures)} failure(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
