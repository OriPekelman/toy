# M3 — sequence-mode forward graph for llama-family

**Status:** Design — implementation queued behind this doc's sign-off.
**Date:** 2026-05-21
**Tasks:** #69 (M3 reusable decode graph), #75 (sequence-mode forward).
**Predecessor:** F1.2 step 6b shipped (persistent Adam, commit `1a1374b`).
**Unblocks:** F1.2 step 6c (varied prefixes), 6d (real alpaca + masked CE).

## Why now

F1.2 step 6b shipped: multi-position SFT runs without NaN'ing because
Adam state is persistent. But every position-switch still tears down
graph_b and re-allocs sched (~6 ms build + a few hundred ms sched-alloc
per switch). At 32 positions per example × 1000 examples, that's
prohibitive — the per-step rebuild dominates.

The fix is a forward graph that processes ALL T positions in ONE compute:
- `T` token IDs + `T` positions in.
- Full T×T causal attention (no KV-cache views).
- Output: `T` logit vectors.
- Loss: masked CE aggregated across positions (prompt-token positions
  masked to weight 0; target positions weighted 1.0).
- One forward + backward + opt_step per example, regardless of T.

This is the `FullForwardFFICache` shape (already proven for GPT-2; see
`project_m1_full_forward_shipped_2026_05_14`), extended to the
llama-family architecture.

## What changes from GPT2FullForwardFFICache

| Op            | GPT-2                          | llama-family                   |
|---            |---                             |---                             |
| Norm          | LayerNorm (γ + β)              | RMSNorm (γ only)               |
| FFN           | gelu(W₁·x + b₁) → W₂· + b₂     | down(silu(gate·x) · up·x)      |
| Pos. embed    | learned, sliced into x_embed   | RoPE on Q + K inline           |
| KV grouping   | MHA (n_kv == n_heads)          | GQA (n_kv ≤ n_heads)           |
| Biases        | Q/K/V/O/FFN all have bias      | usually none (Qwen2.x: QKV)    |
| LM head       | tied                           | tied OR untied (TinyLlama)     |

Every other primitive (matmul, add, concat, softmax, diag_mask_inf,
get_rows) is shared.

## File layout

```
lib/llama_seq_forward_ffi.rb               new — class + module
lib/llama_seq_forward_ffi_cuda.rb          new — CUDA mirror (later)
demos/smollm2_seq_parity.rb                new — T=1 parity vs decode_step
demos/smollm2_seq_parity_t4.rb             new — T=4 parity vs N decode_steps
docs/design/m3-seq-forward-2026-05-21.md   this doc
```

CUDA mirror is deferred to step 2; the CPU class proves the shape first.
Step 1 = CPU forward parity. No training-graph wiring yet (that's step
3, depends on persistent Adam already shipped).

## Class shape (CPU)

```ruby
class LlamaSeqBlockFFI
  attr_accessor :t_rn1_gamma, :t_rn2_gamma,
                :t_w_q, :t_w_k, :t_w_v, :t_w_o,
                :t_b_q, :t_b_k, :t_b_v,     # optional (Qwen2.x)
                :t_w_gate, :t_w_up, :t_w_down
end

class LlamaSeqForwardFFICache
  attr_accessor :sess, :t_token_embed, :t_final_norm_gamma,
                :t_output, :has_untied_output, :has_qkv_bias,
                :blocks_ffi,
                :t_seq, :d_model, :d_ff, :n_heads, :n_kv, :d_head,
                :group_size, :n_layers, :vocab_size,
                :rope_base, :rms_eps, :realized,
                :t_token_ids, :t_positions,
                :t_x_embed, :t_x_final, :t_logits,
                :gguf_handle_keepalive

  # Allocate persistent weights (mmap'd from gguf, same layout as
  # SmolLM2KVFFICache#realize_for_mmap), per-T compute inputs, full
  # forward graph. T_seq is fixed; rebuild for a different T_seq.
  def realize_for_mmap(gguf, cfg, t_seq, untied, qkv_bias)
    # ... weight tensors via tnn_input_2d_persistent_mmap (same offsets
    #     and per-head layout as the KV cache class)
    # ... NO K/V cache buffers (full attention over T)
    # ... tnn_finalize_weights
    # ... compute inputs:
    #       @t_token_ids = TinyNN.tnn_input_1d_i32(sess, t_seq)
    #       @t_positions = TinyNN.tnn_input_1d_i32(sess, t_seq)
    # ... forward graph:
    #       t_embedded = tnn_get_rows(token_embed, ids)  # ne=[d_model, T]
    #       per-block:
    #         t_h = rms_norm(x, rn1_gamma, eps)
    #         per-Q-head:
    #           q_pre = matmul(w_q[h], h) (+ b_q if qkv_bias) # ne=[d_head, T]
    #           q     = rope_ext(q_pre, positions, d_head, rope_base)
    #         per-KV-head:
    #           k_pre = matmul(w_k[h], h) (+ b_k)
    #           k     = rope_ext(k_pre, positions, d_head, rope_base)
    #           v_raw = matmul(w_v[h], h) (+ b_v)            # ne=[d_head, T]
    #         per-Q-head h_q (with GQA: kv_h = h_q / group_size):
    #           scores = matmul(k[kv_h], q[h_q])   # ne=[T, T]
    #           scaled = scale(scores, 1/sqrt(d_head))
    #           masked = diag_mask_inf(scaled, 0)
    #           attn   = softmax(masked)
    #           # head_out shape ne=[d_head, T]
    #           # v has shape ne=[d_head, T]; need transpose to ne=[T, d_head]
    #           v_t    = transpose(v[kv_h])  # OR use weight-first matmul
    #           head_out = matmul(v_t, attn)
    #         concat heads along ne0 → ne=[d_model, T]
    #         out_proj = matmul(w_o, concat) (+ b_o if needed)
    #         x_attn   = x + out_proj
    #         h2       = rms_norm(x_attn, rn2_gamma, eps)
    #         gate     = matmul(w_gate, h2)
    #         up       = matmul(w_up,   h2)
    #         silu_g   = silu(gate)
    #         hidden   = mul(silu_g, up)
    #         down     = matmul(w_down, hidden)
    #         x_out    = x_attn + down
    #       final:
    #         t_x_final = rms_norm(t_x, final_norm_gamma, eps)
    #         t_logits  = matmul(token_embed OR t_output, t_x_final)  # ne=[vocab, T]
    # ... tnn_realize(sess, t_logits)
  end
end
```

## Risks / decisions

1. **V transpose for sequence-mode head_out**: in KV-decode we
   transpose-store V into `t_V[T, d_head]` so `matmul(t_V, attn)` works.
   In sequence-mode there's no persistent V; we compute v fresh each
   step. Easiest: do `v_t = TinyNN.tnn_transpose(...)` once per head.
   Alternative: build the matmul as `matmul(v, attn.T)` if ggml allows
   that shape (it does for matmul; transpose is free).

2. **Causal mask shape**: `tnn_diag_mask_inf(scores, n_past=0)` masks
   the upper triangle of a T×T matrix. n_past=0 is correct for full
   sequence forward (no prior context). Backward smoke confirms.

3. **GQA mapping**: per-Q-head h_q uses kv_h = h_q / group_size. Same
   convention as `SmolLM2KVFFICache#build_decode_step`.

4. **Mmap'd weights**: reuse the same `tnn_input_2d_persistent_mmap`
   per-head allocator as `SmolLM2KVFFICache`. Identical byte offsets;
   the GGUF is the source of truth. No weight duplication if we share
   the gguf_handle (kept alive by `gguf_handle_keepalive`).

5. **Tied vs untied LM head**: tied uses `token_embed` directly in the
   final matmul; untied uses a separate `output.weight` tensor. Matches
   the existing SmolLM2KVFFICache convention (governed by `untied`
   flag at realize time).

6. **Spinel ivar collision (per `project_m1_full_forward_shipped_2026_05_14`)**:
   the M1 memory flagged that having same-named ivars (`t_seq`,
   `d_model`, `d_ff`) across multiple cache classes degrades Spinel's
   inference. CPU-side this was bench-able as a standalone Ruby class.
   For seq-mode we'll either (a) name distinct (`@seq_t`, `@seq_d_model`)
   or (b) compile a smoke standalone and only later integrate into a
   delegating wrapper. (a) is cleaner. **Decision: use distinct names.**

7. **Sched aliasing on backward (CPU)**: M3 itself is forward-only.
   Training-graph wiring (step 3) brings back the CPU sched-aliasing
   risk from
   [[project_cpu_cuda_lora_train_divergence_2026_05_21]] —
   `tnn_pin_all_graph_b_nodes` is the workaround if it bites the
   long-T training graph.

## Acceptance gates

### Step 1 (this session candidate)

- `LlamaSeqForwardFFICache` compiles under Spinel (no widening warnings
  on the new class).
- `demos/smollm2_seq_parity.rb` at T=1: logits from `LlamaSeqForwardFFICache#forward([id], [pos])`
  match `SmolLM2KVFFICache#decode_step(id, pos)` to within FP32 noise
  (max_abs_diff < 1e-4 on SmolLM2-135M f32 weights).
- Runs on the existing prebuilt GGUF (`data/smollm2-135m-native.gguf`).

### Step 2 (next session)

- `demos/smollm2_seq_parity_t4.rb` at T=4: position-T logits match
  `decode_step(id_T, pos_T)` run after `decode_step(id_0, pos_0)` ...
  `decode_step(id_T-1, pos_T-1)` — proves the full attention matrix.
- CUDA mirror class + smoke.

### Step 3 (the F1.2 6c/6d payoff)

- Wire training graph: `tnn_build_backward` + `opt_step_adamw` over
  the full-sequence loss.
- Masked CE primitive (multiply per-position CE by a `[T]` mask before
  sum).
- Verify trajectory on a small alpaca example matches step 6b's
  multi-position smoke convergence shape (but in ONE forward, not
  T forwards).

## What this doesn't do

- LoRA on the seq-mode graph (Step 4): mechanically identical to KV
  decode — splice `B@A` into the Q matmul per head. Same `enable_lora_q!`
  pattern. Pure addition; doesn't change the seq-mode contract.

- Full fine-tune (F3): training all weights, not just LoRA. Step 5 in
  the larger F1/F3/F4 sequence; doesn't change the M3 graph.

- Q8 quantized base + F32 adapter (F4 / QLoRA): orthogonal to M3.

## Open questions for the user

1. **Step ordering**: ship M3 step 1 (CPU forward parity at T=1)
   this session, then steps 2-3 in follow-ups? Or batch step 1+2
   (CPU forward parity at T=1 AND T=4) before committing?
   Recommended: step 1 only — early integration check, faster signal.

2. **Class naming**: `LlamaSeqForwardFFICache` vs `SeqForwardFFICache`
   vs `TransformerSeqForwardCache`? `Llama` is honest (we're targeting
   llama-family with RMSNorm/SwiGLU/RoPE/GQA) but Qwen2.x and Mistral
   ride along. Recommended: `LlamaSeqForwardFFICache` — names the
   architecture family, leaves room for a future
   `GPT2SeqForwardFFICache` rename of the existing `FullForwardFFICache`.

3. **GGUF for parity smoke**: SmolLM2-135M (smallest, fastest) vs
   TinyLlama-1.1B (untied LM head, covers more of the matrix)?
   Recommended: SmolLM2-135M for step 1, TinyLlama for step 2's T=4
   smoke (untied path needs coverage somewhere).
