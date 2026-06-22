# GDN hybrid → llama engine: reintegration path (apply on master/spinelc)

**Status (2026-06-22, union pin `fbb9beb`):** the Phase-5 hybrid trains through a
**dedicated runner** (`lib/toy/run/train_hybrid.rb` → `libexec/toy-train-hybrid`,
`gate-gdn-hybrid`), in its own Spinel compilation unit. Wiring the SAME hybrid into
the shared llama engine (`Toy::LLM::Engine::LlamaSeqEngine#realize_for_random_init`)
so plain `toy train` runs it is **BLOCKED on the union pin**: pulling GDN
alloc/train code into that compilation unit miscompiles the proven byte-exact
attention path (landmine #16 family — same reason `toy-train-lora`/`toy-train-gpt2`
are separate binaries). This doc keeps the re-integration a mechanical re-apply,
not a re-derivation.

## What is ALREADY durable in the main engine path (no re-apply needed)

Committed and byte-exact — the seam is reintegration-ready:
- `LlamaArch`: `seq_layer_kinds` (flat int dispatch array), `seq_gdn_blocks_ffi`
  (monomorphic GDNBlock array), `set_gdn_layer!(idx)` (INT-arg, no array param),
  and `build_forward` dispatches `KIND_GDN` → `seq_gdn_blocks_ffi[li].build_forward`.
- `Toy::LLM::Blocks::GDNBlock` + `Toy::LLM::Primitives::GDN` (`recur_unrolled`,
  `l2_train`, `update_gate_train`) + shims `tnn_sqrt`/`tnn_div`/`tnn_repeat`.
- The engine already `require`s gdn + gdn_block.

## The Spinel triggers found on the union pin (re-test each on master)

All manifested as a miscompiled token-id finalize (`buffer-not-set` /
`grads==NULL` aborts in the **attention** runner), even with the GDN branch never
executed (all-attention). Worked around / blocked:
1. `LayerSpec.kind=` setter on a realize path → **fixed** by int-array dispatch.
2. constructing `LayerSpec` in a realize-path method → **fixed** (no construction).
3. **#688** passing an `is_gdn` ARRAY as a method parameter → **fixed** by
   `set_gdn_layer!(idx)` (int arg).
4. **STILL BLOCKING:** `gblk.alloc_trainable_f32_weights!` / `gblk.set_params!`
   compiled into the engine unit corrupts the backward graph (`grads==NULL`).
   This is the one that forces the separate runner.

**First step on master:** re-add ONLY the alloc branch (below), rebuild
`toy-train`, run `from-scratch --steps 5 --seed 0`. If byte-exact → the codegen
bug is gone; apply the rest. If it still aborts → stay on the runner.

## The re-apply (engine surgery, all spec-gated; default all-attention = byte-exact)

Reference `lib/toy/run/train_hybrid.rb` for the exact op shapes. Each loop branches
on `@seq_is_gdn[li]` (engine-local bool array — array ACCESS is fine, only array
PARAMS trip #688).

**1. API + flags** (add near `enable_full_finetune!`):
```ruby
def set_gdn_layers!(indices)        # call BEFORE realize_for_random_init
  @seq_gdn_layer_indices = indices
end
def build_gdn_flags!                # called inside realize once n_layers known
  @seq_is_gdn = [false]; @seq_is_gdn.pop
  li = 0
  while li < @seq_n_layers; @seq_is_gdn.push(false); li = li + 1; end
  k = 0
  while k < @seq_gdn_layer_indices.length
    idx = @seq_gdn_layer_indices[k]
    if idx >= 0 && idx < @seq_n_layers; @seq_is_gdn[idx] = true; end
    k = k + 1
  end
end
```
ivars in `initialize`: `@seq_gdn_layer_indices = [0]; @seq_gdn_layer_indices.pop`
and `@seq_is_gdn = [false]; @seq_is_gdn.pop`. Delegators:
`seq_gdn_blocks_ffi_ref;  @seq_arch.seq_gdn_blocks_ffi;  end`.

**2. In `realize_for_random_init`, after `seed_blocks!`** — mark kinds (INT arg):
```ruby
build_gdn_flags!
kk = 0
while kk < @seq_gdn_layer_indices.length
  gidx = @seq_gdn_layer_indices[kk]
  if gidx >= 0 && gidx < @seq_n_layers; @seq_arch.set_gdn_layer!(gidx); end
  kk = kk + 1
end
```

**3. Alloc loop** — GDN layers alloc a GDNBlock (s_v = d_head):
```ruby
if @seq_is_gdn[li]
  gblk = self.seq_gdn_blocks_ffi_ref[li]
  gblk.alloc_trainable_f32_weights!(@sess, @seq_d_model, @seq_d_head, @seq_n_heads)
  gblk.set_params!
else
  # ... existing TransformerBlock alloc ...
end
```

**4. `upload_random_init!`** — GDN branch (Gaussian projections, 1.0 gammas,
0.0 a_log/dt_bias). `inner = @seq_n_heads * @seq_d_head`:
```ruby
gblk = self.seq_gdn_blocks_ffi_ref[li]
upload_constant(gblk.t_rn_gamma, @seq_d_model, 1.0)
upload_gaussian(gblk.t_w_q, inner*@seq_d_model, inv_sqrt_d, state)   # k,v,z same
upload_gaussian(gblk.t_w_a, @seq_n_heads*@seq_d_model, inv_sqrt_d, state)  # b same
upload_constant(gblk.t_a_log, @seq_n_heads, 0.0)   # dt_bias same
upload_constant(gblk.t_go_gamma, inner, 1.0)
upload_gaussian(gblk.t_w_o, @seq_d_model*inner, inv_sqrt_d, state)
```

**5. `ft_zero_init_adam`** — GDN branch: `gblk.zero_state!(@sess)` then zero
`gblk.ft_m[i]`/`gblk.ft_v[i]` over `gblk.ft_weights.length`.

**6. `build_training_step`** (full_finetune opt_step loop) — GDN branch: opt_step
over `gblk.ft_weights[wg]` with `gblk.ft_m[wg]`/`gblk.ft_v[wg]`.

**7. CLI/recipe knob:** thread a GDN-layers option (e.g. ENV `GDN_LAYERS=1,3`)
through `lib/toy/run/train.rb` → `Recipes::FromScratch` → `engine.set_gdn_layers!`.

## Verification protocol (master)
1. After step 3 alone: `toy-train from-scratch --steps 5 --seed 0` byte-exact vs
   `prep/fixtures/train_baseline.txt` (the corruption gate).
2. After all: `GDN_LAYERS=1 toy-train from-scratch` trains (loss decreases), and
   no-GDN runs stay byte-exact (from-scratch/warm-start/lora).
3. Retire `toy-train-hybrid` once the engine path is green (keep its smoke as a
   cross-check).
