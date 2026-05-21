# Draft: upstream ggml issue — ggml-cpu sched aliases intermediate
# grad-chain tensor slots in long backward graphs (CUDA correct)
#
# To post at: https://github.com/ggml-org/ggml/issues/new
# When user is happy with the wording, copy-paste the section below.

## Summary

In a long backward computation graph (hundreds of grad nodes from
`ggml_build_backward_expand` over a 30-layer transformer), the
`ggml-cpu` backend's sched reuses buffer slots for intermediate grad
tensors that still have downstream consumers. Downstream reads pick
up stale data, and the produced gradients differ from `ggml-cuda`'s
by 10×–7500× per layer (random sign/magnitude). On CUDA the same
graph computes correctly.

Pinning every node in the backward cgraph with `ggml_set_output`
restores CPU gradient correctness — confirming the slot-reuse
decision is what's wrong, not the per-op math.

## Reproducer

Working tree: https://github.com/OriPekelman/toy_ruby_neural_network
@ commit `a822601` (the pinned-vs-default smoke pair was added there
specifically as a repro for this issue).

Quickest path:

```bash
git clone https://github.com/OriPekelman/toy_ruby_neural_network
cd toy_ruby_neural_network
make setup-ggml        # CPU build
make smollm2_lora_train_ce smollm2_lora_train_ce_pinned

# Default (broken on CPU):
GGUF=data/smollm2-135m-native.gguf ./demos/smollm2_lora_train_ce
# CPU loss: step  1: CE=7.519449
#           step 20: CE=7.518375    ← essentially flat

# Pinned (workaround — set_output on every graph_b node):
GGUF=data/smollm2-135m-native.gguf ./demos/smollm2_lora_train_ce_pinned
# CPU loss: step  1: CE=7.519449
#           step 20: CE=0.210978    ← real convergence
```

For comparison the CUDA build (`make setup-ggml-cuda &&
make smollm2_lora_train_ce_cuda`) at the SAME graph + SGD config
produces step-by-step values matching the pinned-CPU trajectory
within FP32 noise:

| step | CPU pinned | CUDA  |
|---:|---:|---:|
|  1 | 7.519449 | 7.519447 |
|  5 | 6.776536 | 6.776540 |
| 10 | 3.750428 | 3.750362 |
| 20 | 0.210978 | 0.210942 |

The pinning diff in our wrapper is the one-line
`ggml_set_output(node)` walk over `cgraph->nodes` after
`ggml_build_backward_expand`. The actual ggml-side reproducer is
"build any deep autograd graph (transformer with LoRA, in our case),
compute on CPU sched, observe wrong grads at deeper layers".

## What we ruled out (via bisect)

Single-op + single-block bisect smokes
(`tinynn/ab_smoke_lora_train_{rmsnorm,rope,softmax,view,concat}*`)
all agreed CPU == CUDA bit-identically:

| Smoke | CPU final loss | CUDA final loss |
|---|---|---|
| `+ rms_norm`                 | 7.999993324279785          | 7.999993324279785 |
| `+ rope_ext`                 | 36.01875686645508          | 36.01875686645508 |
| `+ scale + softmax + V@attn` | 0.15240749716758728        | 0.15240749716758728 |
| `+ view_2d through K/V`      | 0.3685033917427063         | 0.3685033917427063 |
| `+ concat (2 heads + O proj)`| 0.42687365412712097        | 0.42687365412712097 |

So none of the individual backward op implementations (matmul,
rms_norm_back, rope_ext_back, soft_max_back, scale, mul_mat through
strided views, concat backward (vendored locally — separately PR-able))
are wrong. The bug emerges only at full-graph scale where the sched
has many concurrent live tensors and starts reusing slots.

## Where I suspect the bug lives

`vendor/ggml/src/ggml-alloc.c` + `ggml-backend.cpp`'s buffer
allocator's "last consumer" tracking. The allocator decides when a
tensor's slot returns to the free pool by counting downstream
references; some path is undercounting for the grad chain shape that
`build_backward_expand` produces. CUDA's allocator either tracks
correctly or is conservative enough not to alias at our scale.

I did NOT bisect inside ggml-alloc / ggml-backend for this session;
the wrapper-level reproducer is the cleanest place I can drop in
without spending more time on the ggml allocator internals.

## Why this matters

The CPU backend is the recommended path for development + small
models. A silent gradient-magnitude bug at LLM-shape backward chains
makes any CPU-side training give bad-looking convergence ("loss
decreasing!") that's actually just FP accumulation drift, masking
the bug from gates that only check monotonic-decrease.

I tightened my own gate (`final < 0.5 * initial`) to fail the bug
loudly. Most autograd-using users probably haven't.

## What I tried to side-step it

The `tnn_pin_all_graph_b_nodes` workaround (loop `ggml_set_output`
over every node in graph_b) restores CPU correctness, at the cost
of disabling buffer-slot reuse entirely. It's a workaround,
not a fix, but it's a clean diagnostic that pinpoints the buggy
behavior.

Happy to provide additional details, a minimal-shape repro outside
our Ruby wrapper, or pointers to lines in our setup. Let me know
which would help most.
