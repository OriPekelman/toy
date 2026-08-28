#!/usr/bin/env python3
"""prep/research/build_ast_traversal_pack.py — rev2026-08-28 (D5): G1's
pre-registered escalation, as a bytelm pack for the gtx lane.

WHY THIS EXISTS. G1's masked-node fixture could not discriminate: on a
tree with a grammar, a FROZEN random GNN matches BP, so no credit-rule
comparison was interpretable and G1 reported `UNANSWERABLE`. Its spec
pre-registered the escape: "escalate to a generation-shaped task —
predict the next node in a pre-order AST traversal from the prefix".
That is an autoregressive sequence task, so it belongs on the gtx bytelm
lane (P2-P6's lane), not on the GNN lane. This builder emits it.

THE PACK IS A LADDER, AND THE LADDER IS THE POINT. P6's width ladder
inflated the alphabet with PLANTED UNIFORM NOISE: rung a2504 has 2504
codes carrying the same 65 bits of information, so widening the output
also floods the error with unlearnable variance. D2 then showed the
degradation tracks error-SNR dilution, not any property of B. That
leaves the programme's central law ambiguous in exactly one place:

    is DFA hurt by OUTPUT WIDTH, or by NOISE IN THE ERROR?

The a-series cannot separate them, because there widening IS adding
noise. Here it is not. The traversal is fixed; only how finely LEAVES are
partitioned changes:

    --leaf-vocab 0     leaves emit their TYPE (IDENTIFIER/NUMBER/STRING)
    --leaf-vocab K     the K most frequent leaf VALUES get their own
                       class; the tail falls back to its type token

Every rung has the SAME token positions and the SAME underlying corpus;
refining K splits existing classes into finer ones that carry REAL,
LEARNABLE information. So width rises with no added noise. If DFA
degrades across this ladder as it did across the a-series, the law is
about width. If it holds, the law is about noise — and P6's headline has
to be restated.

Reads the same PINNED_MODULES and the same leaf abstraction as
build_ast_pack.py (G1's graph builder) so the two fixtures describe one
corpus; the pre-order walk is shared with it as well (ast.iter_child_nodes
is field-ordered, so child order is preserved and the traversal is a real
pre-order, not an arbitrary node listing).

Emits the gtx bytelm three-file pack:
    <prefix>.meta.i32   [n_tokens, alphabet]
    <prefix>.tok.i32    n_tokens contiguous class ids
    <prefix>.json       provenance sidecar
"""
import argparse, ast, hashlib, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ast_pack import (PINNED_MODULES, LEAF_IDENT, LEAF_NUM, LEAF_STR,
                            leaf_kind, node_type, collect_functions)

SEP = "<FN>"          # function boundary; the traversal is a stream of trees


def gather(stdlib, max_fn_nodes, max_functions, unit):
    """Pre-order walks of every collected unit, plus the leaf-value counts.
    Deterministic: pinned module list, field-ordered walk.

    `unit` is `function` (G1's corpus definition — short function bodies)
    or `module` (the whole parsed file). Module is the default here
    because the task is GENERATION-shaped: a G2 emits whole files, and the
    function-only corpus saturates at ~288k tokens, which would make this
    fixture 4x shorter than the ae_shakespeare baseline it is read
    against and confound width with how often training wraps the corpus."""
    srcs, missing = [], []
    for m in PINNED_MODULES:
        fp = os.path.join(stdlib, m)
        (srcs if os.path.isfile(fp) else missing).append(fp if os.path.isfile(fp) else m)
    if len(srcs) < 20:
        sys.exit("build_ast_traversal_pack: only %d of %d pinned modules under %s"
                 % (len(srcs), len(PINNED_MODULES), stdlib))

    walks, leaf_counts, leaf_by_kind = [], Counter(), {LEAF_IDENT: set(), LEAF_NUM: set(), LEAF_STR: set()}
    def units():
        if unit == "module":
            for p in srcs:
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        yield p, ast.parse(fh.read(), filename=p)
                except (SyntaxError, UnicodeDecodeError):
                    continue
        else:
            for pf in collect_functions(srcs, max_fn_nodes):
                yield pf

    n_fns = 0
    for path, fn in units():
        if n_fns >= max_functions:
            break
        order = []

        def walk(nd):
            order.append(nd)
            for ch in ast.iter_child_nodes(nd):
                walk(ch)
        walk(fn)

        seq = []
        for nd in order:
            k, raw = leaf_kind(nd)
            if k is None:
                seq.append((type(nd).__name__, None))
            else:
                seq.append((k, raw))
                leaf_counts[(k, raw)] += 1
                leaf_by_kind[k].add(raw)
        walks.append(seq)
        n_fns += 1
    if n_fns == 0:
        sys.exit("build_ast_traversal_pack: no functions collected")
    return srcs, missing, walks, leaf_counts, leaf_by_kind


def build(args):
    srcs, missing, walks, leaf_counts, leaf_by_kind = gather(
        args.stdlib, args.max_fn_nodes, args.max_functions, args.unit)

    # THE LADDER RUNG. Top-K leaf VALUES by frequency become their own
    # classes; everything else falls back to its leaf TYPE. K is applied
    # to a frequency order that is stable under ties (count, then the
    # repr of the key) so two machines pick the same K.
    top = set()
    if args.leaf_vocab > 0:
        ranked = sorted(leaf_counts.items(), key=lambda kv: (-kv[1], repr(kv[0])))
        top = set(k for k, _ in ranked[:args.leaf_vocab])

    # Vocabulary: structural types + the three leaf types + SEP + the
    # promoted leaf values. Sorted so ids are reproducible.
    struct_types, leaf_types = set(), set()
    for seq in walks:
        for kind, raw in seq:
            (leaf_types if raw is not None else struct_types).add(kind)
    vocab = [SEP] + sorted(struct_types) + sorted(leaf_types) + \
            [("V", k[0], k[1]) for k in sorted(top, key=lambda kv: (kv[0], repr(kv[1])))]
    vid = {t: i for i, t in enumerate(vocab)}

    toks = []
    limit = args.n_tokens
    truncated = False
    for seq in walks:
        if limit and len(toks) + 1 + len(seq) > limit:
            # Emit whole functions only, then pad-truncate on the last one
            # so the stream never ends mid-tree unless it has to.
            if len(toks) + 1 >= limit:
                truncated = True
                break
        toks.append(vid[SEP])
        for kind, raw in seq:
            if raw is not None and (kind, raw) in top:
                toks.append(vid[("V", kind, raw)])
            else:
                toks.append(vid[kind])
        if limit and len(toks) >= limit:
            truncated = True
            break
    if limit:
        toks = toks[:limit]

    used = sorted(set(toks))
    if len(used) != len(vocab):
        # Re-index to keep ids CONTIGUOUS and every class ACTIVE: a class
        # that never occurs is not part of this rung's width, and quoting
        # it as such would inflate the axis the ladder measures.
        remap = {o: n for n, o in enumerate(used)}
        toks = [remap[t] for t in toks]
        vocab = [vocab[o] for o in used]

    n_tokens, alphabet = len(toks), len(vocab)
    cnt = Counter(toks)
    ent = -sum((c / n_tokens) * (c / n_tokens and __import__("math").log2(c / n_tokens))
               for c in cnt.values())

    import numpy as np
    out = args.out
    np.array([n_tokens, alphabet], dtype=np.int32).tofile(out + ".meta.i32")
    np.array(toks, dtype=np.int32).tofile(out + ".tok.i32")

    h = hashlib.sha256()
    for p in srcs:
        with open(p, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    side = {
        "corpus": "python-stdlib AST pre-order traversal",
        "task": "next node in a pre-order AST traversal (G1's pre-registered escalation)",
        "builder": "prep/research/build_ast_traversal_pack.py",
        "stdlib": args.stdlib,
        "python": sys.version.split()[0],
        "modules_pinned": len(PINNED_MODULES),
        "modules_found": len(srcs),
        "modules_missing": missing,
        "sources_sha256": h.hexdigest(),
        "unit": args.unit,
        "units": len(walks),
        "leaf_vocab_rung": args.leaf_vocab,
        "n_tokens": n_tokens,
        "alphabet": alphabet,
        "truncated_to_n_tokens": bool(truncated and limit),
        "entropy_bits": ent,
        "structural_types": len(struct_types),
        "leaf_types": sorted(leaf_types),
        "raw_leaf_vocab": {k: len(v) for k, v in leaf_by_kind.items()},
        "raw_leaf_vocab_total": sum(len(v) for v in leaf_by_kind.values()),
    }
    with open(out + ".json", "w") as fh:
        json.dump(side, fh, indent=2, sort_keys=True)
    print(f"{out}: n_tokens={n_tokens} alphabet={alphabet} "
          f"entropy={ent:.4f} bits  (structural={len(struct_types)}, "
          f"leaf_values_promoted={len(top)}, raw_leaf_vocab="
          f"{side['raw_leaf_vocab_total']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdlib", default=os.path.dirname(os.__file__))
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaf-vocab", type=int, default=0,
                    help="0 = leaves as TYPE; K = top-K leaf VALUES promoted to own classes")
    ap.add_argument("--n-tokens", type=int, default=0,
                    help="0 = the corpus's natural length (the pinned modules "
                         "saturate at ~300k nodes, so the TEXT baseline is "
                         "truncated to match rather than this being padded up)")
    ap.add_argument("--unit", choices=("module", "function"), default="module",
                    help="module = whole parsed file (generation-shaped, the default); "
                         "function = G1's short-function corpus")
    ap.add_argument("--max-fn-nodes", type=int, default=400)
    ap.add_argument("--max-functions", type=int, default=1000000)
    build(ap.parse_args())
