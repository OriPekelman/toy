#!/usr/bin/env python3
"""prep/research/build_ast_pack.py — toy#173 (G1): a Python-AST graph bundle for
the GNN lane.

WHY PYTHON AND WHY IN prep/. `prep/` already carries Python steps
(convert_smollm2_to_gguf.py, dump_bpe.py, extract_vit_tiny.py), so a
parse step is not a new dependency class, and Python ships `ast` so
there is no parser to write. The deciding argument is PROVENANCE: an
in-repo builder makes the pack reproducible from source with a pinned
selection, which is how every corpus in this programme has been handled
since P1a. A handed-over pack would be an unreproducible input, and this
arc has already paid once for a corpus that differed in ways its axis did
not name (P4's udhr confound).

NO ENGINE CHANGE. train_gnn already takes GNN_GRAPH (CLI --graph) as a
bundle prefix, and when set the pack supplies the graph AND n_classes
from its own meta. Cora rides that path today. So this emits the same
five files and G1 runs on F17's engine byte-for-byte unchanged — which
matters for the verdict: a `no-go-dfa` cannot be blamed on a new code
path, because there is not one.

    <prefix>.meta.i32   [n_nodes, feat_dim, n_classes, n_edges]
    <prefix>.feat.f32   n_nodes * feat_dim, node-major, ROW-NORMALISED
    <prefix>.edge.i32   2 * n_edges (a, b) — undirected, listed ONCE
    <prefix>.label.i32  n_nodes
    <prefix>.split.i32  n_nodes (0 = train, 1 = val)

THE MEASUREMENT G1 EXISTS FOR. An AST's structural node types are few,
but its LEAVES (identifiers, literals) are an open vocabulary, so the
width problem can return through the leaves. Leaves are therefore
abstracted to TYPE (identifier / number / string), never value, and the
RAW LEAF VOCABULARY is counted and emitted beside the structural
dimension — that count is the quantity that decides `no-go-dim`, so it
is measured rather than asserted.

Determinism: the module list is PINNED (not globbed), nodes are emitted
in a deterministic tree walk, and the split is per-class round-robin in
that order, so two machines produce byte-identical bundles. The
interpreter version and a SHA over the exact source list go in the JSON
sidecar.
"""
import argparse, ast, hashlib, json, os, struct, sys

# PINNED module list, not a glob: globbing whatever the box happens to
# ship makes the pack depend on the machine. These are small, pure-Python,
# PSF-licensed stdlib modules with plenty of ordinary function bodies.
PINNED_MODULES = [
    "argparse.py", "ast.py", "bisect.py", "calendar.py", "cmd.py",
    "codeop.py", "colorsys.py", "configparser.py", "copy.py", "csv.py",
    "dataclasses.py", "datetime.py", "difflib.py", "enum.py", "filecmp.py",
    "fnmatch.py", "fractions.py", "getopt.py", "gettext.py", "glob.py",
    "gzip.py", "heapq.py", "hmac.py", "imaplib.py", "inspect.py",
    "ipaddress.py", "json/decoder.py", "json/encoder.py", "keyword.py",
    "linecache.py", "locale.py", "mailbox.py", "mimetypes.py", "netrc.py",
    "ntpath.py", "numbers.py", "operator.py", "optparse.py", "os.py",
    "pathlib.py", "pickle.py", "pipes.py", "pkgutil.py", "platform.py",
    "plistlib.py", "poplib.py", "posixpath.py", "pprint.py", "queue.py",
    "quopri.py", "random.py", "re/_parser.py", "sched.py", "secrets.py",
    "selectors.py", "shelve.py", "shlex.py", "shutil.py", "smtplib.py",
    "socket.py", "ssl.py", "stat.py", "statistics.py", "string.py",
    "tabnanny.py", "tarfile.py", "telnetlib.py", "tempfile.py",
    "textwrap.py", "threading.py", "timeit.py", "token.py", "tokenize.py",
    "trace.py", "traceback.py", "types.py", "typing.py", "uuid.py",
    "warnings.py", "wave.py", "weakref.py", "zipfile.py",
]

# Leaves are abstracted to TYPE, never value — that abstraction IS the
# hypothesis under test, so it is explicit rather than incidental.
LEAF_IDENT, LEAF_NUM, LEAF_STR = "IDENTIFIER", "NUMBER", "STRING"


def leaf_kind(node):
    """Return (kind, raw_value) for a leaf-bearing node, else (None, None).

    raw_value is collected ONLY to count the open vocabulary; it never
    reaches the pack.
    """
    if isinstance(node, ast.Name):
        return LEAF_IDENT, node.id
    if isinstance(node, ast.arg):
        return LEAF_IDENT, node.arg
    if isinstance(node, ast.Attribute):
        return LEAF_IDENT, node.attr
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, str):
            return LEAF_STR, v
        if isinstance(v, (int, float, complex)) and not isinstance(v, bool):
            return LEAF_NUM, repr(v)
    return None, None


def node_type(node, merge_leaves=False):
    """The LABEL: structural node type, with leaves collapsed to their
    abstract kind. This is the output dimension G1 is measuring.

    `merge_leaves` collapses the three leaf kinds into ONE class. That is
    G1's leaf-separability probe (read 3): if structural accuracy is
    unchanged between the two packs, leaf-type prediction is separable
    from structural prediction and a G2 could factor leaves out to their
    own mechanism. If it moves, the two are entangled and the
    small-output claim does not survive the leaves.
    """
    k, _ = leaf_kind(node)
    if k:
        return "LEAF" if merge_leaves else k
    return type(node).__name__


def collect_functions(paths, max_nodes):
    """Yield (path, FunctionDef) for short functions, deterministically."""
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=p)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n = sum(1 for _ in ast.walk(fn))
                if 8 <= n <= max_nodes:
                    yield p, fn


def build(args):
    stdlib = args.stdlib
    srcs, missing = [], []
    for m in PINNED_MODULES:
        fp = os.path.join(stdlib, m)
        (srcs if os.path.isfile(fp) else missing).append(fp if os.path.isfile(fp) else m)
    if len(srcs) < 20:
        sys.exit("build_ast_pack: only %d of %d pinned modules present under %s — "
                 "the pinned list does not match this interpreter, and a globbed "
                 "fallback would make the pack machine-dependent"
                 % (len(srcs), len(PINNED_MODULES), stdlib))

    nodes_kind, edges, fn_of_node = [], [], []
    leaf_raw = {LEAF_IDENT: set(), LEAF_NUM: set(), LEAF_STR: set()}
    n_fns = 0

    for path, fn in collect_functions(srcs, args.max_fn_nodes):
        if n_fns >= args.max_functions:
            break
        base = len(nodes_kind)
        idx, order = {}, []

        # Deterministic pre-order walk; ast.iter_child_nodes is
        # field-ordered, so this is stable across runs and machines.
        def walk(nd):
            i = len(order)
            idx[id(nd)] = base + i
            order.append(nd)
            for ch in ast.iter_child_nodes(nd):
                walk(ch)
        walk(fn)

        for nd in order:
            nodes_kind.append(node_type(nd, args.merge_leaves))
            fn_of_node.append(n_fns)
            k, raw = leaf_kind(nd)
            if k is not None:
                leaf_raw[k].add(raw)
        # Parent/child edges, plus sibling edges so the graph is not a
        # pure tree (a tree makes neighbourhood aggregation trivially
        # directional; siblings give the aggregator something to average).
        for nd in order:
            kids = list(ast.iter_child_nodes(nd))
            for ch in kids:
                edges.append((idx[id(nd)], idx[id(ch)]))
            for a, b in zip(kids, kids[1:]):
                edges.append((idx[id(a)], idx[id(b)]))
        n_fns += 1

    if n_fns == 0:
        sys.exit("build_ast_pack: no functions collected")

    # ---- labels: the structural node-type vocabulary ----
    kinds = sorted(set(nodes_kind))
    kind_id = {k: i for i, k in enumerate(kinds)}
    labels = [kind_id[k] for k in nodes_kind]
    n_nodes, n_classes = len(nodes_kind), len(kinds)

    # ---- features: SHAPE ONLY. No node-type information, anywhere. ----
    #
    # THE FIRST VERSION OF THIS WAS DEGENERATE AND THE CONTROL CAUGHT IT.
    # It used the PARENT's kind and the CHILD-kind histogram. I had
    # excluded the node's own type, reasoning that a label must not be a
    # copy of an input — but in a GRAMMAR the parent's type and the
    # children's types nearly determine a node's own type, so the label
    # leaked in through the neighbours. Measured consequence: a frozen
    # body (head only) reached .175 and IMPROVED with training while BP
    # DEGRADED (.067 -> .028), because propagation averages away exactly
    # the signal the head was reading off. A control that wins is a task
    # that answers itself.
    #
    # So features are now purely SHAPE: degree, child count, depth,
    # sibling index, subtree size. None of them carries a node type, so
    # the type has to come from the graph — which is the only way this
    # lane measures credit assignment rather than feature lookup.
    #
    # Note what this costs, honestly: the three LEAF classes are
    # shape-identical (all childless), so they are not separable by these
    # features. That is not a defect of the encoding — it IS G1's read 3,
    # made visible instead of hidden behind a leaked label.
    parent = [-1] * n_nodes
    for a, b in edges:
        if parent[b] == -1 and b > a:
            parent[b] = a
    kids_of = [0] * n_nodes
    for a, b in edges:
        if parent[b] == a:
            kids_of[a] += 1
    sib_ix = [0] * n_nodes
    seen_kids = {}
    for i in range(n_nodes):
        p_i = parent[i]
        if p_i != -1:
            seen_kids[p_i] = seen_kids.get(p_i, 0) + 1
            sib_ix[i] = seen_kids[p_i]
    subtree = [1] * n_nodes
    for i in range(n_nodes - 1, -1, -1):
        if parent[i] != -1:
            subtree[parent[i]] += subtree[i]

    def buckets(v, edges_list):
        return [1.0 if v <= e else 0.0 for e in edges_list]

    B_DEG, B_KID, B_DEP, B_SIB, B_SUB = (
        [1, 2, 3, 5, 8, 99], [0, 1, 2, 3, 5, 99], [0, 1, 2, 3, 5, 8, 99],
        [0, 1, 2, 3, 5, 99], [1, 2, 4, 8, 16, 64, 999])
    feat_dim = len(B_DEG) + len(B_KID) + len(B_DEP) + len(B_SIB) + len(B_SUB)
    feats = [0.0] * (n_nodes * feat_dim)
    deg = [0] * n_nodes
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    depth = [0] * n_nodes
    for i in range(n_nodes):
        pp, d = parent[i], 0
        while pp != -1 and d < 64:
            d += 1
            pp = parent[pp]
        depth[i] = d
    for i in range(n_nodes):
        row = (buckets(deg[i], B_DEG) + buckets(kids_of[i], B_KID) +
               buckets(depth[i], B_DEP) + buckets(sib_ix[i], B_SIB) +
               buckets(subtree[i], B_SUB))
        off = i * feat_dim
        for j, v in enumerate(row):
            feats[off + j] = v

    # Row-normalise, matching fetch_cora's convention.
    for i in range(n_nodes):
        off = i * feat_dim
        srow = sum(feats[off:off + feat_dim])
        if srow > 0:
            for j in range(feat_dim):
                feats[off + j] /= srow

    # ---- split ----
    #
    # TWO DIFFERENT CONVENTIONS, and using the wrong one made the masked
    # task unlearnable. Cora's is semi-supervised node classification:
    # 20 labelled per class, EVERYTHING ELSE val. Masking that val set
    # zeroes 94% of the graph, so a masked node's neighbours are masked
    # too and there is no signal anywhere to propagate — measured, every
    # arm at or BELOW chance (bp .0025 vs chance .0132).
    #
    # A masked-prediction task needs the opposite balance: mask a small
    # FRACTION, keep the rest intact as context. So under --mask the
    # scored set is `mask_frac` of nodes and the other ~85% are training
    # context with features; without it the Cora convention stands.
    if args.mask:
        split = [0] * n_nodes
        stride = max(2, int(round(1.0 / args.mask_frac)))
        for i in range(n_nodes):
            if i % stride == 0:
                split[i] = 1
    else:
        seen, split = {}, [1] * n_nodes
        for i in range(n_nodes):
            c = labels[i]
            seen[c] = seen.get(c, 0) + 1
            if seen[c] <= args.train_per_class:
                split[i] = 0

    # ---- MASKING: the task is "predict a node's type from the tree
    # AROUND it", so a masked node's OWN features must be gone. ----
    #
    # THE ERROR THIS FIXES, recorded because it cost a full sweep. The
    # spec says "mask a fraction of nodes, predict their type from the
    # surrounding tree" and I built UNMASKED classification: every node
    # kept its own shape features. But shape largely DETERMINES type in a
    # grammar — childless implies a leaf, two children at a given depth
    # implies a BinOp — so a head read the answer straight off the node
    # and the body contributed nothing. Measured: frozen .4746 vs bp
    # .4766, a 0.002 margin, control winning 2 of 5 seeds. The
    # precondition failed and no credit number was interpretable.
    #
    # With the val nodes' own features zeroed, the type has to come from
    # the neighbourhood, which is the only version of this task that
    # measures propagation at all.
    if args.mask:
        for i in range(n_nodes):
            if split[i] == 1:
                off = i * feat_dim
                for j in range(feat_dim):
                    feats[off + j] = 0.0

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    edges = sorted(set((min(a, b), max(a, b)) for a, b in edges if a != b))
    with open(out + ".meta.i32", "wb") as f:
        f.write(struct.pack("<4i", n_nodes, feat_dim, n_classes, len(edges)))
    with open(out + ".feat.f32", "wb") as f:
        f.write(struct.pack("<%df" % len(feats), *feats))
    with open(out + ".edge.i32", "wb") as f:
        flat = [x for e in edges for x in e]
        f.write(struct.pack("<%di" % len(flat), *flat))
    with open(out + ".label.i32", "wb") as f:
        f.write(struct.pack("<%di" % n_nodes, *labels))
    with open(out + ".split.i32", "wb") as f:
        f.write(struct.pack("<%di" % n_nodes, *split))

    src_sha = hashlib.sha256()
    for p in srcs:
        with open(p, "rb") as fh:
            src_sha.update(hashlib.sha256(fh.read()).digest())

    # THE TWO NUMBERS G1 TURNS ON: the structural dimension (the head
    # width) and the RAW LEAF VOCABULARY it was bought with.
    raw_total = sum(len(v) for v in leaf_raw.values())
    meta = {
        "source": "python-stdlib (PSF)", "python": sys.version.split()[0],
        "stdlib": stdlib, "modules_pinned": len(PINNED_MODULES),
        "modules_found": len(srcs), "modules_missing": missing,
        "sources_sha256": src_sha.hexdigest(),
        "n_functions": n_fns, "n_nodes": n_nodes, "n_edges": len(edges),
        "feat_dim": feat_dim,
        "structural_node_types": n_classes,
        "node_type_names": kinds,
        "raw_leaf_vocab_total": raw_total,
        "raw_leaf_vocab": {k: len(v) for k, v in leaf_raw.items()},
        "masked": args.mask,
        "mask_frac": args.mask_frac if args.mask else None,
        "merge_leaves": args.merge_leaves,
        "train_per_class": args.train_per_class,
        "train_nodes": sum(1 for s in split if s == 0),
    }
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print("ast pack -> %s" % out)
    print("  functions=%d nodes=%d edges=%d feat_dim=%d" % (n_fns, n_nodes, len(edges), feat_dim))
    print("  STRUCTURAL NODE TYPES (head width) = %d" % n_classes)
    print("  RAW LEAF VOCAB (the open-vocab risk) = %d  (%s)" %
          (raw_total, ", ".join("%s %d" % (k, len(v)) for k, v in sorted(leaf_raw.items()))))
    print("  ratio raw_leaf/structural = %.1fx" % (raw_total / float(n_classes)))
    print("  train nodes=%d of %d" % (meta["train_nodes"], n_nodes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ast_code")
    ap.add_argument("--stdlib", default=None)
    ap.add_argument("--max-functions", type=int, default=400)
    ap.add_argument("--max-fn-nodes", type=int, default=120)
    ap.add_argument("--train-per-class", type=int, default=20)
    ap.add_argument("--mask-frac", type=float, default=0.15)
    ap.add_argument("--mask", action="store_true",
                    help="zero the val nodes' OWN features — the task Tao specced")
    ap.add_argument("--merge-leaves", action="store_true",
                    help="collapse the three leaf kinds into one class (G1 read 3)")
    a = ap.parse_args()
    if a.stdlib is None:
        import sysconfig
        a.stdlib = sysconfig.get_paths()["stdlib"]
    build(a)
