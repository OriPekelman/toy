# `prep/research/` — the concluded programme's reproducibility record

One-shot sweep harnesses and report scripts from the DFA-vs-BP arc
(toy#111–#173). They are kept, not deleted: they are how each published
number was produced, and a result whose harness is gone is a result nobody
can re-derive.

They are **not** framework tooling. Nothing here is referenced by the
Makefile, no gate invokes them, and `make gates` does not run them. That is
the whole reason they sit in their own directory — `prep/` proper is build
and data tooling that the framework depends on, and mixing the two made a
106-entry directory in which neither was findable.

| | |
|---|---|
| `p3_depth.sh` `p3_sweep.sh` | toy#170 P3 — the byte-LM anchor and the depth sweep |
| `p4_alphabet.sh` | toy#170 P4 — alphabet size (the arm whose collapse turned out to be a language/difficulty confound) |
| `p5_headwidth.sh` `p5_remap.sh` `p5_report.rb` | toy#170 P5 — nominal head width vs a controlled remap |
| `p6_ladder.sh` | toy#170 P6 — the arithmetic-rank ladder |
| `e1_instrument.sh` `e1_report.rb` | toy#172 E1.1 — the anisotropy instrument |
| `e1_ndfa.sh` `e1_ndfa_report.rb` | toy#172 E1.2 — nDFA, the error-side preconditioner |
| `e2_ldfa.sh` `e2_ldfa_report.rb` | toy#172 E2 — LDFA, adaptive low-rank feedback via Oja's rule |
| `g1_ast.sh` `g1_leafsep.sh` `g1_report.rb` `build_ast_pack.py` | toy#173 G1 — masked AST-node-type prediction |
| `difflm_report.rb` | toy#166 capstone P1b — generation metrics, computed outside the runner |

## Running them

They `cd` to the repo root themselves (two levels up from here), so invoke
them from anywhere:

```sh
zsh prep/research/p6_ladder.sh
```

They are `zsh` scripts, not `bash` — `bash -n` will report syntax errors on
constructs that are valid zsh.

## What is deliberately NOT here

`prep/remap_alphabet.rb` and `prep/fetch_text.rb` stay in `prep/` proper.
They build corpus packs, and `lib/` error messages name them as the way to
produce one — a product-path message must not point into the research tree.

The findings themselves are not here either. They live in the issues and in
`docs/research/`; this directory is only the machinery.
