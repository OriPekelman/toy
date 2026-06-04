# tep_demo: HTTP serving via Tep + Spinel

[Tep](https://github.com/OriPekelman/tep) is a Sinatra-flavoured Ruby
framework that compiles via [Spinel](https://github.com/matz/spinel)
to a native HTTP server. This directory holds the remaining Tep+Spinel
demo endpoints (a `GET /` smoke, a toy inference server, and a GPT-2
OpenAI-shape server).

> **The canonical llama-family OpenAI server moved into the toy CLI.**
> What used to be `tep_demo/openai_api_llama.rb` is now `toy serve`,
> backed by `lib/toy/serve/openai/`. Serve any llama-family GGUF
> (SmolLM2, Qwen2.5, TinyLlama, Llama-3.x) with:
> ```sh
> toy serve data/SmolLM2-135M-Instruct-Q8_0.gguf --port 4567 --name smol
> ```
> See the project README "serve" section for the endpoint contract.
> The demos below are the leftover Tep+Spinel examples.

## Demo servers (curated)

For real serving, use **`toy serve <model.gguf>`** (above). These are the
remaining Tep+Spinel *framework* demos:

| Source | Binary | What it does | Status |
|---|---|---|---|
| `hello_api.rb` | `tep_demo/hello` | Minimal `GET /` smoke; baseline HTTP throughput | **kept** — the minimal Tep demo |
| `openai_api.rb` | `tep_demo/openai_api` | DistilGPT2/GPT-2 KV-cache decode behind `POST /v1/chat/completions` (`lib/bpe.rb` server tokenizer; `chat.rb` is a client, `post_smoke.rb` a POST smoke) | **pending** the GPT-2 serve convergence ([toy#30](https://github.com/OriPekelman/toy/issues/30)) |

`legacy/inference_api.rb` (toy random-init `FullForwardFFICache`, `/generate?n=N`)
moved to [`legacy/`](legacy/) — a second serving demo, superseded by `toy serve`
as the canonical serving path. It still builds (`make tep_demo/api`).

## Build

```sh
make setup-ggml                            # one-time
make tep_demo/hello                        # ~5 s build
make tep_demo/api                          # toy inference HTTP server
./prep/build_tep_app.sh tep_demo/openai_api.rb tep_demo/openai_api
                                           # GPT-2 server (legacy build wrapper)
```

The GPT-2 server (`openai_api.rb`) uses the older
`prep/build_tep_app.sh` lib-concat wrapper because Tep's translator
once dropped external `require_relative`s (see
[`docs/archive/upstream/issues-tep/01-warn-on-external-require-relative.md`](../docs/archive/upstream/issues-tep/01-warn-on-external-require-relative.md)).

## OpenAI-compatible API — GPT-2 (`openai_api`)

```sh
./tep_demo/openai_api -p 4585 > /tmp/api.log 2>&1 &

curl -s http://127.0.0.1:4585/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt2","messages":[{"role":"user","content":"Once upon a time"}],"max_tokens":20}' \
  | jq .
```

For an interactive chat against it, use the bundled client:

```sh
TOY_API=http://127.0.0.1:4585/v1 ./tep_demo/chat.rb
# > Hello, my name is
# John. I'm a writer, and I'm a writer. I'm a writer. ...
#   [turn 1, 1932 ms]
# > /tokens 20
# > /system You are a helpful assistant.
# > /quit
```

`tep_demo/chat.rb` is CRuby host-side (Net::HTTP + JSON, no deps).
It owns the conversation history; the server is stateless. Slash
commands: `/reset`, `/system <text>`, `/tokens <N>`, `/quit`.

Same endpoint works with the official `openai` Python client:

```py
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:4585/v1", api_key="unused")
print(c.chat.completions.create(
    model="gpt2",
    messages=[{"role": "user", "content": "Once upon a time"}],
    max_tokens=20,
).choices[0].message.content)
```

End-to-end throughput probe: `./tep_demo/bench_api.sh`. On an M2 Air
this lands around **~50 tok/s** through the full HTTP pipeline; the
KV-cache decode (~14 ms/token) is the dominant cost.

## HTTP-only throughput

`hello_api.rb` (no inference) measures Tep+Spinel's HTTP path in
isolation. Numbers on gx10 (NVIDIA GB10 / aarch64, 1 worker,
`Connection: close`):

| Concurrency | Throughput |
|---|---:|
| 1 thread   |   7,745 req/s |
| 4 threads  |  22,054 req/s |
| 16 threads |  19,587 req/s |
| 64 threads |  18,122 req/s |

With keep-alive + Tep's prefork (workers > 1) the upstream Tep README
quotes ~167k req/s; we use `Connection: close` here for simplicity.

## Caveats

- **Tep vendoring uses the spinelgems convention.** As of 2026-05-27,
  `tep_demo/*.rb` requires Tep via `vendor/spinel/deps` (placed by
  `spinel-compat vendor` from the `Gemfile` declaring `gem "tep",
  path: "../tep"`). Re-vendor when Tep moves:
  ```sh
  make vendor-tep    # bundle lock → spinel-compat vendor → post_vendor_tep.rb
  ```
  See `docs/archive/spinelgems-tep-adoption-2026-05-27.md` for the
  background. The older `prep/sync_tep.rb` + `tep_demo/_tep_lib/`
  rsync hack was retired in the same commit.
- **Spinel name collisions.** `Mat#add` was renamed to `Mat#plus` to
  avoid a dispatch clash with `Tep::Router#add` — method names are
  the unit of collision, so `Mat#add!` (in-place) is fine.
- **OpenAI parser is a hand-rolled byte scan**, not a JSON parser.
  See [`docs/archive/upstream/issues-spinel/03-string-index-returns-minus-one.md`](../docs/archive/upstream/issues-spinel/03-string-index-returns-minus-one.md)
  for why `String#index` + the `pos.nil?` idiom didn't work on Spinel.
- **`inference_api.rb` is the older toy path** (random weights, no
  KV cache, T_SEQ-padded forward) — kept for the per-step latency
  table below and as a minimum-deps example. For real llama-family
  serving use `toy serve` (see the project README).

## `inference_api.rb` numbers (for the record)

Toy model: `vocab=16, d_model=32, d_ff=64, n_heads=4, n_layers=2,
T=16`, prompt `[3,7,1]`, gx10, 1 worker, `Connection: close`:

| Request shape       | Threads | Throughput   | Tokens/sec |
|---|---|---:|---:|
| `/generate?n=5`     |   1  |  1,314 req/s |   6,570 |
| `/generate?n=5`     |   4  |  1,758 req/s |   8,790 |
| `/generate?n=5`     |   8  |  1,706 req/s |   8,529 |
| `/generate?n=5`     |  16  |  2,738 req/s |  13,689 |
| `/generate?n=10`    |   4  |    920 req/s |   9,201 |
| `/generate?n=32`*   |   4  |    308 req/s |   9,868 |

\* `n=32` exceeds the realized `T_SEQ=16`, so generation after token 16
gets the wrong context — the bench measures HTTP+forward throughput,
not coherent generation.
