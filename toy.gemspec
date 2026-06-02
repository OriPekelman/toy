require_relative "lib/toy/version"

Gem::Specification.new do |s|
  s.name        = "toy"
  s.version     = Toy::VERSION
  s.summary     = "Transformer LM primitives in Ruby, Spinel-compiled to native"
  s.description = <<~TEXT.strip
    Toy is a pure-Ruby transformer LM that compiles to a native binary
    via Spinel. Inference (KV-cache decode on CPU/CUDA/Metal), training
    (LoRA/full-FT/from-scratch via tinynn-FFI'd ggml), GGUF load + save,
    Tao-compatible events.jsonl emission. As a gem, it exposes the
    primitives a research project composes: tinynn FFI bridges,
    the sequence-forward training graph, ViT-Tiny, the Llama / SmolLM2 /
    Qwen2.5 inference cache, the GGUF loader, drift/grad/CKA observability,
    cosine LR schedules, and the GGUF checkpoint writer. Filed as toy#19;
    pairs with bundler-spinel / spinelgems for the consumer vendor flow.
  TEXT
  s.authors     = ["Ori Pekelman"]
  s.email       = ["ori@pekelman.com"]
  s.license     = "MIT"
  s.homepage    = "https://github.com/OriPekelman/toy"
  s.metadata    = {
    "source_code_uri"   => "https://github.com/OriPekelman/toy",
    "bug_tracker_uri"   => "https://github.com/OriPekelman/toy/issues",
    "documentation_uri" => "https://github.com/OriPekelman/toy#readme",
  }

  # Runtime target is Spinel's Ruby level. The compiled binaries embed
  # everything (no Ruby runtime); consumers vendoring toy's lib/ via the
  # spinelgems convention have no Ruby-version constraint coming from us.
  s.required_ruby_version = ">= 3.2.0"

  # Packaging model: option (c) from toy#28 — ship the backend BUILD
  # INPUTS so `gem install toy && toy install` works (install orchestrates
  # the ggml + tinynn-shim build inside the gem dir). The backend-free
  # CRuby commands (new/list/describe/fetch/--manifest) work with just the
  # .rb + tinynn sources; `toy install` adds the build. (Fat gem with
  # prebuilt per-platform .a — option (a) — is a tracked follow-up in #28.)
  #
  # The gem ships:
  #   - Every .rb under lib/ (the primitives + the CRuby CLI shell)
  #   - The Makefile (toy install's build orchestrator) + its output
  #     filters (prep/quietly, prep/progress)
  #   - vendor-patches/*.patch (the 6 ggml patches `make setup-ggml` applies)
  #   - The tinynn C/Metal shim BUILD SOURCES (CPU: ggml/gguf/trace/events;
  #     Metal: tinynn_backend_metal.m; CUDA: tinynn_backend_cuda.c). NOT the
  #     dev probes (cuda_byo_smoke, rms_norm_back_probe*) — not build inputs.
  #   - README + LICENSE + CHANGELOG
  # We do NOT ship:
  #   - vendor/ggml/* (cloned fresh by `make setup` at install time)
  #   - data/* (model weights / pretokenized corpora — caller-managed)
  #   - the prebuilt .a (built on the user box by `toy install`)
  # Ship only git-TRACKED files matching these globs. Intersecting with
  # `git ls-files` keeps gitignored build artifacts OUT (compiled .o/.a, the
  # native example/tinynn binaries, libexec/, data/*.bin) even if a glob would
  # otherwise scoop them — belt-and-suspenders matching tep.gemspec. The
  # `.reject` also drops any stray object files for a no-git unpacked build.
  tracked = (`git ls-files -z`.split("\x0") rescue [])
  s.files = Dir[
    "README.md", "LICENSE", "CHANGELOG.md",
    "Makefile",
    "prep/quietly", "prep/progress",
    "lib/**/*.rb",
    "vendor-patches/*.patch",
    "tinynn/tinynn_ggml.h",   "tinynn/tinynn_ggml.c",
    "tinynn/tinynn_gguf.h",   "tinynn/tinynn_gguf.c",
    "tinynn/tinynn_trace.h",  "tinynn/tinynn_trace.c",
    "tinynn/tinynn_events.h", "tinynn/tinynn_events.c",
    "tinynn/tinynn_backend_metal.m",
    "tinynn/tinynn_backend_cuda.c",
  ].reject { |f| File.directory?(f) || f =~ /\.(o|so|a|dylib|bundle)$/ }
   .select { |f| tracked.empty? || tracked.include?(f) }

  # bin/toy is a plain-MRI binstub (the CRuby CLI shell under
  # lib/toy/core/). RubyGems ships it from bin/. (Was a stale exe/[]
  # placeholder copied from tep; reconciled in favour of bin/toy, which
  # every roadmap/design ref and `toy new`'s own scaffold use.)
  s.bindir         = "bin"
  s.executables    = ["toy"]
  s.require_paths  = ["lib"]

  # No runtime dependencies — toy is Spinel-compiled. Build-time deps
  # (the spinelgems convention + bundler-spinel) are the consumer's
  # concern; same recipe as tep.gemspec.
end
