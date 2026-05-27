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

  # The gem ships:
  #   - Every .rb under lib/ (the primitives consumers compose)
  #   - tinynn/*.{h,c} (the FFI C shim — consumers link against the
  #     prebuilt .a files in toy's tree, but the headers are needed at
  #     compile time for some tooling)
  #   - The .gitignore-exempted README + LICENSE
  # We do NOT ship:
  #   - vendor/ggml/build/* (~6 MB of build artifacts; consumers point
  #     their link paths at toy's in-place tree via the FFI manifest)
  #   - data/* (model weights / pretokenized corpora — caller-managed)
  #   - tep_demo/_tep_lib/ + similar build scratch
  s.files = Dir[
    "README.md", "LICENSE", "CHANGELOG.md",
    "lib/**/*.rb",
    "tinynn/tinynn_ggml.h",
    "tinynn/tinynn_ggml.c",
    "tinynn/tinynn_gguf.h",
    "tinynn/tinynn_gguf.c",
    "tinynn/tinynn_events.h",
    "tinynn/tinynn_events.c",
  ].reject { |f| File.directory?(f) }

  s.bindir         = "exe"
  s.executables    = []
  s.require_paths  = ["lib"]

  # No runtime dependencies — toy is Spinel-compiled. Build-time deps
  # (the spinelgems convention + bundler-spinel) are the consumer's
  # concern; same recipe as tep.gemspec.
end
