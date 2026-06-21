# lib/toy/llm/archs/layer_spec.rb — Phase 3 of the Dragon-GDN arc: the
# per-layer descriptor that lets one arch forward loop build a heterogeneous
# layer stack via flat-int kind dispatch (no polymorphic receiver). See
# docs/roadmap/dragon-gdn-arch-2026-06-20.md.

module Toy; module LLM; module Archs
  # Per-layer descriptor — the seam that lets ONE arch forward loop build a
  # heterogeneous layer stack (homogeneous Llama attention today; Dragon's
  # Gated-DeltaNet + selective-attention mix from Phase 5) WITHOUT polymorphic
  # method dispatch.
  #
  # The `kind` field is a FLAT INTEGER, deliberately not a class, symbol, or
  # block object. The arch loop branches on `spec.kind == KIND_*` and then
  # calls a CONCRETE typed block method inside each branch, so every
  # `.build_forward` call site keeps a single receiver class. Funnelling
  # heterogeneous receiver types through ONE call site is the Spinel
  # poly-dispatch landmine (the #11/#12 family, matz/spinel#1043) the whole
  # Dragon seam is shaped to avoid — see dragon-gdn-arch-2026-06-20.md
  # "Phase 3 — the per-layer descriptor seam."
  #
  # Hand-written positional class, NEVER Struct.new (landmine #16 / #1043): a
  # Struct's synthesized accessors unify across modules and miscompile
  # unrelated callers, exactly like LlamaArchForwardOut / TransformerBlockCtx.
  # Carries values, no behavior.
  class LayerSpec
    # Layer kinds. Flat ints so the dispatch branch stays monomorphic. The
    # Phase-3 refactor gate only exercises KIND_ATTENTION (every layer); the
    # GDN kind is reserved here so the seam shape is fixed before Phase 5
    # actually wires a Gated-DeltaNet block into a branch.
    KIND_ATTENTION = 0   # standard Llama-style attention + SwiGLU FFN
    KIND_GDN       = 1   # Dragon Gated-DeltaNet block (Phase 5)

    attr_accessor :kind

    def initialize(kind)
      @kind = kind
    end
  end
end; end; end
