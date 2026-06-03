# lib/toy_card.rb — structured representation of a Phuong–Hutter
# algorithm card.
#
# Background
# ----------
# Each transformer building block (LayerNorm, FFN, GQAttention, …) and
# each top-level model (Toy::GPT2, Toy::SmolLM2) has an `algorithm`
# method that returns a `Toy::Card`. The Card holds the algorithm as
# structured data — inputs, outputs, hyperparameters, parameters, a
# body of `Step` records — rather than a pre-rendered string.
#
# Renderers are pure functions on a Card:
#   * `render_pseudocode` → the human-readable Phuong–Hutter view
#     (what `algorithm_card` used to build by hand).
#
# Future renderers (each ~30 lines):
#   * `render_ruby` — emit a Ruby snippet that constructs the model.
#     The natural target for the round-trip parser
#     (prep/card_to_code.rb), since both sides then speak the same IR.
#   * `render_dot`  — draw the data-flow graph.
#
# Spinel friendliness
# -------------------
# All Card / Step / Item fields are concrete types (String / Int /
# Array<concrete>). Array fields use the literal-seed-then-pop pattern
# so Spinel pins them as PtrArray<ConcreteRow>. Step#kind is a String
# (not a Symbol) — fewer surprises in dispatch. Reserved-word methods
# (loop, return) are spelled `step_loop` / `step_return`.
#
# See docs/lowerer-design.md for how this IR also serves as the future
# target of a Prism-based source walker.

module Toy
  # One labelled row in the Input/Output/Param header.
  #
  #   add_input("x", "{1..V}^T", "token IDs")
  #     →  "  Input:    x ∈ {1..V}^T   (token IDs)"
  class CardItem
    attr_accessor :name_, :type_, :note

    def initialize(name_, type_, note)
      @name_ = name_
      @type_ = type_
      @note  = note
    end
  end

  # One `key=value` entry in the Hyper line.
  #
  #   add_hyper("D", "768")  →  contributes " D=768" to the Hyper line.
  class CardHyper
    attr_accessor :key, :value

    def initialize(key, value)
      @key   = key
      @value = value
    end
  end

  # One step in the algorithm body.
  #
  # `kind` discriminates how the renderer formats this step:
  #
  #   "bind"        var ← expr_str                  (numbered)
  #   "update"      var ← expr_str                  (numbered; same as bind, semantically distinct)
  #   "loop_open"   for <expr_str> do               (numbered; indents subsequent steps)
  #   "loop_close"  end for                         (numbered; outdents)
  #   "return"      return var                      (numbered)
  #   "comment"     ▷ note                          (un-numbered; for inline ▷ comments
  #                                                  attached to the previous step)
  #
  # `shape` is the right-side annotation (`e ∈ R^{T×D}` etc.); blank
  # skips it. `note` is an inline ▷ comment appended after the step.
  class Step
    attr_accessor :kind, :var, :expr_str, :shape, :note

    def initialize(kind, var, expr_str, shape, note)
      @kind     = kind
      @var      = var
      @expr_str = expr_str
      @shape    = shape
      @note     = note
    end
  end

  # A complete algorithm card.
  #
  #   c = Toy::Card.new("Toy::GPT2.forward(x, p_start)", "HF GPT-2 family")
  #   c.add_input("x", "{1..V}^T", "token IDs")
  #   c.add_output("P", "R^{T×V}", "logits")
  #   c.add_hyper("V", cfg.vocab.to_s)
  #   c.add_param("W_e", "R^{V×D}", "")
  #   c.add_param_extra("(total " + Toy.fmt_count(param_count) + ", tied)")
  #   c.step_bind("e", "W_e[x] + W_p[p_start : p_start+T]", "e ∈ R^{T×D}")
  #   c.step_loop("ℓ ← 1, …, N", "")
  #   c.step_update("e", "e + Attn(LN(e; γ_ℓ^1, β_ℓ^1, ε); θ_ℓ^attn)", "e ∈ R^{T×D}", "")
  #   c.step_loop_close
  #   c.step_return("P")
  #   puts c.render_pseudocode
  class Card
    attr_accessor :name_, :family,
                  :inputs, :outputs, :hypers, :params, :param_extras,
                  :steps

    def initialize(name_, family)
      @name_  = name_
      @family = family

      # Seed-then-pop pins each Array as PtrArray<ConcreteRow> in
      # Spinel's type inference. Without the seed it would be
      # untyped-empty and the first push would lock it in incorrectly.
      @inputs = [Toy::CardItem.new("", "", "")]
      @inputs.pop
      @outputs = [Toy::CardItem.new("", "", "")]
      @outputs.pop
      @hypers = [Toy::CardHyper.new("", "")]
      @hypers.pop
      @params = [Toy::CardItem.new("", "", "")]
      @params.pop
      @param_extras = [""]
      @param_extras.pop
      @steps = [Toy::Step.new("bind", "", "", "", "")]
      @steps.pop
    end

    # --- builder API --------------------------------------------------

    def add_input(name_, type_, note)
      @inputs.push(Toy::CardItem.new(name_, type_, note))
    end

    def add_output(name_, type_, note)
      @outputs.push(Toy::CardItem.new(name_, type_, note))
    end

    def add_hyper(key, value)
      @hypers.push(Toy::CardHyper.new(key, value))
    end

    def add_param(name_, type_, note)
      @params.push(Toy::CardItem.new(name_, type_, note))
    end

    # Free-text continuation lines under "Param:" (e.g. "(total Xxx)").
    def add_param_extra(line)
      @param_extras.push(line)
    end

    def step_bind(var, expr_str, shape)
      @steps.push(Toy::Step.new("bind", var, expr_str, shape, ""))
    end

    def step_update(var, expr_str, shape, note)
      @steps.push(Toy::Step.new("update", var, expr_str, shape, note))
    end

    # `header` is the loop header without "for"/"do" — the renderer
    # adds those. `note` is an inline ▷ comment shown beside the line.
    def step_loop(header, note)
      @steps.push(Toy::Step.new("loop_open", "", header, "", note))
    end

    def step_loop_close
      @steps.push(Toy::Step.new("loop_close", "", "", "", ""))
    end

    def step_return(var)
      @steps.push(Toy::Step.new("return", var, "", "", ""))
    end

    # --- renderer (pseudocode) ----------------------------------------

    # Right-pad `body` to `col` characters then append `tail` — only
    # if `tail` is non-empty. Used to align shape annotations.
    def self.pad_to(body, col, tail)
      return body if tail == ""
      n = body.length
      if n >= col
        body + "   " + tail
      else
        body + (" " * (col - n)) + tail
      end
    end

    SHAPE_COL = 70  # target column for the shape annotation

    def render_pseudocode
      s = "Algorithm: " + @name_
      if @family != ""
        # Tag in square brackets, padded with a few spaces.
        pad = "      "
        s = s + pad + "[" + @family + "]"
      end
      s = s + "\n"

      # Input / Output / Hyper / Param headers.
      first = true
      @inputs.each do |it|
        prefix = first ? "  Input:    " : "            "
        line = prefix + it.name_ + " ∈ " + it.type_
        if it.note != ""
          line = line + "   (" + it.note + ")"
        end
        s = s + line + "\n"
        first = false
      end

      first = true
      @outputs.each do |it|
        prefix = first ? "  Output:   " : "            "
        line = prefix + it.name_ + " ∈ " + it.type_
        if it.note != ""
          line = line + "   (" + it.note + ")"
        end
        s = s + line + "\n"
        first = false
      end

      if @hypers.length > 0
        line = "  Hyper:   "
        @hypers.each do |h|
          line = line + " " + h.key + "=" + h.value
        end
        s = s + line + "\n"
      end

      first = true
      @params.each do |it|
        prefix = first ? "  Param:    " : "            "
        line = prefix + it.name_ + " ∈ " + it.type_
        if it.note != ""
          line = line + "   (" + it.note + ")"
        end
        s = s + line + "\n"
        first = false
      end
      @param_extras.each do |line|
        s = s + "            " + line + "\n"
      end

      # Body: walk steps, track step number and indent depth.
      n = 1
      depth = 0
      @steps.each do |st|
        kind = st.kind
        if kind == "loop_close"
          depth = depth - 1
        end
        indent = "   " * depth
        head = "  " + n.to_s.rjust(2) + ": " + indent
        body = ""

        if kind == "bind" || kind == "update"
          body = head + st.var + " ← " + st.expr_str
        elsif kind == "loop_open"
          body = head + "for " + st.expr_str + " do"
        elsif kind == "loop_close"
          body = head + "end for"
        elsif kind == "return"
          body = head + "return " + st.var
        elsif kind == "comment"
          body = head + "▷ " + st.note
        else
          body = head + st.expr_str
        end

        tail = st.shape
        if st.note != "" && kind != "loop_close" && kind != "comment"
          if tail == ""
            tail = "▷ " + st.note
          else
            tail = tail + "   ▷ " + st.note
          end
        end

        s = s + Toy::Card.pad_to(body, Toy::Card::SHAPE_COL, tail) + "\n"

        if kind == "loop_open"
          depth = depth + 1
        end
        n = n + 1
      end

      s
    end

    # --- summary helpers ----------------------------------------------

    # Look up a hyper by key. Returns "" if absent — used by the
    # round-trip emitter (prep/card_to_code.rb) so it can read the
    # IR directly instead of regexing the rendered text.
    def hyper(key)
      @hypers.each do |h|
        return h.value if h.key == key
      end
      ""
    end
  end
end
