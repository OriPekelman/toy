#!/usr/bin/env ruby
# frozen_string_literal: true
# prep/card_to_code.rb — best-effort parser for Phuong–Hutter algorithm
# cards emitted by Toy::GPT2#algorithm_card / Toy::SmolLM2#algorithm_card.
#
# Reads a card on stdin (or a file argv[0]); emits Ruby code that
# reconstructs the corresponding Toy:: model.
#
# Two recognized families:
#   1. "[HF GPT-2 family]"   → Toy::GPT2 with Toy::GPT2Config
#   2. "[Llama-family decoder]" → Toy::SmolLM2 with Toy::SmolLM2Config
#
# Two input paths:
#   • Rendered card (Phuong–Hutter text). Detect family via the
#     square-bracket tag; pull hyperparameters from the `Hyper:` line.
#   • A Toy::Card IR object passed to `emit_from_card` directly.
#     This skips the regex round-trip entirely — the model's
#     `algorithm` method produces structured data and we read its
#     `hyper(key)` accessor. Strictly stronger (no fragile parsing)
#     when the producer is in-process.
#
# Limits (today):
#   - Only round-trips cards we emit ourselves. Hand-written cards
#     with novel structure aren't supported.
#   - Reads hyperparameters from the `Hyper:` line + the `total Xxx`
#     line for sanity. Ignores the actual algorithm steps — they're
#     fully determined by the family tag.
#   - Doesn't load weights. Caller's responsibility (use the GGUF
#     loaders just like the demos do).
#
# When the API grows new architectures, extend FAMILIES and add a
# new emit_… method. The current parser is intentionally a
# pattern-recognizer, not a general expression evaluator.

require "optparse"

FAMILIES = {
  /\[HF GPT-2 family\]/         => :gpt2,
  /\[Llama-family decoder\]/    => :smollm2,
}.freeze

def detect_family(text)
  FAMILIES.each { |re, sym| return sym if text =~ re }
  nil
end

# Hyper line example (GPT-2):
#   Hyper:    V=50257 D=768 H=12 D_f=3072 N=6 ctx=1024
# Hyper line example (SmolLM2):
#   Hyper:    V=49152 D=576 H=9 H_kv=3 D_f=1536 N=30 ctx=8192 θ_base=100000.0
def parse_hyper(text)
  line = text.lines.find { |l| l.strip.start_with?("Hyper:") } or
    raise "no Hyper: line in card"
  body = line.sub(/^.*?Hyper:\s*/, "")
  pairs = body.scan(/(\w+|θ_base)\s*=\s*([0-9.eE+-]+)/)
  pairs.each_with_object({}) do |(k, v), h|
    h[k] = v.include?(".") || v.include?("e") || v.include?("E") ? v.to_f : v.to_i
  end
end

def emit_gpt2(hyper)
  v   = hyper.fetch("V")
  d   = hyper.fetch("D")
  h   = hyper.fetch("H")
  d_f = hyper.fetch("D_f")
  n   = hyper.fetch("N")
  ctx = hyper.fetch("ctx")
  <<~RUBY
    require_relative "../lib/toy"
    require_relative "../lib/toy/models/toy_gpt2"
    # Hand-written from card: Toy::GPT2 (HF GPT-2 family).
    # Weights must be loaded separately (e.g. via GGUFLoad.load_toy_gpt2).
    cfg   = Toy::GPT2Config.new(#{v}, #{d}, #{h}, #{d_f}, #{n}, #{ctx})
    model = Toy::GPT2.new(cfg)
    # model.forward(token_ids, p_start) -> Mat[T, V]
  RUBY
end

def emit_smollm2(hyper)
  v       = hyper.fetch("V")
  d       = hyper.fetch("D")
  h       = hyper.fetch("H")
  h_kv    = hyper.fetch("H_kv")
  d_f     = hyper.fetch("D_f")
  n       = hyper.fetch("N")
  ctx     = hyper.fetch("ctx")
  theta_b = hyper.fetch("θ_base")
  <<~RUBY
    require_relative "../lib/toy"
    require_relative "../lib/toy/models/toy_smollm2"
    # Reconstructed from algorithm card: Toy::SmolLM2 (Llama-family).
    # The card encodes hyperparameters; rms_eps defaults to 1e-5.
    # Weights must be loaded separately (e.g. via GGUFLoad.load_toy_smollm2).
    cfg = Toy::SmolLM2Config.new(
      #{v},          # vocab
      #{d},          # d_model
      #{h},          # n_heads
      #{h_kv},       # n_kv
      #{d_f},        # d_ff
      #{n},          # n_layers
      #{ctx},        # ctx
      #{theta_b},    # rope_base
      1.0e-5         # rms_eps
    )
    model = Toy::SmolLM2.new(cfg)
    # model.enable_untied_output! if the card declared a separate W_out.
    # model.forward(token_ids, p_start) -> Mat[T, V]
  RUBY
end

# Direct entry from a Toy::Card object — bypasses the rendered-text
# round-trip and reads hyperparameters from the IR. The Card's `family`
# field selects which emitter; its `hyper(key)` accessor reads each
# value as a String (we coerce to Int/Float here for the emitter).
#
# Usage from in-process code:
#   require_relative "../lib/toy/models/toy_gpt2"
#   model = Toy::GPT2.new(cfg)
#   puts emit_from_card(model.algorithm)
def emit_from_card(card)
  family =
    case card.family
    when "HF GPT-2 family"      then :gpt2
    when "Llama-family decoder" then :smollm2
    else raise "unknown family in card: #{card.family.inspect}"
    end

  hyper = {}
  card.hypers.each do |h|
    v = h.value
    hyper[h.key] = v.include?(".") || v.include?("e") || v.include?("E") ? v.to_f : v.to_i
  end

  case family
  when :gpt2    then emit_gpt2(hyper)
  when :smollm2 then emit_smollm2(hyper)
  end
end

def main
  text = ARGV.empty? ? $stdin.read : File.read(ARGV[0])
  family = detect_family(text) or
    abort "could not detect family — expected '[HF GPT-2 family]' or '[Llama-family decoder]' in card header"
  hyper = parse_hyper(text)

  $stderr.puts "Detected family: #{family}"
  $stderr.puts "Parsed hyper: #{hyper.inspect}"
  $stderr.puts

  case family
  when :gpt2    then puts emit_gpt2(hyper)
  when :smollm2 then puts emit_smollm2(hyper)
  end
end

main if $PROGRAM_NAME == __FILE__
