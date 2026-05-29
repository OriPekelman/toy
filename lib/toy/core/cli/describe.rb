# lib/toy/core/cli/describe.rb — `toy describe <model.gguf>`.
#
# Read a GGUF's metadata (pure-Ruby header parse, no ggml session) and
# render the ARCH-DERIVED Phuong-Hutter Card via the MRI-clean
# lib/toy_card.rb renderer.
#
# SCOPE NOTE (deviation from roadmap wording :72): this is NOT
# ToyDescribeFlow.card — that is a graph-walk over a realized ggml
# session (Spinel-only, needs full weight load, counts-only ceiling). The
# CLI's describe builds the Card from GGUF *metadata* + the known
# transformer shape. It reuses toy_card.rb's renderer + the arch.rb key
# list / family logic (ported in ModelScan), not the heavyweight flow.

require "json"
require_relative "exit_codes"
require_relative "../model_scan"
require_relative "../gguf_meta"

# toy_card.rb is MRI-clean (verified: no ffi_lib). Reused verbatim.
require_relative "../../../toy_card"

module Toy
  module Core
    module CLI
      class Describe
        def initialize(argv)
          @argv = argv
          @json = false
          @model = nil
        end

        def run
          return EXIT_BAD_INPUT unless parse_args
          path = File.expand_path(@model)

          unless File.file?(path)
            return fail_out("no such file: #{path}")
          end

          begin
            meta = GGUFMeta.read(path)
          rescue GGUFMeta::ParseError => e
            return fail_out("could not parse GGUF: #{e.message}")
          end

          arch = ModelScan.read_arch(meta)
          if arch.nil?
            return fail_out("#{path}: no llama-family arch metadata (non-llama GGUF?)")
          end
          arch[:n_params] = ModelScan.estimate_params(arch)

          if @json
            emit_json(path, arch)
          else
            emit_human(path, arch)
          end
          EXIT_OK
        end

        private

        def parse_args
          rest = []
          @argv.each do |tok|
            case tok
            when "--json" then @json = true
            when /\A-/
              $stderr.puts "toy describe: unknown flag #{tok.inspect}"
              return false
            else rest << tok
            end
          end
          if rest.empty?
            $stderr.puts "toy describe: missing required argument <model.gguf>"
            return false
          end
          @model = rest.first
          true
        end

        def emit_human(path, arch)
          card = build_card(arch)
          puts card.render_pseudocode
          puts ""
          puts "Source: #{path}"
          puts "Family: #{arch[:family]} (header-derived; coarse)"
          puts "Params: ~#{fmt_count(arch[:n_params])}"
        end

        def emit_json(path, arch)
          puts JSON.pretty_generate(
            "format" => "toy/describe-v1",
            "path" => path,
            "family" => arch[:family].to_s,
            "n_params" => arch[:n_params],
            "arch" => {
              "vocab" => arch[:vocab],
              "d_model" => arch[:d_model],
              "n_layers" => arch[:n_layers],
              "n_heads_q" => arch[:n_q],
              "n_heads_kv" => arch[:n_kv],
              "d_head" => arch[:d_head],
              "d_ff" => arch[:d_ff],
              "context_length" => arch[:ctx],
              "rope_freq_base" => arch[:rope_base],
              "rms_eps" => arch[:rms_eps],
              "untied_lm_head" => arch[:untied],
              "moe" => arch[:moe],
              "n_experts" => arch[:n_experts],
              "n_experts_used" => arch[:n_experts_used]
            },
            "pseudocode" => build_card(arch).render_pseudocode
          )
        end

        # Build the arch-derived Card. Mirrors a SwiGLU/GQA/RMSNorm Llama-
        # family forward pass with the dims read from the GGUF header.
        def build_card(arch)
          c = Toy::Card.new("Toy::#{family_label(arch)}.forward(x)", arch[:family].to_s)
          c.add_input("x", "{1..V}^T", "token IDs")
          c.add_output("P", "R^{T×V}", "next-token logits")

          c.add_hyper("V", arch[:vocab].to_s)
          c.add_hyper("D", arch[:d_model].to_s)
          c.add_hyper("L", arch[:n_layers].to_s)
          c.add_hyper("n_q", arch[:n_q].to_s)
          c.add_hyper("n_kv", arch[:n_kv].to_s)
          c.add_hyper("d_head", arch[:d_head].to_s)
          c.add_hyper("d_ff", arch[:d_ff].to_s)
          c.add_hyper("ctx", arch[:ctx].to_s)
          c.add_hyper("MoE", arch[:moe] ? "#{arch[:n_experts_used]}/#{arch[:n_experts]}" : "no")

          c.add_param("W_e", "R^{V×D}", "token embedding")
          c.add_param("θ_ℓ^attn", "GQA(n_q,n_kv)", "per layer")
          c.add_param("θ_ℓ^ffn", arch[:moe] ? "MoE-SwiGLU" : "SwiGLU", "per layer")
          c.add_param(arch[:untied] ? "W_u" : "W_e^T",
                      "R^{D×V}", arch[:untied] ? "untied unembed" : "tied unembed")
          c.add_param_extra("(total ~#{fmt_count(arch[:n_params])} params)")

          c.step_bind("e", "W_e[x]", "e ∈ R^{T×D}")
          c.step_loop("ℓ ← 1, …, L", "")
          c.step_update("e", "e + GQA(RMSNorm(e); θ_ℓ^attn, RoPE)", "e ∈ R^{T×D}", "")
          c.step_update("e", "e + FFN(RMSNorm(e); θ_ℓ^ffn)", "e ∈ R^{T×D}",
                        arch[:moe] ? "top-#{arch[:n_experts_used]} experts" : "")
          c.step_loop_close
          c.step_bind("ê", "RMSNorm(e)", "ê ∈ R^{T×D}")
          c.step_bind("P", arch[:untied] ? "ê · W_u" : "ê · W_e^T", "P ∈ R^{T×V}")
          c.step_return("P")
          c
        end

        def family_label(arch)
          arch[:family] == :qwen2 ? "Qwen2" : "Llama"
        end

        def fmt_count(n)
          if n >= 1_000_000_000
            format("%.1fB", n.to_f / 1e9)
          elsif n >= 1_000_000
            format("%.1fM", n.to_f / 1e6)
          else
            n.to_s
          end
        end

        def fail_out(msg)
          if @json
            puts JSON.pretty_generate("format" => "toy/describe-v1", "error" => msg)
          else
            $stderr.puts "toy describe: #{msg}"
          end
          EXIT_FAILURE
        end
      end
    end
  end
end
