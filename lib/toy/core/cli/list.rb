# lib/toy/core/cli/list.rb — `toy list`.
#
# Find GGUF models across the project's data/, ./models, env override,
# and standard caches (HF / Ollama / LM Studio). CRuby ONLY — wraps the
# pure-Ruby ModelScan port (no ggml FFI, no built binary).
#
# UX delta vs the old llama-only binary: lists ALL *.gguf; unparseable
# or non-llama files degrade to family=unknown, params=0 rather than
# being silently dropped.

require "json"
require_relative "exit_codes"
require_relative "../model_scan"

module Toy
  module Core
    module CLI
      class List
        def initialize(argv)
          @argv = argv
          @json = false
        end

        def run
          return EXIT_BAD_INPUT unless parse_args
          entries = ModelScan.scan

          if @json
            emit_json(entries)
          else
            emit_human(entries)
          end
          EXIT_OK
        end

        private

        def parse_args
          @argv.each do |tok|
            case tok
            when "--json" then @json = true
            else
              $stderr.puts "toy list: unknown argument #{tok.inspect}"
              return false
            end
          end
          true
        end

        def emit_json(entries)
          puts JSON.pretty_generate(
            "format" => "toy/list-v1",
            "sources" => ModelScan.default_sources,
            "models" => entries.map { |e|
              {
                "name" => e.name,
                "path" => e.path,
                "family" => e.family.to_s,
                "n_params" => e.n_params,
                "params" => e.params_summary,
                "size_b" => e.size_b,
                "size" => e.size_summary,
                "source" => e.source
              }
            }
          )
        end

        def emit_human(entries)
          if entries.empty?
            puts "No GGUF models found."
            puts "Set TOY_MODEL_DIR to a directory containing them, or place"
            puts "*.gguf under ./data, or download via huggingface-cli / ollama pull."
            return
          end
          puts "Found #{entries.length} model(s):"
          entries.each do |e|
            puts "  [#{e.source}] #{e.name}  " \
                 "#{e.family} · #{e.params_summary} · #{e.size_summary}"
            puts "    #{e.path}"
          end
        end
      end
    end
  end
end
