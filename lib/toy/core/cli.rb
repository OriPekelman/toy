# lib/toy/core/cli.rb — the `toy` command dispatcher.
#
# CRuby ONLY (the CLI shell). Idiomatic Ruby — Hash/case dispatch, NO
# Spinel DSL, NO Struct.new restriction. This file and everything under
# lib/toy/core/ are plain MRI Ruby; they must NEVER require_relative the
# Spinel-compiled libs (tinynn.rb, arch.rb, model_index.rb,
# transformer.rb, toy_describe_flow.rb) — those use `ffi_lib` directives
# that fail to load under CRuby. (toy_card.rb IS MRI-clean and is reused.)
#
# Single source of truth: the COMMANDS registry below drives `--help`,
# `--manifest`, AND dispatch — no drift between them.
#
# Exit-code contract (design §11):
#   0  ok
#   2  bad input (unknown command, missing required arg)
#   1  execution failure (GGUF unreadable, scaffold target exists, ...)

require_relative "../version"
require_relative "cli/exit_codes"
require_relative "cli/new"
require_relative "cli/list"
require_relative "cli/describe"
require_relative "cli/fetch"
require_relative "cli/install"
require_relative "cli/infer"
require_relative "cli/train"
require_relative "cli/eval"
require_relative "cli/serve"
require_relative "cli/research"
require_relative "cli/manifest"

module Toy
  module Core
    module CLI
      # Single source of truth. Each entry: the command class + a
      # machine/human description of its surface. The manifest is
      # generated from this; --help reads summaries from it.
      COMMANDS = {
        "new" => {
          class: New,
          summary: "Scaffold a conventional toy project tree",
          args:  [{ name: "path", required: true, desc: "target dir" }],
          flags: [{ name: "--lib", desc: "scaffold a library-composition project (Gemfile + experiment.rb) instead of an app" },
                  { name: "--force", desc: "overwrite a non-empty dir" },
                  { name: "--json", desc: "machine output" }]
        },
        "list" => {
          class: List,
          summary: "Find GGUF models in caches + project data/",
          args:  [],
          flags: [{ name: "--json", desc: "machine output" }]
        },
        "describe" => {
          class: Describe,
          summary: "Read GGUF metadata, render the arch-derived Card",
          args:  [{ name: "model", required: true, desc: "path to a .gguf file" }],
          flags: [{ name: "--json", desc: "machine output" }]
        },
        "fetch" => {
          class: Fetch,
          summary: "Download a GGUF from HuggingFace into the cache + data/ symlink",
          args:  [{ name: "hf-repo", required: true, desc: "HF repo id" },
                  { name: "file.gguf", required: false, desc: "GGUF filename in the repo" }],
          flags: [{ name: "--json", desc: "machine output" }]
        },
        "install" => {
          class: Install,
          summary: "Build/verify the CPU backend for this project",
          args:  [],
          flags: [{ name: "--json", desc: "machine output" }]
        },
        "infer" => {
          class: Infer,
          summary: "Generate text from a GGUF model (greedy decode)",
          args:  [{ name: "model", required: true, desc: "path to a .gguf file" }],
          flags: [{ name: "--prompt", desc: "prompt text (default \"Once upon a time\")" },
                  { name: "--prompt-ids", desc: "space-separated token IDs (tokenizer-less models; overrides --prompt)" },
                  { name: "--n", desc: "tokens to generate (default 16)" },
                  { name: "--device", desc: "cpu (default) | cuda | metal (macOS)" },
                  { name: "--json", desc: "machine output" }]
        },
        "train" => {
          class: Train,
          summary: "Train a model (records runs/<id>/ + loss curve)",
          args:  [{ name: "recipe", required: true, desc: "from-scratch | lora | warm-start | vit-tiny" }],
          flags: [{ name: "--steps", desc: "training steps (default 5)" },
                  { name: "--seed", desc: "random-init seed (default 0)" },
                  { name: "--arch", desc: "llama (default) | gpt2 (from-scratch, CPU)" },
                  { name: "--device", desc: "cpu (default) | cuda | metal (macOS)" },
                  { name: "--out", desc: "run dir override (default runs/<id>)" },
                  { name: "--json", desc: "machine output" }]
        },
        "research" => {
          class: Research,
          summary: "Run a research fixture lane (toy's own capability tests)",
          args:  [{ name: "subcommand", required: true, desc: "'train' | 'eval'" }],
          subcommands: [{ name: "train", desc: "train a research fixture lane (see docs/research/lanes.md)" },
                        { name: "eval", desc: "research evaluations (`eval lmc`)" }],
          flags: []
        },
        "eval" => {
          class: Eval,
          summary: "Score a GGUF model (per-token logprobs)",
          args:  [{ name: "model", required: true, desc: "path to a .gguf file" }],
          subcommands: [{ name: "ce", desc: "cross-entropy of a checkpoint over a corpus" },
                        { name: "lmc", desc: "two-checkpoint linear mode connectivity (research; prefer `toy research eval lmc`)" }],
          flags: [{ name: "--top-k", desc: "top-K logprobs to report (default 5)" },
                  { name: "--device", desc: "cpu (default) | cuda | metal (macOS)" },
                  { name: "--json", desc: "machine output" }]
        },
        "serve" => {
          class: Serve,
          summary: "Serve a GGUF model over an OpenAI-compatible HTTP API (CPU)",
          args:  [{ name: "model", required: true, desc: "path to a .gguf file" }],
          flags: [{ name: "--port", desc: "TCP port to bind (default 4567)" },
                  { name: "--name", desc: "model label in /v1/models (default GGUF basename)" }]
        }
      }.freeze

      GLOBAL_FLAGS = [
        { name: "--manifest", desc: "emit the machine-readable command manifest (JSON)" },
        { name: "--help",     desc: "show usage" },
        { name: "--version",  desc: "show the toy version" }
      ].freeze

      module_function

      def run(argv)
        argv = argv.dup

        # (1) Peel global flags BEFORE subcommand lookup so they work with
        #     no subcommand (e.g. `toy --manifest`, `toy --version`).
        return Manifest.new([]).run if argv.include?("--manifest")
        if argv.include?("--version")
          puts "toy #{Toy::VERSION}"
          return EXIT_OK
        end
        if argv.empty? || argv.first == "--help" || argv.first == "-h"
          print_usage($stdout)
          return argv.empty? ? EXIT_BAD_INPUT : EXIT_OK
        end

        # (2) Shift the subcommand token.
        name = argv.shift

        # (3) Unknown subcommand → usage to stderr + bad-input.
        entry = COMMANDS[name]
        unless entry
          $stderr.puts "toy: unknown command #{name.inspect}"
          print_usage($stderr)
          return EXIT_BAD_INPUT
        end

        # (4) `toy <cmd> --help`, GENERATED from this registry rather than
        #     hand-written per command. Every subcommand except `train` and
        #     `research` used to answer `unknown flag "--help"` — their flag
        #     parsers reach a `when /\A-/` catch-all that swallows it — so
        #     the only way to learn a command's arguments was to run it
        #     wrong and read the rejection.
        #
        #     Generated, because the registry ALREADY carries the summary,
        #     the args and the per-flag descriptions used by `--manifest`.
        #     Hand-writing eleven help blocks would be eleven more strings
        #     to drift out of step with the parsers, which is the failure
        #     this repo keeps paying for. A new command gets help for free.
        #
        #     train and research are EXEMPT and print their own: train has
        #     146 flags and needs the Data / Model / Optimization grouping,
        #     and research leads with the fixture lanes. Both are richer
        #     than a registry rendering, so the generic path must not
        #     shadow them.
        if (argv.include?("--help") || argv.include?("-h")) &&
           !OWN_HELP.include?(name)
          print_command_help($stdout, name, entry)
          return EXIT_OK
        end

        # (5) Instantiate the command with remaining argv; #run returns
        #     an Integer exit code. Each command parses its OWN flags.
        entry[:class].new(argv).run
      rescue Interrupt
        EXIT_FAILURE
      end

      # Commands that print their OWN --help and must not be shadowed by
      # the generated one. Keep this list SHORT: a command earns a place
      # here by having help the registry genuinely cannot render, not by
      # preference.
      OWN_HELP = %w[train research].freeze

      def print_command_help(io, name, entry)
        args  = entry[:args]  || []
        flags = entry[:flags] || []
        usage = args.map { |a| a[:required] ? "<#{a[:name]}>" : "[#{a[:name]}]" }
        io.puts "usage: toy #{name}#{usage.empty? ? '' : ' ' + usage.join(' ')}#{flags.empty? ? '' : ' [flags]'}"
        io.puts ""
        io.puts entry[:summary]
        unless args.empty?
          io.puts ""
          io.puts "Arguments:"
          args.each do |a|
            io.puts format("  %-14s %s%s", a[:name], a[:desc],
                           a[:required] ? "" : " (optional)")
          end
        end
        subs = entry[:subcommands] || []
        unless subs.empty?
          io.puts ""
          io.puts "Subcommands:"
          subs.each { |c| io.puts format("  %-14s %s", c[:name], c[:desc]) }
        end
        unless flags.empty?
          io.puts ""
          io.puts "Flags:"
          flags.each { |f| io.puts format("  %-14s %s", f[:name], f[:desc]) }
        end
        io.puts ""
        io.puts "Full reference: docs/cli.md"
      end

      def print_usage(io)
        io.puts "usage: toy <command> [args] [flags]"
        io.puts ""
        io.puts "Commands:"
        COMMANDS.each do |cmd_name, entry|
          io.puts format("  %-10s %s", cmd_name, entry[:summary])
        end
        io.puts ""
        io.puts "Global flags:"
        GLOBAL_FLAGS.each do |f|
          io.puts format("  %-12s %s", f[:name], f[:desc])
        end
      end
    end
  end
end
