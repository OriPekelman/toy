# lib/toy/core/cli/research.rb — the research namespace.
#
# toy is a framework built for ML research, so it carries two kinds of thing
# and telling them apart is the whole point of this file:
#
#   CAPABILITY — alternative credit assignment (`--policy chain|dfa|frozen`),
#     the feedback-matrix family (DfaB, Kolen-Pollack, nDFA, LDFA), optimizers,
#     LR schedules, checkpointing, the error-conditioning instrument. These are
#     framework features. They live on `toy train` and are documented in
#     docs/cli.md like any other capability.
#
#   FIXTURE — the eleven synthetic task lanes. They exist so toy can test those
#     capabilities INDEPENDENTLY of any one research question: a client asks for
#     a capability, toy builds it and proves it on a fixture, the capability
#     stays in the arsenal. Supported and gated, but not the product path — a
#     newcomer never needs them, and they should not be the second thing anyone
#     reads. They live here.
#
# THE SEAM IS PRESENTATIONAL, and that is a load-bearing property rather than a
# convenience. The CLI is a pure ENV-marshalling shim over per-lane compiled
# binaries; routing a lane through this namespace changes no runner, no ENV var
# and no graph. Every recorded cell stays byte-reproducible through either entry
# point, so a concluded programme's results are not put at risk by tidying. The
# migration and that guarantee are tao#25.
#
# Plain MRI Ruby — no require_relative into the Spinel tree (see cli.rb).

module Toy
  module Core
    module CLI
      class Research

        SUBCOMMANDS = {
          "train" => "Train a research fixture lane (see docs/research/lanes.md)",
          "eval"  => "Research evaluations (`eval lmc` — two-checkpoint linear mode connectivity)"
        }.freeze

        def initialize(argv)
          @argv = argv
        end

        def run
          sub = @argv.first

          if sub.nil? || sub == "--help" || sub == "-h"
            print_usage(sub.nil? ? $stderr : $stdout)
            return sub.nil? ? EXIT_BAD_INPUT : EXIT_OK
          end

          case sub
          when "train"
            Train.new(@argv.drop(1), research: true).run
          when "eval"
            # `toy research eval lmc ...` -> the existing Eval dispatcher,
            # which already routes its own `lmc` sub-subcommand.
            Eval.new(@argv.drop(1)).run
          else
            $stderr.puts "toy research: unknown subcommand #{sub.inspect}"
            print_usage($stderr)
            EXIT_BAD_INPUT
          end
        end

        private

        def print_usage(io)
          io.puts "usage: toy research <subcommand> [args] [flags]"
          io.puts ""
          io.puts "Subcommands:"
          SUBCOMMANDS.each { |n, d| io.puts format("  %-8s %s", n, d) }
          io.puts ""
          io.puts "Fixture lanes for `toy research train`:"
          Train::FIXTURE_RECIPES.each_slice(6) { |row| io.puts "  #{row.join('  ')}" }
          io.puts ""
          io.puts "These are toy's own capability tests, not the product path."
          io.puts "For training a model, see `toy train --help`."
          io.puts "Reference: docs/research/lanes.md"
        end
      end
    end
  end
end
