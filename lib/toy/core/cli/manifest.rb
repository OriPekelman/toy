# lib/toy/core/cli/manifest.rb — `toy --manifest`.
#
# Emit a machine-readable description of the CLI surface as JSON,
# generated FROM the same COMMANDS registry the dispatcher uses (single
# source of truth — no drift between --help, --manifest, and dispatch).
#
# Format tag is 'toy/manifest-v1' — DISTINCT from the 'toy/v1' event /
# describe-flow tag to avoid collision.

require "json"
require_relative "exit_codes"
require_relative "../../version"

module Toy
  module Core
    module CLI
      class Manifest
        def initialize(argv)
          @argv = argv
        end

        def run
          commands = COMMANDS.map do |name, entry|
            {
              "name" => name,
              "summary" => entry[:summary],
              "args" => entry[:args],
              "flags" => entry[:flags]
            }
          end

          manifest = {
            "format" => "toy/manifest-v1",
            "toy_version" => Toy::VERSION,
            "commands" => commands,
            "global_flags" => GLOBAL_FLAGS,
            "exit_codes" => {
              "ok" => EXIT_OK,
              "bad_input" => EXIT_BAD_INPUT,
              "failure" => EXIT_FAILURE
            }
          }
          puts JSON.pretty_generate(manifest)
          EXIT_OK
        end
      end
    end
  end
end
