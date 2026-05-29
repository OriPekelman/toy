# lib/toy/core/cli/exit_codes.rb — the exit-code contract (design §11).
#
# Pulled into its own file so the command classes can require it without
# depending on cli.rb's load order (cli.rb requires the commands, so the
# constants must exist before those requires run).

module Toy
  module Core
    module CLI
      EXIT_OK        = 0
      EXIT_FAILURE   = 1
      EXIT_BAD_INPUT = 2
    end
  end
end
