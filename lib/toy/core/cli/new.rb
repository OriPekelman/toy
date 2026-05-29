# lib/toy/core/cli/new.rb — `toy new <path>`.
#
# Scaffold the conventional toy project tree. CRuby ONLY.
#
#   myproj/
#   ├── toy.yml          # minimal config (run_id_template + algos_path)
#   ├── algos/           # user code, same 4-layer shape as framework llm/
#   │   ├── primitives/.keep   (L1 custom ops — rare)
#   │   ├── blocks/.keep       (L2 custom blocks — rare)
#   │   ├── archs/.keep        (L3 custom architectures — common)
#   │   └── recipes/.keep      (L4 training plans / curricula — common)
#   ├── data/.keep       # GGUFs (HF-cache symlinks ok), corpora
#   ├── runs/.keep       # event streams + checkpoints (Tao reads here)
#   └── bin/toy          # courtesy binstub (thin ruby stub)
#
# The 4 algos/ subdirs are the documented convention but optional at use
# time — discovery also accepts a single algos/my_llama.rb.

require "fileutils"
require "json"
require_relative "exit_codes"

module Toy
  module Core
    module CLI
      class New
        TOY_YML = <<~YAML
          # toy.yml — minimal project config. An empty file is valid; all
          # defaults then apply.

          # Template for runs/<run_id>/ directory names. Recognised brace
          # tokens: {arch} {date}(YYYYMMDD) {time}(HHMMSS) {seq}(daily counter).
          run_id_template: "{arch}-{date}-{seq}"

          # Relative dir the framework discovers your algos in (L1-L4).
          # A single algos/my_llama.rb works too; the subdirs are optional.
          algos_path: "algos"
        YAML

        BINSTUB = <<~RUBY
          #!/usr/bin/env ruby
          # Project-local binstub. The gem-installed `toy` works without this;
          # it is a courtesy so `./bin/toy` resolves inside the project too.
          require "toy/core/cli"
          exit Toy::Core::CLI.run(ARGV)
        RUBY

        ALGO_SUBDIRS = %w[primitives blocks archs recipes].freeze

        def initialize(argv)
          @argv = argv
          @json = false
          @force = false
          @path = nil
        end

        def run
          return EXIT_BAD_INPUT unless parse_args
          target = File.expand_path(@path)

          if File.exist?(target) && !Dir.empty?(target) && !@force
            return fail_out("target #{target.inspect} exists and is not empty (use --force)")
          end

          created = scaffold(target)

          if @json
            puts JSON.pretty_generate(
              "format" => "toy/new-v1",
              "path" => target,
              "created" => created
            )
          else
            puts "Created toy project at #{target}"
            created.each { |rel| puts "  #{rel}" }
            puts ""
            puts "Next: cd #{@path} && toy list"
          end
          EXIT_OK
        rescue SystemCallError => e
          fail_out("could not create project: #{e.message}")
        end

        private

        def parse_args
          rest = []
          @argv.each do |tok|
            case tok
            when "--json"  then @json = true
            when "--force" then @force = true
            when /\A-/
              $stderr.puts "toy new: unknown flag #{tok.inspect}"
              return false
            else rest << tok
            end
          end
          if rest.empty?
            $stderr.puts "toy new: missing required argument <path>"
            return false
          end
          @path = rest.first
          true
        end

        # Returns the list of relative paths created (dirs + files).
        def scaffold(target)
          created = []
          dirs = ["algos"] +
                 ALGO_SUBDIRS.map { |d| "algos/#{d}" } +
                 %w[data runs bin]
          dirs.each do |rel|
            FileUtils.mkdir_p(File.join(target, rel))
            created << "#{rel}/"
          end

          # .keep files so empty dirs survive git.
          keep_dirs = ALGO_SUBDIRS.map { |d| "algos/#{d}" } + %w[data runs]
          keep_dirs.each do |rel|
            write_file(File.join(target, rel, ".keep"), "")
            created << "#{rel}/.keep"
          end

          write_file(File.join(target, "toy.yml"), TOY_YML)
          created << "toy.yml"

          binstub = File.join(target, "bin", "toy")
          write_file(binstub, BINSTUB)
          File.chmod(0o755, binstub)
          created << "bin/toy"

          created
        end

        def write_file(path, content)
          File.write(path, content)
        end

        def fail_out(msg)
          if @json
            puts JSON.pretty_generate("format" => "toy/new-v1", "error" => msg)
          else
            $stderr.puts "toy new: #{msg}"
          end
          EXIT_FAILURE
        end
      end
    end
  end
end
