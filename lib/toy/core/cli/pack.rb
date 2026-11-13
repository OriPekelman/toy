# lib/toy/core/cli/pack.rb — `toy pack <model.gguf> -o <artifact> [--prefix F] [--name NAME] [--device cpu|cuda] [--ctx N] [--port N] [--stop-ids N,M]`.
#
# Builds ONE executable that carries the model, its tokenizer (inside the
# GGUF), a fixed serve config, a description, and optionally a precomputed
# KV prefix, and that starts the OpenAI-compatible server with no arguments
# (toy#211). CRuby only.
#
# The artifact is the serve runner (libexec/toy-serve or -cuda) with files
# appended after the ELF image:
#
#   [runner][pad][model][pad][kv][pad][config][pad][describe][pad][index][trailer]
#
# Every entry starts on a 64 KiB boundary and is followed by zeros up to the
# next one. The index is text ("<name> <offset> <length>\n" per entry) and
# the trailer is "TOYPACK1" + index offset + index length (u64 LE). The
# reading side is tinynn/tinynn_gguf.c: the runner finds the trailer in its
# own executable, maps the model from there (weights are read in place, as
# from a file), and restores the KV prefix from there.
#
# The KV prefix is produced by running the SAME runner once on the model
# with SERVE_BOOT_ONLY=1, so it is exactly the file `toy serve
# --prefix-file F --prefix-cache K` would write on this machine and device.
#
# Exit codes: 2 bad input, 1 failure, 0 artifact written and verified.

require "digest"
require "fileutils"
require "open3"
require "stringio"
require "tmpdir"
require_relative "exit_codes"
require_relative "../toy_root"
require_relative "describe"

module Toy
  module Core
    module CLI
      class Pack
        ALIGN   = 65_536
        MAGIC   = "TOYPACK1"
        DEFAULT_PORT = 4567

        def initialize(argv)
          @argv = argv
          @model = nil
          @out = nil
          @prefix = nil
          @name = nil
          @device = "cpu"
          @ctx = nil
          @port = DEFAULT_PORT
          @stop_ids = nil
        end

        def run
          parsed = parse_args
          return parsed unless parsed == true

          model = File.expand_path(@model)
          return bad_arg("no such file: #{model}") unless File.file?(model)
          prefix = @prefix && File.expand_path(@prefix)
          return bad_arg("no such file: #{prefix}") if prefix && !File.file?(prefix)
          out = File.expand_path(@out)
          return bad_arg("#{out} is a directory") if File.directory?(out)
          return bad_arg("no such directory: #{File.dirname(out)}") unless File.directory?(File.dirname(out))
          name = @name || File.basename(model).sub(/\.gguf\z/, "")
          return bad_arg("--name must not contain whitespace or '=': #{name.inspect}") if name =~ /[\s=]/

          root = ToyRoot.locate_root
          return fail_out("could not locate toy's install root (set TOY_HOME to a toy checkout)") unless root
          target = @device == "cuda" ? "libexec/toy-serve-cuda" : "libexec/toy-serve"
          ok, err = ToyRoot.ensure_built(root, target)
          return fail_out(err) unless ok
          runner = File.join(root, target)
          return fail_out("runner missing after build: #{runner}") unless File.executable?(runner)

          Dir.mktmpdir("toy-pack-", File.dirname(out)) do |tmp|
            kv = nil
            prefix_tokens = nil
            if prefix
              kv = File.join(tmp, "prefix.kv")
              $stderr.puts "toy pack: evaluating the prefix on #{@device}..."
              env = { "MODEL_PATH" => model, "MODEL_NAME" => name, "PREFIX_FILE" => prefix,
                      "PREFIX_CACHE" => kv, "SERVE_BOOT_ONLY" => "1",
                      "KV_FUSED" => nil, "KV_BATCH" => nil, "KV_REUSE" => nil,
                      "TOY_RUN_DIR" => nil, "TAO_RUN_DIR" => nil, "STOP_IDS" => @stop_ids }
              env["MAX_T"] = @ctx ? @ctx.to_s : nil
              log, st = Open3.capture2e(env, runner)
              unless st.success? && File.file?(kv)
                return fail_out("the runner could not build the KV prefix:\n#{log.lines.last(8).join}")
              end
              prefix_tokens = log[/KV prefix: evaluated (\d+) tokens/, 1]
              return fail_out("the runner did not report the prefix evaluation:\n#{log.lines.last(8).join}") if prefix_tokens.nil?
            end

            config = +"MODEL_NAME=#{name}\nPORT=#{@port}\nDEVICE=#{@device}\n"
            config << "MAX_T=#{@ctx}\n" if @ctx
            config << "STOP_IDS=#{@stop_ids}\n" if @stop_ids
            $stderr.puts "toy pack: hashing the model..."
            describe = describe_text(root, model, name, runner, target, prefix, prefix_tokens, config)

            entries = [["model", model]]
            entries << ["kv", kv] if kv
            cfg_path = File.join(tmp, "config");   File.binwrite(cfg_path, config)
            dsc_path = File.join(tmp, "describe"); File.binwrite(dsc_path, describe)
            entries << ["config", cfg_path] << ["describe", dsc_path]

            staged = File.join(tmp, "artifact")
            $stderr.puts "toy pack: writing #{out}..."
            write_artifact(staged, runner, entries)
            File.chmod(0o755, staged)

            # The artifact must describe itself with the text just embedded,
            # read back through the same code path it will serve with.
            back, _err, st = Open3.capture3({ "PORT" => nil }, staged, "--describe")
            unless st.success? && back.strip == describe.strip
              return fail_out("the packed artifact does not read itself back (--describe exited " \
                              "#{st.exitstatus}):\n#{back.lines.first(5).join}")
            end
            File.rename(staged, out)
          end

          size = File.size(out)
          puts "packed #{out} (#{fmt_bytes(size)}): #{name}, #{@device}, port #{@port}" \
               "#{prefix ? ", KV prefix" : ''}"
          puts "run it:      #{out} [--port N]"
          puts "describe it: #{out} --describe"
          EXIT_OK
        end

        private

        def write_artifact(path, runner, entries)
          index = +""
          File.open(path, "wb") do |o|
            IO.copy_stream(runner, o)
            entries.each do |ename, src|
              pad(o)
              off = o.pos
              len = IO.copy_stream(src, o)
              index << "#{ename} #{off} #{len}\n"
            end
            pad(o)
            ioff = o.pos
            o.write(index)
            o.write(MAGIC)
            o.write([ioff, index.bytesize].pack("Q<Q<"))
          end
        end

        def pad(io)
          rem = io.pos % ALIGN
          io.write("\0" * (ALIGN - rem)) if rem != 0
        end

        def describe_text(root, model, name, runner, target, prefix, prefix_tokens, config)
          lines = []
          lines << "toy pack: #{name}"
          lines << ""
          lines << card_text(model)
          lines << ""
          lines << "Provenance"
          lines << "  model file     #{File.basename(model)} (#{File.size(model)} bytes)"
          lines << "  model sha256   #{Digest::SHA256.file(model).hexdigest}"
          lines << "  toy            #{git_rev(root)}"
          lines << "  spinel         #{spinel_rev(root)}"
          lines << "  ggml           #{git_rev(File.join(root, 'vendor', 'ggml'), dirty: false)}"
          lines << "  runner         #{target} (#{File.size(runner)} bytes, sha256 #{Digest::SHA256.file(runner).hexdigest})"
          lines << "  built on       #{RUBY_PLATFORM}"
          lines << "  packed at      #{Time.now.utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"
          lines << ""
          lines << "Config"
          config.each_line { |l| lines << "  #{l.chomp}" }
          lines << ""
          if prefix
            lines << "KV prefix"
            lines << "  source         #{File.basename(prefix)} (#{File.size(prefix)} bytes, sha256 #{Digest::SHA256.file(prefix).hexdigest})"
            lines << "  tokens         #{prefix_tokens} (the system turn every request is expected to start with)"
          else
            lines << "KV prefix: none"
          end
          needs = shared_libs(runner)
          unless needs.empty?
            lines << ""
            lines << "Shared libraries needed at run time"
            lines << "  #{needs.join(' ')}"
          end
          lines.join("\n") + "\n"
        end

        # The `toy describe` card for the model; a one-line note where
        # describe declines the architecture.
        def card_text(model)
          old_out, old_err = $stdout, $stderr
          cap_out, cap_err = StringIO.new, StringIO.new
          $stdout, $stderr = cap_out, cap_err
          code = begin
            Describe.new([model]).run
          ensure
            $stdout, $stderr = old_out, old_err
          end
          # describe's "Source:" line is a path on the packing machine; the
          # Provenance block names the file instead.
          return cap_out.string.lines.reject { |l| l.start_with?("Source: ") }.join.rstrip if code == EXIT_OK
          "Card: not rendered (#{cap_err.string.strip.sub(/\Atoy describe: /, '')})"
        end

        def git_rev(dir, dirty: true)
          rev, st = Open3.capture2e("git", "-C", dir, "rev-parse", "HEAD")
          return "unknown" unless st.success?
          rev = rev.strip
          if dirty
            status, = Open3.capture2e("git", "-C", dir, "status", "--porcelain", "--untracked-files=no")
            rev += " (with uncommitted changes)" unless status.strip.empty?
          end
          rev
        end

        def spinel_rev(root)
          dir, st = Open3.capture2e("make", "-s", "-C", root, "print-spinel-dir")
          return "unknown" unless st.success?
          git_rev(dir.strip, dirty: false)
        end

        # DT_NEEDED names, when the platform has ldd; [] otherwise.
        def shared_libs(runner)
          out, st = Open3.capture2e("ldd", runner)
          return [] unless st.success?
          out.lines.map { |l| l[/^\s*(\S+)\s+=>/, 1] }.compact.sort
        rescue SystemCallError
          []
        end

        def fmt_bytes(n)
          return "#{(n / 1024.0**3).round(2)} GiB" if n >= 1024**3
          "#{(n / 1024.0**2).round(1)} MiB"
        end

        def parse_args
          rest = []
          i = 0
          take = lambda do |flag|
            i += 1
            @argv[i] || (return bad_arg("#{flag} requires a value"))
          end
          while i < @argv.length
            tok = @argv[i]
            case tok
            when "-o", "--out" then @out = take.call(tok)
            when "--prefix"    then @prefix = take.call(tok)
            when "--name"      then @name = take.call(tok)
            when "--device"
              @device = take.call(tok)
              return bad_arg("--device must be one of cpu, cuda, got #{@device.inspect}") unless %w[cpu cuda].include?(@device)
            when "--ctx"
              v = take.call(tok)
              return bad_arg("--ctx must be a positive integer, got #{v.inspect}") unless v =~ /\A\d+\z/ && v.to_i > 0
              @ctx = v.to_i
            when "--port"
              v = take.call(tok)
              return bad_arg("--port must be in 1..65535, got #{v.inspect}") unless v =~ /\A\d+\z/ && (1..65_535).cover?(v.to_i)
              @port = v.to_i
            when "--stop-ids"
              v = take.call(tok)
              return bad_arg("--stop-ids must be comma-separated token ids, got #{v.inspect}") unless v =~ /\A\d+(,\d+)*\z/
              @stop_ids = v
            when /\A-/ then return bad_arg("unknown flag #{tok.inspect}")
            else rest << tok
            end
            i += 1
          end
          return bad_arg("missing required argument <model.gguf>") if rest.empty?
          return bad_arg("unexpected extra arguments: #{rest[1..].join(' ')}") if rest.length > 1
          return bad_arg("missing -o <artifact>") if @out.nil?
          @model = rest.first
          true
        end

        def bad_arg(msg)
          $stderr.puts "toy pack: #{msg}"
          EXIT_BAD_INPUT
        end

        def fail_out(msg)
          $stderr.puts "toy pack: #{msg}"
          EXIT_FAILURE
        end
      end
    end
  end
end
