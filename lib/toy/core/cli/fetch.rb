# lib/toy/core/cli/fetch.rb — `toy fetch <hf-repo> [<file.gguf>]`.
#
# Download a GGUF from HuggingFace into the standard HF cache and drop a
# RELATIVE symlink at data/<basename>.gguf so `toy list` / `toy describe`
# and the example GGUF= paths resolve. CRuby ONLY (the CLI shell) — NO
# Spinel, NO mirrors. Folds in prep/fetch_model.sh.
#
# Download-tool probe order (drift note: the old bash script + README say
# `huggingface-cli`, which is DEPRECATED and absent on modern installs;
# the modern tool is `hf`):
#   1. hf download <repo> <file>            (modern; reads its `<abs>` stdout)
#   2. huggingface-cli download <repo> <file> (legacy; then snapshot-glob)
#   3. curl -fL https://huggingface.co/<repo>/resolve/main/<file>  (fallback)
# All three populate the same HF cache layout.
#
# Behavior delta from fetch_model.sh:85 — that script wrote an ABSOLUTE
# symlink; per the P3 brief this writes a RELATIVE one (data/ → cache).

require "json"
require "open3"
require "fileutils"
require_relative "exit_codes"
require_relative "../model_scan"

module Toy
  module Core
    module CLI
      class Fetch
        FORMAT = "toy/fetch-v1"

        def initialize(argv)
          @argv = argv
          @json = false
          @repo = nil
          @file = nil
        end

        def run
          parsed = parse_args
          return parsed unless parsed == true

          unless @file
            # No file: print the common-pick hints (folded from
            # fetch_model.sh:34-40) and return bad-input. The bash script's
            # `exit 1` predates the exit-code contract; 2 (missing required
            # arg) is the consistent choice here.
            return missing_file
          end

          cache_path = download
          return cache_path if cache_path.is_a?(Integer) # error code

          link = link_into_data(cache_path)
          return link if link.is_a?(Integer)

          if @json
            puts JSON.pretty_generate(
              "format" => FORMAT,
              "repo" => @repo,
              "file" => @file,
              "cache_path" => cache_path,
              "link" => link
            )
          else
            puts "linked #{link} -> #{cache_path}"
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
              $stderr.puts "toy fetch: unknown flag #{tok.inspect}"
              return EXIT_BAD_INPUT
            else rest << tok
            end
          end
          if rest.empty?
            $stderr.puts "toy fetch: missing required argument <hf-repo>"
            return EXIT_BAD_INPUT
          end
          @repo = rest[0]
          @file = rest[1]
          true
        end

        def hf_root
          File.join(ENV["HOME"] || "/", ".cache/huggingface/hub")
        end

        # Probe + run a downloader. Returns the absolute cache path of the
        # downloaded file on success, or an Integer exit code (with a clean
        # message already emitted) on failure.
        def download
          if tool?("hf")
            return run_hf("hf")
          elsif tool?("huggingface-cli")
            return run_hf("huggingface-cli")
          elsif tool?("curl")
            return run_curl
          else
            return fail_out(
              "no download tool found (need `hf`, `huggingface-cli`, or `curl` on PATH)"
            )
          end
        end

        def tool?(name)
          ENV["PATH"].to_s.split(File::PATH_SEPARATOR).any? do |dir|
            p = File.join(dir, name)
            File.file?(p) && File.executable?(p)
          end
        end

        # `hf download <repo> <file>` and the legacy `huggingface-cli
        # download <repo> <file>` both print the resolved local path on the
        # last stdout line. `hf` prints a bare absolute path; we also probe
        # the HF cache snapshot layout as a fallback (legacy CLI may print
        # progress noise).
        def run_hf(bin)
          out, status = Open3.capture2e(bin, "download", @repo, @file)
          unless status.success?
            return fail_out(classify_dl_error(out))
          end
          # Last non-empty stdout line is the resolved path for `hf`.
          line = out.lines.map(&:strip).reject(&:empty?).last
          if line && File.file?(line)
            return File.expand_path(line)
          end
          # Fallback: glob the snapshot layout (legacy CLI path).
          snap = resolve_snapshot
          return snap if snap
          fail_out("download reported success but the cached file could not be located")
        end

        def run_curl
          repo_dir = File.join(hf_root, "models--#{@repo.gsub('/', '--')}")
          target_dir = File.join(repo_dir, "snapshots", "manual")
          FileUtils.mkdir_p(target_dir)
          target = File.join(target_dir, File.basename(@file))
          # Subdir files (e.g. tinyllamas/foo.gguf) still resolve via /main/.
          url = "https://huggingface.co/#{@repo}/resolve/main/#{@file}"
          out, status = Open3.capture2e(
            "curl", "-fL", "--retry", "2", "-o", target, url
          )
          unless status.success?
            File.delete(target) if File.file?(target) && File.size(target).zero?
            return fail_out(classify_dl_error(out, url: url))
          end
          File.expand_path(target)
        end

        # Find the snapshot copy of @file in the HF cache (legacy CLI +
        # curl-fallback layouts). Returns abs path or nil.
        def resolve_snapshot
          repo_dir = File.join(hf_root, "models--#{@repo.gsub('/', '--')}")
          base = File.basename(@file)
          candidates = Dir.glob(File.join(repo_dir, "snapshots", "*", "**", base))
          hit = candidates.find { |c| File.file?(c) }
          hit ? File.expand_path(hit) : nil
        end

        # Turn an external tool's stderr/stdout into a CLEAN one-line
        # message (bad repo / 404 / auth / network), never a raw dump.
        def classify_dl_error(output, url: nil)
          o = output.to_s
          if o =~ /401|403|gated|authenticat|unauthor/i
            "access denied for #{@repo} (private/gated repo, or auth required)"
          elsif o =~ /404|not found|does not exist|RepositoryNotFound|EntryNotFound/i
            "not found: #{@repo}/#{@file} (check the repo id and filename)"
          elsif o =~ /could not resolve host|connection|timed out|network|getaddrinfo|temporary failure/i
            "network error fetching #{url || @repo} (no connection / DNS)"
          else
            first = o.lines.map(&:strip).reject(&:empty?).first
            "download failed for #{@repo}/#{@file}" + (first ? ": #{first}" : "")
          end
        end

        # Drop a RELATIVE symlink at data/<basename> pointing at the
        # resolved cache path. Idempotent. Returns the link path string, or
        # an Integer exit code on failure.
        def link_into_data(cache_path)
          FileUtils.mkdir_p("data")
          link = File.join("data", File.basename(@file))
          # Relative target FROM the data/ dir TO the cache file, so the
          # symlink is portable relative to data/.
          rel = relative_path(File.expand_path(cache_path), File.expand_path("data"))

          if File.symlink?(link)
            existing = File.readlink(link)
            return link if existing == rel || File.expand_path(existing, "data") == File.expand_path(cache_path)
            File.delete(link)
          elsif File.exist?(link)
            return fail_out("data/#{File.basename(@file)} exists and is not a symlink")
          end
          File.symlink(rel, link)
          link
        rescue SystemCallError => e
          fail_out("could not link into data/: #{e.message}")
        end

        # Compute a relative path from `from_dir` to `target` (both abs).
        def relative_path(target, from_dir)
          require "pathname"
          Pathname.new(target).relative_path_from(Pathname.new(from_dir)).to_s
        end

        def missing_file
          msg = "missing required argument <file.gguf>"
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy fetch: #{msg}"
            $stderr.puts "toy fetch: pass a GGUF filename in the repo. Common picks (instruction-tuned, q8_0):"
            $stderr.puts "    bartowski/SmolLM2-135M-Instruct-GGUF  SmolLM2-135M-Instruct-Q8_0.gguf"
            $stderr.puts "    bartowski/Llama-3.2-1B-Instruct-GGUF  Llama-3.2-1B-Instruct-Q8_0.gguf"
            $stderr.puts "    Qwen/Qwen2.5-1.5B-Instruct-GGUF       qwen2.5-1.5b-instruct-q8_0.gguf"
          end
          EXIT_BAD_INPUT
        end

        def fail_out(msg)
          if @json
            puts JSON.pretty_generate("format" => FORMAT, "error" => msg)
          else
            $stderr.puts "toy fetch: #{msg}"
          end
          EXIT_FAILURE
        end
      end
    end
  end
end
