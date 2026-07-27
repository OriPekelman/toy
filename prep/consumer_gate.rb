#!/usr/bin/env ruby
# prep/consumer_gate.rb — toy#60 item 4: the COLD-START consumer gate.
#
# Proves the README / docs/framework.md quickstart is TESTED, not
# aspirational. Two legs, both in a throwaway tmpdir:
#
#   APP leg  — `toy new gatelab`:
#     * the scaffold seeds data/ts_seqs.{bin,txt} (so `toy train` works)
#     * algos/recipes/hello.rb compiles with $SPINEL_DIR's spinel
#     * ./hello runs with default ENV (3 "step N: loss=" lines, finite)
#     * D_MODEL=128 STEPS=10 ./hello runs WITHOUT recompiling (one
#       compile, many runs) and produces a different curve
#     * `toy train from-scratch --steps 2` in the project prints losses
#       AND writes runs/<id>/events.jsonl (run_start + step events)
#     * the fail-loud guard: with data/ts_seqs.txt removed, `toy train`
#       exits non-zero and NAMES the missing path (spinel-dev#17 class)
#
#   LIB leg  — `toy new gatelib --lib`:
#     * Gemfile pointed at this checkout (path:), `bundle lock`
#     * `spinel-compat vendor` builds ggml + tinynn INSIDE vendor/
#     * `./build.sh cpu` compiles, `./experiment_cpu` trains to
#       "experiment: ok (device=cpu)"
#     SKIPS LOUDLY (exit 0, like the model-gated gates) when bundler or
#     spinel-compat is not available — the app leg always runs.
#
# Structural gate (loss finite + decreasing-ish, files exist), NOT
# byte-exact: a consumer cold-start exercises scaffold + build + run
# wiring, not numerics (those are train_gate's job).
#
#   make gate-consumer            # or: ruby prep/consumer_gate.rb
#
# Env: SPINEL_DIR (default ~/sites/spinel), SPINELGEMS_DIR (default
# ~/sites/spinelgems), KEEP=1 to keep the tmpdir for inspection.

require "open3"
require "tmpdir"
require "fileutils"
require "json"

ROOT       = File.expand_path("..", __dir__)
# Toolchain via make (pin-guard-validated; toy#119 class — the old
# ~/sites/spinel fallback silently compiled with a stale toolchain).
require "open3"
def resolve_spinel_dir
  env = ENV["SPINEL_DIR"].to_s
  return env unless env.empty?
  out, st = Open3.capture2("make", "--no-print-directory", "-s", "print-spinel-dir",
                           chdir: File.expand_path("..", __dir__))
  abort "consumer_gate: SPINEL_DIR resolution via make failed (pin guard?):\n#{out}" unless st.success?
  out.strip
end
SPINEL_DIR = resolve_spinel_dir
SPINEL_BIN = File.join(SPINEL_DIR, "spinel")

abort "GATE FAIL [consumer]: no spinel at #{SPINEL_BIN} (set SPINEL_DIR=)" unless File.executable?(SPINEL_BIN)

def run!(desc, cmd, env: {}, chdir: Dir.pwd, expect_fail: false)
  out, status = Open3.capture2e(env, *cmd, chdir: chdir)
  if expect_fail
    if status.success?
      puts out
      abort "GATE FAIL [consumer]: #{desc} — expected non-zero exit, got success"
    end
  elsif !status.success?
    puts out
    abort "GATE FAIL [consumer]: #{desc} — exit #{status.exitstatus.inspect}"
  end
  out
end

def loss_lines(out)
  out.lines.select { |l| l.start_with?("step ") && l.include?("loss=") }
end

def losses(out)
  loss_lines(out).map { |l| l[/loss=([-\d.eE+]+)/, 1].to_f }
end

# Tool discovery for the LIB leg (loud skip when absent).
def find_bundle
  return "bundle" if system("command -v bundle >/dev/null 2>&1")
  rv = File.join(Dir.home, ".local", "share", "rv", "rubies", "ruby-3.4.9", "bin", "bundle")
  return rv if File.executable?(rv)
  nil
end

def find_spinel_compat
  env = ENV["SPINELGEMS_DIR"]
  cands = []
  cands << File.join(env, "exe", "spinel-compat") if env && !env.empty?
  cands << File.join(Dir.home, "sites", "spinelgems", "exe", "spinel-compat")
  cands.find { |c| File.executable?(c) }
end

# Precondition: the checkout's CPU archive (hello links -Ltinynn from ROOT).
run!("build tinynn archive", ["make", "-C", ROOT, "tinynn/libtinynn_ggml.a"])

tmp = Dir.mktmpdir("toy60-consumer-")
puts "consumer gate: tmpdir #{tmp}#{ENV['KEEP'] == '1' ? ' (kept)' : ''}"
begin
  base_env = { "TOY_HOME" => ROOT, "SPINEL_DIR" => SPINEL_DIR }

  # ── APP leg ─────────────────────────────────────────────────────────
  lab = File.join(tmp, "gatelab")
  out = run!("toy new gatelab", ["ruby", File.join(ROOT, "bin", "toy"), "new", lab], env: base_env)
  %w[algos/recipes/hello.rb data/ts_seqs.bin data/ts_seqs.txt toy.yml bin/toy].each do |rel|
    abort "GATE FAIL [consumer]: scaffold missing #{rel}\n#{out}" unless File.exist?(File.join(lab, rel))
  end
  link = File.join(lab, "algos", "recipes", "toy_lib")
  abort "GATE FAIL [consumer]: toy_lib symlink missing/dangling" unless File.exist?(link)
  puts "  ok: scaffold tree + seeded corpus + toy_lib link"

  hello = File.join(lab, "hello")
  out = run!("compile hello.rb",
             [SPINEL_BIN, File.join(lab, "algos", "recipes", "hello.rb"), "-o", hello],
             chdir: ROOT)   # relative -Ltinynn/-Lvendor link flags resolve in ROOT
  abort "GATE FAIL [consumer]: hello binary not produced\n#{out}" unless File.executable?(hello)
  compile_mtime = File.mtime(hello)
  puts "  ok: hello.rb compiled (#{SPINEL_BIN})"

  out1 = run!("./hello (defaults)", [hello], chdir: lab)
  l1 = losses(out1)
  abort "GATE FAIL [consumer]: expected 3 default-step loss lines, got #{l1.length}\n#{out1}" unless l1.length == 3
  abort "GATE FAIL [consumer]: non-finite loss in default run\n#{out1}" unless l1.all?(&:finite?)
  puts "  ok: hello defaults — #{loss_lines(out1).first.strip} … #{loss_lines(out1).last.strip}"

  out2 = run!("./hello (D_MODEL=128 STEPS=10, no recompile)", [hello],
              env: { "D_MODEL" => "128", "STEPS" => "10" }, chdir: lab)
  l2 = losses(out2)
  abort "GATE FAIL [consumer]: expected 10 override-step lines, got #{l2.length}\n#{out2}" unless l2.length == 10
  abort "GATE FAIL [consumer]: non-finite loss in override run\n#{out2}" unless l2.all?(&:finite?)
  abort "GATE FAIL [consumer]: hello was recompiled between runs" unless File.mtime(hello) == compile_mtime
  abort "GATE FAIL [consumer]: D_MODEL override had no effect (identical curves)" if l1 == l2.first(3)
  puts "  ok: ENV override (one compile, many runs) — final loss #{l2.last}"

  out = run!("toy train from-scratch in the scaffold",
             ["ruby", File.join(lab, "bin", "toy"), "train", "from-scratch", "--steps", "2"],
             env: base_env, chdir: lab)
  lt = losses(out)
  abort "GATE FAIL [consumer]: toy train printed #{lt.length} loss lines (want 2)\n#{out}" unless lt.length == 2
  bundles = Dir[File.join(lab, "runs", "*", "events.jsonl")]
  abort "GATE FAIL [consumer]: toy train wrote no runs/<id>/events.jsonl\n#{out}" if bundles.empty?
  first_ev = JSON.parse(File.readlines(bundles.first).first)
  abort "GATE FAIL [consumer]: events.jsonl first event is not run_start" unless first_ev["kind"] == "run_start"
  puts "  ok: toy train — losses print + #{bundles.first.sub(lab + '/', '')} written"

  # Fail-loud leg: a missing corpus must NAME the path, never SEGV/mask.
  FileUtils.mv(File.join(lab, "data", "ts_seqs.txt"), File.join(lab, "data", "ts_seqs.txt.off"))
  out = run!("toy train with corpus removed (must fail loud)",
             ["ruby", File.join(lab, "bin", "toy"), "train", "from-scratch", "--steps", "1"],
             env: base_env, chdir: lab, expect_fail: true)
  unless out.include?("corpus not found: data/ts_seqs.txt")
    puts out
    abort "GATE FAIL [consumer]: missing-corpus error does not name the path"
  end
  FileUtils.mv(File.join(lab, "data", "ts_seqs.txt.off"), File.join(lab, "data", "ts_seqs.txt"))
  puts "  ok: fail-loud — missing corpus names the path (no silent \"\" read)"

  # ── LIB leg ─────────────────────────────────────────────────────────
  bundle = find_bundle
  compat = find_spinel_compat
  if bundle.nil? || compat.nil?
    puts "GATE SKIP [consumer/lib]: #{bundle ? '' : 'bundler '}#{compat ? '' : 'spinel-compat '}not available — app leg passed; lib leg skipped loudly"
  else
    libp = File.join(tmp, "gatelib")
    run!("toy new gatelib --lib", ["ruby", File.join(ROOT, "bin", "toy"), "new", libp, "--lib"], env: base_env)
    # Point the Gemfile at THIS checkout (the scaffold's rubygems form
    # works once toy is published; the gate must be hermetic).
    File.write(File.join(libp, "Gemfile"), <<~GEMFILE)
      source "https://rubygems.org"
      ruby "3.2.3", engine: "spinel", engine_version: "0.0.0"
      gem "toy", path: #{ROOT.inspect}
    GEMFILE
    run!("bundle lock", [bundle, "lock"], chdir: libp)
    run!("spinel-compat vendor", [compat, "vendor"], env: { "SPINEL_DIR" => SPINEL_DIR }, chdir: libp)
    %w[vendor/spinel/toy/tinynn/libtinynn_ggml.a vendor/spinel/toy/vendor/ggml/build/src/libggml.a].each do |rel|
      abort "GATE FAIL [consumer]: vendor step did not build #{rel}" unless File.file?(File.join(libp, rel))
    end
    puts "  ok: vendored (lib/ + ggml + tinynn archives inside the project)"

    # build.sh calls bare `spinel` — give it a PATH shim to $SPINEL_BIN.
    shim = File.join(tmp, "binshim"); FileUtils.mkdir_p(shim)
    FileUtils.ln_sf(SPINEL_BIN, File.join(shim, "spinel"))
    run!("./build.sh cpu", ["./build.sh", "cpu"],
         env: { "PATH" => "#{shim}:#{ENV['PATH']}" }, chdir: libp)
    exp = File.join(libp, "experiment_cpu")
    abort "GATE FAIL [consumer]: build.sh produced no experiment_cpu" unless File.executable?(exp)
    out = run!("./experiment_cpu", [exp], env: { "STEPS" => "5" }, chdir: libp)
    le = losses(out)
    abort "GATE FAIL [consumer]: experiment_cpu printed #{le.length} loss lines (want 5)\n#{out}" unless le.length == 5
    abort "GATE FAIL [consumer]: experiment_cpu missing final ok line\n#{out}" unless out.include?("experiment: ok (device=cpu)")
    puts "  ok: --lib leg — vendored build + experiment_cpu trains (final loss #{le.last})"
  end

  puts "GATE PASS [consumer]: cold-start quickstart works (scaffold + hello compile/run + ENV re-run + toy train runs/ bundle#{bundle && compat ? ' + --lib vendor/build/run' : ''})"
ensure
  FileUtils.remove_entry(tmp) unless ENV["KEEP"] == "1"
end
