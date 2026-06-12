#!/usr/bin/env ruby
# prep/run_log_gate.rb — unit gate for Toy::RunLog (toy#64 item 6).
#
# CRuby-only (RunLog is a lib/toy/core/ CLI-side class — no Spinel).
# Self-contained: builds a deterministic synthetic runs/ fixture in a
# tmpdir and asserts the full surface (open / config / steps /
# loss_curve / final_loss / scan ordering / fail-loud arms). If the
# repo's own runs/ root has real train-gate bundles, additionally
# opens the freshest one as an integration sniff.
#
#   ruby prep/run_log_gate.rb    # exit 0 PASS / 1 FAIL

require "tmpdir"
require "json"
require "fileutils"
require_relative "../lib/toy/core/run_log"

$failures = []
def check(name)
  ok = yield
  if ok
    puts "  ok: #{name}"
  else
    puts "  FAIL: #{name}"
    $failures << name
  end
rescue => e
  puts "  FAIL: #{name} raised #{e.class}: #{e.message}"
  $failures << name
end

def write_run(root, id, losses, final: true, run_start_extra: {})
  dir = File.join(root, id)
  FileUtils.mkdir_p(dir)
  File.open(File.join(dir, "events.jsonl"), "w") do |f|
    f.puts JSON.generate({ "kind" => "run_start", "schema" => "toy/v1",
                           "run_id" => id, "phase" => "train",
                           "model" => { "arch" => "llama", "d_model" => 64 },
                           "config" => { "steps" => losses.length, "seed" => 0 } }
                         .merge(run_start_extra))
    losses.each_with_index do |loss, i|
      f.puts JSON.generate({ "kind" => "step", "phase" => "train",
                             "step" => i + 1, "loss" => loss })
    end
    if final
      f.puts JSON.generate({ "kind" => "run_end", "reason" => "completed",
                             "final_step" => losses.length,
                             "final_loss" => losses.last })
    end
  end
  dir
end

Dir.mktmpdir("toy_run_log_gate") do |root|
  good = write_run(root, "llama-20990101-001", [6.4, 6.3, 6.2])
  best = write_run(root, "llama-20990101-002", [6.4, 6.1, 5.9])
  cut  = write_run(root, "llama-20990101-003", [6.5, 6.45], final: false)
  write_run(root, "llama-20990101-004", [])                     # never stepped
  FileUtils.mkdir_p(File.join(root, "not-a-run"))               # no events.jsonl

  log = Toy::RunLog.open(good)
  check("config carries run_start fields") do
    log.config["run_id"] == "llama-20990101-001" &&
      log.config["model"]["d_model"] == 64 &&
      log.config["config"]["steps"] == 3
  end
  check("run_id reader")        { log.run_id == "llama-20990101-001" }
  check("steps are the step events in order") do
    log.steps.length == 3 && log.steps.map { |s| s["step"] } == [1, 2, 3]
  end
  check("loss_curve")           { log.loss_curve == [6.4, 6.3, 6.2] }
  check("final_loss from run_end") { log.final_loss == 6.2 }

  check("interrupted run falls back to last step loss") do
    Toy::RunLog.open(cut).final_loss == 6.45
  end
  check("never-stepped run has nil final_loss") do
    Toy::RunLog.open(File.join(root, "llama-20990101-004")).final_loss.nil?
  end

  scanned = Toy::RunLog.scan(root)
  check("scan finds the 4 bundles, skips non-run dirs") { scanned.length == 4 }
  check("scan sorts by final_loss ascending, lossless last") do
    scanned.map(&:run_id) ==
      ["llama-20990101-002", "llama-20990101-001",
       "llama-20990101-003", "llama-20990101-004"]
  end
  check("the 3-line 'find my best run' works") do
    Toy::RunLog.scan(root).first.final_loss == 5.9
  end

  check("open on a dir without events.jsonl fails loud") do
    begin
      Toy::RunLog.open(File.join(root, "not-a-run"))
      false
    rescue ArgumentError => e
      e.message.include?("no events.jsonl")
    end
  end
  check("malformed JSONL line fails loud with line number") do
    bad = File.join(root, "bad-run")
    FileUtils.mkdir_p(bad)
    File.write(File.join(bad, "events.jsonl"),
               %({"kind":"run_start","run_id":"bad-run"}\n{not json}\n))
    begin
      Toy::RunLog.open(bad)
      false
    rescue ArgumentError => e
      e.message.include?(":2:") && e.message.include?("malformed")
    end
  end
  check("missing run_start fails loud") do
    no_start = File.join(root, "no-start")
    FileUtils.mkdir_p(no_start)
    File.write(File.join(no_start, "events.jsonl"),
               %({"kind":"step","step":1,"loss":1.0}\n))
    begin
      Toy::RunLog.open(no_start)
      false
    rescue ArgumentError => e
      e.message.include?("no run_start")
    end
  end
end

# Toy::RunBundle round-trip (toy#73 item 1): run the compiled example 01
# (which writes its bundle through Toy::RunBundle) twice with the same
# RUN_ID and assert (a) RunLog parses the bundle, (b) the second run
# REPLACED the first (tnn_events_open_trunc — no doubled events). Needs
# the Spinel-built binary; SKIPs loudly when absent (CRuby-only envs).
example_bin = File.expand_path("../examples/example_01_train_tiny", __dir__)
if File.executable?(example_bin)
  require "open3"
  rt_id  = "run-log-gate-roundtrip"
  rt_dir = File.expand_path("../runs/#{rt_id}", __dir__)
  rt_env = { "RUN_ID" => rt_id, "STEPS" => "3" }
  root   = File.expand_path("..", __dir__)
  out1, st1 = Open3.capture2e(rt_env, example_bin, chdir: root)
  out2, st2 = Open3.capture2e(rt_env, example_bin, chdir: root)
  check("round-trip: example 01 (RunBundle) runs") do
    unless st1.success? && st2.success?
      puts out1 unless st1.success?
      puts out2 unless st2.success?
    end
    st1.success? && st2.success?
  end
  check("round-trip: RunLog parses the RunBundle bundle") do
    log = Toy::RunLog.open(rt_dir)
    log.run_id == rt_id &&
      log.config["schema"] == "toy/v1" &&
      log.config["model"]["arch"] == "llama" &&
      log.config["config"]["steps"] == 3 &&
      log.loss_curve.length == 3 &&
      log.final_loss.is_a?(Float)
  end
  check("round-trip: run_start stamps backend{kind} (toy#73 A.3)") do
    # example 01 is the CPU compute entry → Toy::Device.name == "cpu".
    # git{} is opt-in (bundle.git! — example 01 stays vendor-free) so
    # its absence here is the documented default.
    cfgev = Toy::RunLog.open(rt_dir).config
    cfgev["backend"].is_a?(Hash) && cfgev["backend"]["kind"] == "cpu" &&
      !cfgev.key?("git")
  end
  check("round-trip: re-run TRUNCATES (3 steps, not 6)") do
    Toy::RunLog.open(rt_dir).steps.length == 3
  end
  FileUtils.rm_rf(rt_dir)
else
  puts "  SKIP: examples/example_01_train_tiny not built — RunBundle " \
       "round-trip not exercised (run `make example_01` first)"
end

# Per-arch run_start round-trip (toy#73 A.3): example 07 writes its
# bundle through run_start_vit! — the model{} block is the image shape,
# config{} has no context key. SKIPs loudly when the binary is absent.
vit_bin = File.expand_path("../examples/example_07_vit_tiny", __dir__)
if File.executable?(vit_bin)
  require "open3"
  vt_id  = "run-log-gate-vit"
  vt_dir = File.expand_path("../runs/#{vt_id}", __dir__)
  vt_env = { "RUN_ID" => vt_id, "STEPS" => "2" }
  vt_root = File.expand_path("..", __dir__)
  vt_out, vt_st = Open3.capture2e(vt_env, vit_bin, chdir: vt_root)
  check("vit round-trip: example 07 (run_start_vit!) runs") do
    puts vt_out unless vt_st.success?
    vt_st.success?
  end
  check("vit round-trip: RunLog parses the ViT-shaped run_start") do
    vlog = Toy::RunLog.open(vt_dir)
    vlog.run_id == vt_id &&
      vlog.config["schema"] == "toy/v1" &&
      vlog.config["backend"]["kind"] == "cpu" &&
      vlog.config["model"]["arch"] == "vit" &&
      vlog.config["model"]["image_size"] == 224 &&
      vlog.config["model"]["patch_size"] == 16 &&
      vlog.config["model"]["num_classes"] == 10 &&
      !vlog.config["config"].key?("context") &&
      vlog.config["config"]["steps"] == 2 &&
      vlog.loss_curve.length == 2 &&
      vlog.final_loss.is_a?(Float)
  end
  FileUtils.rm_rf(vt_dir)
else
  puts "  SKIP: examples/example_07_vit_tiny not built — per-arch " \
       "run_start_vit! round-trip not exercised (run `make example_07` first)"
end

# Integration sniff against real train-gate bundles, when present.
repo_runs = File.expand_path("../runs", __dir__)
if Dir.exist?(repo_runs) &&
   Dir.children(repo_runs).any? { |d| File.file?(File.join(repo_runs, d, "events.jsonl")) }
  check("integration: scan(repo runs/) parses real bundles") do
    logs = Toy::RunLog.scan(repo_runs)
    !logs.empty? && logs.all? { |l| l.config["schema"] == "toy/v1" } &&
      logs.first.final_loss.is_a?(Float)
  end
else
  puts "  note: repo runs/ has no bundles — integration sniff skipped " \
       "(run `ruby prep/train_gate.rb` to generate)"
end

if $failures.empty?
  puts "GATE PASS [run-log]: Toy::RunLog parses run bundles (config/steps/loss_curve/final_loss/scan)"
  exit 0
else
  puts "GATE FAIL [run-log]: #{$failures.length} failure(s): #{$failures.join('; ')}"
  exit 1
end
