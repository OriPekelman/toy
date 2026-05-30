#!/usr/bin/env ruby
# prep/serve_gate.rb — deterministic FUNCTIONAL HTTP gate for `toy serve`.
#
# Proves the moved lib/toy/serve/openai/* Server (compiled to the persistent
# runner libexec/toy-serve, source lib/toy/run/serve.rb) reproduces a
# RECORDED BASELINE byte-for-byte over a real HTTP request, AND guarantees
# the server process is torn down — even on failure / Ctrl-C / exception.
#
#   ruby prep/serve_gate.rb     # exit 0 byte-for-byte match, 1 fail, 2 setup
#
# CRuby ONLY (stdlib net/http + socket + timeout + json). NO Spinel — the
# gate is the HTTP CLIENT; the server is the Spinel binary. The gate does
# NOT go through `toy serve` (whose Kernel.exec foreground launch never
# returns); it spawns the runner directly so it can poll + kill it.
#
# DETERMINISM: greedy argmax decode (no rand/seed/temperature). The volatile
# completion `id`/`created` fields are time-based and would differ every run,
# so "byte-for-byte" applies to the parsed choices[0].ids array (exactly as
# infer_gate compares the parsed line, not raw stdout). Do NOT loosen.

require "net/http"
require "json"
require "socket"
require "timeout"
require "tmpdir"

ROOT     = File.expand_path("..", __dir__)
RUNNER   = File.join(ROOT, "libexec", "toy-serve")
BASELINE = File.join(ROOT, "prep", "fixtures", "serve_baseline.txt")
MODEL    = File.join(ROOT, "data", "smollm2-135m-native.gguf")
MODEL_BASENAME = "smollm2-135m-native.gguf"
MODEL_NAME = "test"
# Locked gate request — MUST match the baseline recording exactly.
PROMPT_IDS = [1, 100, 200]
MAX_TOKENS = 8
READY_DEADLINE = 60.0   # generous: model realize at boot takes seconds
POLL_INTERVAL  = 0.25

# ----- Preflight (exit 2 on any missing precondition) -----------------------

unless File.file?(MODEL)
  warn "serve_gate: missing model: #{MODEL}"
  exit 2
end
unless File.file?(BASELINE)
  warn "serve_gate: missing recorded baseline: #{BASELINE}"
  exit 2
end

# Build the runner directly (the gate spawns it, not via `toy serve`).
unless File.file?(RUNNER) && File.executable?(RUNNER)
  $stderr.puts "serve_gate: building #{RUNNER}..."
end
build_out, build_st = nil, nil
require "open3"
build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-serve")
unless build_st.success?
  warn "serve_gate: `make libexec/toy-serve` failed:\n#{build_out.lines.last(20).join}"
  exit 2
end
unless File.file?(RUNNER) && File.executable?(RUNNER)
  warn "serve_gate: runner missing after build: #{RUNNER}"
  exit 2
end

# Load the recorded baseline: gguf-basename → expected JSON id array.
baseline_ids = nil
File.foreach(BASELINE) do |line|
  next if line.strip.empty? || line.start_with?("#")
  key, want = line.chomp.split("\t", 2)
  next unless key && want
  if key == MODEL_BASENAME
    baseline_ids = JSON.parse(want)
  end
end
if baseline_ids.nil?
  warn "serve_gate: no baseline record for #{MODEL_BASENAME} in #{BASELINE}"
  exit 2
end

# ----- Server lifecycle with GUARANTEED TEARDOWN ---------------------------

@pid = nil
@torn = false
@logf = nil

def teardown
  return if @torn
  @torn = true
  pid = @pid
  return if pid.nil?
  begin
    pgid = Process.getpgid(pid)
    Process.kill("TERM", -pgid)
  rescue Errno::ESRCH, Errno::ECHILD
    # already gone
  end
  begin
    Timeout.timeout(5) { Process.wait(pid) }
  rescue Timeout::Error
    begin
      pgid = Process.getpgid(pid)
      Process.kill("KILL", -pgid)
      Process.wait(pid)
    rescue Errno::ESRCH, Errno::ECHILD
    end
  rescue Errno::ECHILD
    # already reaped
  end
ensure
  if @logf && File.file?(@logf)
    File.delete(@logf) rescue nil
  end
end

# Install teardown three ways so a bound-port server is NEVER leaked:
#   (a) ensure block around the lifecycle (normal/assert/exception)
#   (b) at_exit (covers exits not caught by ensure)
#   (c) INT/TERM traps (Ctrl-C on the GATE itself still kills the server;
#       at_exit alone does NOT fire on a default-SIGINT terminate)
at_exit { teardown }
["INT", "TERM"].each do |sig|
  trap(sig) { teardown; exit 130 }
end

exit_code = 1
begin
  # 1. Grab a free port (close immediately; the server rebinds it).
  sock = TCPServer.new("127.0.0.1", 0)
  port = sock.addr[1]
  sock.close

  # 2. Spawn the runner with the controlled env hash; pgroup so a
  #    group-kill reaches any child tree; stdout/stderr to a tmpfile so
  #    boot chatter doesn't pollute the gate.
  @logf = File.join(Dir.tmpdir, "serve_gate_srv_#{Process.pid}.log")
  @pid = Process.spawn(
    { "MODEL_PATH" => MODEL, "MODEL_NAME" => MODEL_NAME, "PORT" => port.to_s },
    RUNNER,
    chdir: ROOT, pgroup: true,
    out: @logf, err: @logf
  )

  # 3. Readiness poll: GET /v1/models until 200 or deadline. Each iter
  #    also checks for early death (bad model / EADDRINUSE) → fail loud.
  ready = false
  deadline = Time.now + READY_DEADLINE
  while Time.now < deadline
    dead = Process.waitpid(@pid, Process::WNOHANG)
    if dead
      st = $?
      tail = File.file?(@logf) ? File.read(@logf).lines.last(20).join : "(no log)"
      warn "serve_gate: server exited early (status #{st.exitstatus}) before ready:\n#{tail}"
      @pid = nil  # already reaped
      exit_code = 1
      raise "server died early"
    end
    begin
      uri = URI("http://127.0.0.1:#{port}/v1/models")
      resp = Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 5) do |h|
        h.get("/v1/models")
      end
      if resp.code == "200"
        ready = true
        break
      end
    rescue Errno::ECONNREFUSED, Errno::ECONNRESET, EOFError,
           Timeout::Error, SocketError, Net::OpenTimeout, Net::ReadTimeout
      # not up yet
    end
    sleep POLL_INTERVAL
  end

  unless ready
    tail = File.file?(@logf) ? File.read(@logf).lines.last(20).join : "(no log)"
    warn "serve_gate: server never became ready within #{READY_DEADLINE}s\n#{tail}"
    exit_code = 1
    raise "not ready"
  end

  # 4. POST /v1/completions with the locked request body.
  body = JSON.dump(
    "model" => MODEL_NAME,
    "prompt" => PROMPT_IDS,
    "max_tokens" => MAX_TOKENS
  )
  resp = Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 30) do |h|
    req = Net::HTTP::Post.new("/v1/completions", "Content-Type" => "application/json")
    req.body = body
    h.request(req)
  end

  # 5. Guard: never mask. A 4xx/5xx body lacks choices and would
  #    NoMethodError — fail loud with the body tail instead.
  unless resp.code == "200"
    warn "serve_gate: /v1/completions returned #{resp.code}, body:\n#{resp.body.to_s[0, 500]}"
    exit_code = 1
    raise "non-200"
  end
  parsed = JSON.parse(resp.body)
  choices = parsed["choices"]
  unless choices.is_a?(Array) && choices[0].is_a?(Hash) && choices[0]["ids"].is_a?(Array)
    warn "serve_gate: response missing choices[0].ids, body:\n#{resp.body.to_s[0, 500]}"
    exit_code = 1
    raise "missing ids"
  end
  got_ids = choices[0]["ids"]

  # 6. Byte-for-byte compare the parsed id arrays (greedy ⇒ deterministic).
  puts "model   : #{MODEL_BASENAME}"
  puts "prompt  : #{PROMPT_IDS.inspect}  max_tokens=#{MAX_TOKENS}"
  puts "expected: #{baseline_ids.inspect}"
  puts "actual  : #{got_ids.inspect}"
  if got_ids == baseline_ids
    puts "GATE PASS: toy serve reproduces recorded /v1/completions baseline byte-for-byte"
    exit_code = 0
  else
    warn "GATE FAIL: toy serve diverged from recorded baseline"
    exit_code = 1
  end
rescue => e
  # Distinguish the deliberate early-exit raises (already warned) from
  # an unexpected exception.
  unless ["server died early", "not ready", "non-200", "missing ids"].include?(e.message)
    warn "serve_gate: unexpected error: #{e.class}: #{e.message}"
  end
  exit_code = 1 if exit_code == 0
ensure
  teardown
end

exit exit_code
