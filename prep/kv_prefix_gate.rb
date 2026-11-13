#!/usr/bin/env ruby
# prep/kv_prefix_gate.rb — toy#210 gate: KV prefix reuse and the KV prefix file.
#
# Boots the serve runner (CPU by default, `--device cuda` for
# libexec/toy-serve-cuda) on data/smollm2-135m-tok.gguf and checks, against
# a reference server started with KV_REUSE=0 (every prompt evaluated from
# position 0, the behaviour before toy#210):
#
#   reuse    a sequence of chat requests sharing a system message returns
#            the reference's replies, and the request events show the
#            shared tokens were NOT re-evaluated (cached_tokens); the
#            reference shows 0 throughout
#   exact    the same request twice, and a prompt that is a strict prefix
#            of the cached one, still return the reference's ids
#   save     --prefix-file + --prefix-cache with no file yet: the file is
#            written, and the first request already has the whole system
#            turn cached
#   restore  a second boot given ONLY the file (so nothing can be rebuilt)
#            serves the first request with the same cached count and the
#            same reply
#   stale    the file is refused with a different model, with a damaged
#            fingerprint and when truncated (boot exits non-zero); with a
#            different --prefix-file it is rebuilt, not used
#
#   ruby prep/kv_prefix_gate.rb [--device cuda]   # exit 0 pass, 1 fail, 2 unusable

require "net/http"
require "json"
require "socket"
require "timeout"
require "tmpdir"
require "open3"
require "fileutils"

ROOT    = File.expand_path("..", __dir__)
DEVICE  = (i = ARGV.index("--device")) ? ARGV[i + 1].to_s : "cpu"
TARGET  = DEVICE == "cuda" ? "libexec/toy-serve-cuda" : "libexec/toy-serve"
RUNNER  = ENV["TOY_GATE_RUNNER"] || File.join(ROOT, TARGET)
TOKENIZE = File.join(ROOT, "prep", "smokes", "smoke_tokenize")
GGUF     = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
LABEL    = DEVICE == "cuda" ? "kv-prefix-cuda" : "kv-prefix"

unless %w[cpu cuda].include?(DEVICE)
  warn "kv_prefix_gate: --device must be cpu or cuda"
  exit 2
end
unless File.file?(GGUF)
  warn "kv_prefix_gate: missing fixture #{GGUF} (never skipped silently)"
  exit 2
end
targets = ["prep/smokes/smoke_tokenize"]
targets << TARGET unless ENV["TOY_GATE_RUNNER"]
targets.each do |t|
  out, st = Open3.capture2e("make", "-C", ROOT, t)
  next if st.success?
  warn "kv_prefix_gate: `make #{t}` failed:\n#{out.lines.last(15).join}"
  exit 2
end

TMP = Dir.mktmpdir("toy210-")
@pid = nil
@run_dir = nil
@boots = 0

def stop_server
  pid = @pid
  @pid = nil
  return if pid.nil?
  begin
    Process.kill("TERM", -Process.getpgid(pid))
  rescue Errno::ESRCH, Errno::ECHILD
  end
  begin
    Timeout.timeout(5) { Process.wait(pid) }
  rescue Timeout::Error
    begin
      Process.kill("KILL", -Process.getpgid(pid))
      Process.wait(pid)
    rescue Errno::ESRCH, Errno::ECHILD
    end
  rescue Errno::ECHILD
  end
end

at_exit { stop_server; FileUtils.rm_rf(TMP) }
%w[INT TERM].each { |s| trap(s) { stop_server; exit 130 } }

# Boots the runner. Returns [port, nil] when ready, or [nil, exit status]
# when the process exits first (the stale legs expect that).
def start_server(gguf, extra_env = {})
  stop_server
  @boots += 1
  sock = TCPServer.new("127.0.0.1", 0)
  port = sock.addr[1]
  sock.close
  @run_dir = File.join(TMP, "run#{@boots}")
  FileUtils.mkdir_p(@run_dir)
  log = File.join(TMP, "srv#{@boots}.log")
  env = { "MODEL_PATH" => gguf, "MODEL_NAME" => "gate", "PORT" => port.to_s,
          "TOY_RUN_DIR" => @run_dir, "TOY_RUN_ID" => "gate" }.merge(extra_env)
  @pid = Process.spawn(env, RUNNER, chdir: ROOT, pgroup: true, out: log, err: log)
  deadline = Time.now + 180
  while Time.now < deadline
    if Process.waitpid(@pid, Process::WNOHANG)
      @pid = nil
      return [nil, $?.exitstatus]
    end
    begin
      r = Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 5) { |h| h.get("/health") }
      return [port, nil] if r.code == "200"
    rescue Errno::ECONNREFUSED, Errno::ECONNRESET, EOFError, Timeout::Error, SocketError,
           Net::OpenTimeout, Net::ReadTimeout
    end
    sleep 0.25
  end
  raise "server never became ready"
end

def post(port, path, payload)
  r = Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 180) do |h|
    req = Net::HTTP::Post.new(path, "Content-Type" => "application/json")
    req.body = JSON.dump(payload)
    h.request(req)
  end
  raise "#{path} returned #{r.code}: #{r.body.to_s[0, 160]}" unless r.code == "200"
  JSON.parse(r.body)
end

def chat(port, system, user)
  j = post(port, "/v1/chat/completions", "model" => "gate", "max_tokens" => 10,
           "messages" => [{ "role" => "system", "content" => system }, { "role" => "user", "content" => user }])
  [j.dig("choices", 0, "message", "content"), j.dig("usage", "prompt_tokens")]
end

def complete(port, ids)
  post(port, "/v1/completions", "model" => "gate", "prompt" => ids, "max_tokens" => 8).dig("choices", 0, "ids")
end

# cached_tokens of every request event of the CURRENT server, in order.
def cached
  File.readlines(File.join(@run_dir, "events.jsonl")).map { |l| JSON.parse(l) }
      .select { |e| e["name"] == "request" }.map { |e| e.dig("extra", "cached_tokens") }
end

def encode(text)
  out, st = Open3.capture2e({ "GGUF" => GGUF, "TEXT" => text }, TOKENIZE)
  raise "smoke_tokenize failed: #{out.lines.last}" unless st.success?
  out[/^ids: (.*)$/, 1].to_s.split.map(&:to_i)
end

SYSTEM = "You are a careful assistant. " + (1..12).map { |i| "Rule #{i}: answer briefly and in plain words." }.join(" ")
SYS_N  = encode("<|im_start|>system\n#{SYSTEM}<|im_end|>\n").length
Q1, Q2 = "What is 2+2?", "Name a colour."
IDS    = [2020, 719, 253, 655, 28, 665, 436, 253, 555, 28].freeze

failures = []
ok = ->(leg, what) { puts "  ok: #{leg} — #{what}" }

begin
  # ---- reference: no reuse ----
  port, = start_server(GGUF, "KV_REUSE" => "0")
  ref = [chat(port, SYSTEM, Q1), chat(port, SYSTEM, Q2), chat(port, SYSTEM, Q1)]
  ref_ids = [complete(port, IDS), complete(port, IDS), complete(port, IDS[0, 6])]
  ref_cached = cached
  if ref_cached.any? { |c| c != 0 }
    failures << "reuse: the KV_REUSE=0 reference reused rows (#{ref_cached.inspect})"
  end

  # ---- reuse / exact ----
  port, = start_server(GGUF)
  got = [chat(port, SYSTEM, Q1), chat(port, SYSTEM, Q2), chat(port, SYSTEM, Q1)]
  c = cached
  if got != ref
    failures << "reuse: replies differ from the no-reuse reference\n      ref: #{ref.inspect}\n      got: #{got.inspect}"
  elsif c[0] != 0 || c[1] < SYS_N || c[2] < SYS_N || c[1] >= got[1][1]
    failures << "reuse: cached_tokens #{c.inspect}; expected 0, then at least the #{SYS_N}-token system turn"
  else
    ok.call("reuse", "same replies; cached_tokens #{c.inspect} (system turn is #{SYS_N} tokens)")
  end
  got_ids = [complete(port, IDS), complete(port, IDS), complete(port, IDS[0, 6])]
  c = cached[3, 3]
  if got_ids != ref_ids
    failures << "exact: ids differ from the reference\n      ref: #{ref_ids.inspect}\n      got: #{got_ids.inspect}"
  elsif c[1] != IDS.length - 1 || c[2] != 5
    failures << "exact: cached_tokens #{c.inspect}; expected #{IDS.length - 1} for the repeat and 5 for the shorter prompt"
  else
    ok.call("exact", "a repeated prompt and a shorter one return the reference ids (cached #{c.inspect})")
  end

  # ---- save ----
  sysfile = File.join(TMP, "system.txt")
  kvfile  = File.join(TMP, "system.kv")
  File.write(sysfile, SYSTEM + "\n")
  port, = start_server(GGUF, "PREFIX_FILE" => sysfile, "PREFIX_CACHE" => kvfile)
  first = chat(port, SYSTEM, Q1)
  c = cached
  if !File.file?(kvfile) || File.size(kvfile) < 1024
    failures << "save: no KV prefix file was written"
  elsif c[0] != SYS_N
    failures << "save: first request cached #{c[0].inspect} tokens, the system turn is #{SYS_N}"
  elsif first != ref[0]
    failures << "save: reply differs from the reference: #{first.inspect}"
  else
    ok.call("save", "file written (#{File.size(kvfile)} bytes); first request has #{c[0]} tokens cached")
  end
  saved = File.binread(kvfile)

  # ---- restore: the file alone ----
  port, st = start_server(GGUF, "PREFIX_CACHE" => kvfile)
  if port.nil?
    failures << "restore: boot with only the KV prefix file exited #{st}"
  else
    first = chat(port, SYSTEM, Q1)
    c = cached
    if c[0] != SYS_N
      failures << "restore: first request cached #{c[0].inspect} tokens, expected #{SYS_N}"
    elsif first != ref[0]
      failures << "restore: reply differs from the reference: #{first.inspect}"
    else
      ok.call("restore", "a boot from the file alone serves the first request with #{c[0]} tokens cached, same reply")
    end
  end

  # ---- stale ----
  stale = []
  # "Another model": the same file with 8 MiB of weights changed in the
  # middle — same size, same shape, same metadata.
  other = File.join(TMP, "other.gguf")
  FileUtils.cp(GGUF, other)
  File.open(other, "r+b") do |f|
    f.seek(File.size(other) / 2)
    span = f.read(8 * 1024 * 1024)
    f.seek(File.size(other) / 2)
    f.write(span.bytes.map { |b| b ^ 1 }.pack("C*"))
  end
  port, st = start_server(other, "PREFIX_CACHE" => kvfile)
  stale << "accepted with a different model file" unless port.nil? && st != 0
  bad = saved.dup
  at = bad.index("toy-kv-prefix-1") + 15 + 5 + 3   # inside the fingerprint string
  bad.setbyte(at, bad.getbyte(at) ^ 1)
  File.binwrite(kvfile, bad)
  port, st = start_server(GGUF, "PREFIX_CACHE" => kvfile)
  stale << "accepted with a damaged fingerprint" unless port.nil? && st != 0
  File.binwrite(kvfile, saved[0, saved.bytesize / 2])
  port, st = start_server(GGUF, "PREFIX_CACHE" => kvfile)
  stale << "accepted when truncated" unless port.nil? && st != 0
  # A different prefix file: the old cache must be rebuilt, not used.
  File.binwrite(kvfile, saved)
  other_sys = "Be terse."
  File.write(sysfile, other_sys + "\n")
  other_n = encode("<|im_start|>system\n#{other_sys}<|im_end|>\n").length
  port, st = start_server(GGUF, "PREFIX_FILE" => sysfile, "PREFIX_CACHE" => kvfile)
  if port.nil?
    stale << "boot with a changed --prefix-file exited #{st}"
  else
    chat(port, other_sys, Q1)
    c = cached
    stale << "changed --prefix-file: cached #{c[0].inspect}, expected #{other_n}" unless c[0] == other_n
    stale << "changed --prefix-file: the cache file was not rewritten" if File.binread(kvfile) == saved
  end
  if stale.empty?
    ok.call("stale", "refused for another model, a damaged fingerprint and a truncated file; rebuilt for a changed prefix")
  else
    stale.each { |m| failures << "stale: #{m}" }
  end
rescue StandardError => e
  failures << "gate aborted: #{e.message.lines.first.to_s.strip}"
ensure
  stop_server
end

if failures.empty?
  puts "GATE PASS [#{LABEL}]: cached prompt prefixes are reused, saved and restored with unchanged output, on #{DEVICE}"
  exit 0
end
failures.each { |m| puts "  FAIL: #{m}" }
puts "GATE FAIL [#{LABEL}]: #{failures.length} failure(s)"
exit 1
