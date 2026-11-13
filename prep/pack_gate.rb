#!/usr/bin/env ruby
# frozen_string_literal: true

# prep/pack_gate.rb — toy#211 gate: `toy pack` builds one executable that
# carries the model, config, description and KV prefix and serves with no
# arguments.
#
# Packs data/smollm2-135m-tok.gguf (CPU by default, `--device cuda` for the
# CUDA runner) with a system-message prefix, then holds:
#
#   alone     the artifact, started with an EMPTY environment, no arguments
#             and an empty working directory, serves on the packed port
#             under the packed name. Nothing else on the command line or in
#             the environment could have told it where a model is.
#   same      its replies (chat with the packed system message, and
#             /v1/completions ids) equal those of the plain serve runner on
#             the GGUF file with the same --prefix-file
#   prefix    the KV prefix was RESTORED from the executable, with the token
#             count the plain runner reports evaluating, and not evaluated
#   mapped    the model is mapped from the executable itself, read-only, at
#             the model entry's offset and at least its length
#             (/proc/<pid>/maps; Linux only)
#   fixed     MODEL_PATH / MODEL_NAME / MAX_T in the environment do not
#             change what a pack serves; PORT moves the port and --port
#             beats PORT
#   describe  --describe prints the model's sha256 and the toy and ggml
#             commits, loads nothing, exits 0; an unknown argument exits 2
#   damaged   a pack whose trailer is damaged or cut refuses to start and
#             says so (it does not turn into a plain runner looking for a
#             model elsewhere); a pack whose KV entry was made from another
#             model refuses to boot
#   bare      a pack made without --prefix serves, with no KV prefix
#   usage     `toy pack` without -o, or with a missing model, exits 2
#
#   ruby prep/pack_gate.rb [--device cuda]   # exit 0 pass, 1 fail, 2 unusable

require "net/http"
require "json"
require "socket"
require "timeout"
require "tmpdir"
require "open3"
require "fileutils"
require "digest"

ROOT   = File.expand_path("..", __dir__)
DEVICE = (i = ARGV.index("--device")) ? ARGV[i + 1].to_s : "cpu"
TARGET = DEVICE == "cuda" ? "libexec/toy-serve-cuda" : "libexec/toy-serve"
RUNNER = File.join(ROOT, TARGET)
TOY    = File.join(ROOT, "bin", "toy")
GGUF   = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
LABEL  = DEVICE == "cuda" ? "pack-cuda" : "pack"
NAME   = "packed-gate"
SYSTEM = "You are a terse assistant. Answer in one short sentence."

unless %w[cpu cuda].include?(DEVICE)
  warn "pack_gate: --device must be cpu or cuda"
  exit 2
end
unless File.file?(GGUF)
  warn "pack_gate: missing fixture #{GGUF} (never skipped silently)"
  exit 2
end
out, st = Open3.capture2e("make", "-C", ROOT, TARGET)
unless st.success?
  warn "pack_gate: `make #{TARGET}` failed:\n#{out.lines.last(15).join}"
  exit 2
end

# Heavy files go next to the repo's other scratch output, not on /tmp.
TMP_BASE = File.directory?("/srv/data/scratch") && File.writable?("/srv/data/scratch") ? "/srv/data/scratch" : Dir.tmpdir
TMP   = Dir.mktmpdir("toy211-gate-", TMP_BASE)
EMPTY = File.join(TMP, "empty")
FileUtils.mkdir_p(EMPTY)
# A runner's stdout is block-buffered when it is a file; the legs read boot
# lines while the server is still up, so run it line-buffered. stdbuf execs
# the runner, so the pid (and /proc/<pid>/maps) is the runner's.
STDBUF_BIN = %w[/usr/bin/stdbuf /bin/stdbuf /opt/homebrew/bin/gstdbuf].find { |b| File.executable?(b) }
if STDBUF_BIN.nil?
  warn "pack_gate: needs stdbuf (coreutils) to read a running server's boot lines"
  exit 2
end
STDBUF = [STDBUF_BIN, "-oL"].freeze
@pid = nil
@boots = 0
@log = nil

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

at_exit { stop_server; FileUtils.rm_rf(TMP) unless ENV["PACK_GATE_KEEP"] == "1" }
%w[INT TERM].each { |s| trap(s) { stop_server; exit 130 } }

def free_port
  s = TCPServer.new("127.0.0.1", 0)
  p = s.addr[1]
  s.close
  p
end

# Starts `argv` and waits for `port`. Returns true when it serves, or the
# exit status when the process exits first. `clean` empties the environment.
def boot(argv, env, port, chdir:, clean:)
  stop_server
  @boots += 1
  @log = File.join(TMP, "srv#{@boots}.log")
  @pid = Process.spawn(env, *STDBUF, *argv, chdir: chdir, pgroup: true, out: @log, err: @log,
                       unsetenv_others: clean)
  deadline = Time.now + 180
  while Time.now < deadline
    if Process.waitpid(@pid, Process::WNOHANG)
      @pid = nil
      return $?.exitstatus || 128
    end
    begin
      r = Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 5) { |h| h.get("/health") }
      return true if r.code == "200"
    rescue Errno::ECONNREFUSED, Errno::ECONNRESET, EOFError, Timeout::Error, SocketError,
           Net::OpenTimeout, Net::ReadTimeout
    end
    sleep 0.25
  end
  tail = File.exist?(@log) ? File.readlines(@log).last(4).map(&:strip).join(" | ") : "no log"
  raise "server never became ready on port #{port} (#{File.basename(argv.first)}; log ends: #{tail})"
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

def model_id(port)
  r = Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 10) { |h| h.get("/v1/models") }
  JSON.parse(r.body).dig("data", 0, "id")
end

PROMPT_IDS = (1..12).map { |i| 200 + (i * 37) % 4000 }.freeze

# The replies the `same` leg compares: two chats with the packed system
# message, then a completion on raw ids.
def replies(port)
  users = ["What is the capital of France?", "Name a colour."]
  chats = users.map do |u|
    j = post(port, "/v1/chat/completions", "model" => "gate", "max_tokens" => 10,
             "messages" => [{ "role" => "system", "content" => SYSTEM }, { "role" => "user", "content" => u }])
    [j.dig("choices", 0, "message", "content"), j.dig("usage", "prompt_tokens")]
  end
  ids = post(port, "/v1/completions", "model" => "gate", "prompt" => PROMPT_IDS, "max_tokens" => 8).dig("choices", 0, "ids")
  [chats, ids]
end

# [[name, offset, length], ...] from an artifact's trailer; [] when none.
def pack_index(path)
  File.open(path, "rb") do |f|
    return [] if f.size < 24
    f.seek(-24, IO::SEEK_END)
    tr = f.read(24)
    return [] unless tr[0, 8] == "TOYPACK1"
    ioff, ilen = tr[8, 16].unpack("Q<Q<")
    f.seek(ioff)
    f.read(ilen).lines.map { |l| n, o, len = l.split; [n, o.to_i, len.to_i] }
  end
end

def toy_pack(*args)
  Open3.capture2e({ "TOY_HOME" => ROOT }, TOY, "pack", *args, chdir: ROOT)
end

failures = []
ok = ->(leg, what) { puts "  ok: #{leg} — #{what}" }

begin
  sys_file = File.join(TMP, "system.txt")
  File.write(sys_file, SYSTEM + "\n")
  art  = File.join(TMP, "packed")
  port = free_port
  out, st = toy_pack(GGUF, "--prefix", sys_file, "--name", NAME, "--device", DEVICE,
                     "--port", port.to_s, "-o", art)
  raise "`toy pack` exited #{st.exitstatus}: #{out.lines.last(6).join.strip}" unless st.success? && File.executable?(art)
  index = pack_index(art)
  model_entry = index.find { |e| e[0] == "model" }
  kv_entry    = index.find { |e| e[0] == "kv" }
  raise "the artifact has no model/kv entries: #{index.inspect}" if model_entry.nil? || kv_entry.nil?

  # ---- reference: the plain runner on the GGUF file ----
  ref_port = free_port
  r = boot([RUNNER], { "MODEL_PATH" => GGUF, "MODEL_NAME" => NAME, "PORT" => ref_port.to_s,
                       "PREFIX_FILE" => sys_file }, ref_port, chdir: ROOT, clean: false)
  raise "the reference server exited #{r}" unless r == true
  ref = replies(ref_port)
  ref_tokens = File.read(@log)[/KV prefix: evaluated (\d+) tokens/, 1].to_i
  raise "the reference server did not report evaluating the prefix" if ref_tokens <= 0

  # ---- alone ----
  r = boot([art], {}, port, chdir: EMPTY, clean: true)
  if r == true && model_id(port) == NAME
    ok.call("alone", "empty environment, no arguments, empty directory: serves #{NAME} on the packed port")
  else
    failures << "alone: the artifact #{r == true ? "served #{model_id(port).inspect}" : "exited #{r}"}; " \
                "expected it to serve #{NAME} on port #{port}"
    raise "the artifact does not serve; the remaining legs cannot run"
  end
  packed_log = File.read(@log)
  pid = @pid

  # ---- mapped ----
  if File.readable?("/proc/#{pid}/maps")
    real = File.realpath(art)
    maps = File.readlines("/proc/#{pid}/maps").map(&:split).select { |f| f[5] == real }
    hit = maps.find do |f|
      a, b = f[0].split("-").map { |x| x.to_i(16) }
      f[2].to_i(16) == model_entry[1] && b - a >= model_entry[2] && f[1].start_with?("r-")
    end
    if hit
      ok.call("mapped", "the #{model_entry[2]}-byte model is mapped read-only from the executable at offset #{model_entry[1]}")
    else
      failures << "mapped: no read-only mapping of the executable at the model entry's offset " \
                  "#{model_entry[1]} covering #{model_entry[2]} bytes (#{maps.length} mappings of it)"
    end
  else
    puts "  not run: mapped — no /proc/<pid>/maps on this platform"
  end

  # ---- same ----
  got = replies(port)
  if got == ref
    ok.call("same", "2 chat replies and 1 completion equal the plain runner's on the GGUF file")
  else
    failures << "same: replies differ from the plain runner's (packed #{got.inspect} vs #{ref.inspect})"
  end

  # ---- prefix ----
  restored = packed_log[/KV prefix: restored (\d+) tokens from @pack:kv/, 1].to_i
  p = []
  p << "restored #{restored} tokens from the executable, the plain runner evaluated #{ref_tokens}" unless restored == ref_tokens
  p << "the packed boot evaluated a prefix" if packed_log.include?("KV prefix: evaluated")
  if p.empty?
    ok.call("prefix", "#{restored}-token KV prefix restored from the executable, not evaluated")
  else
    p.each { |m| failures << "prefix: #{m}" }
  end

  # ---- fixed ----
  f = []
  p2 = free_port
  r = boot([art], { "MODEL_PATH" => "/nonexistent.gguf", "MODEL_NAME" => "other", "MAX_T" => "8",
                    "PORT" => p2.to_s }, p2, chdir: EMPTY, clean: true)
  if r == true
    f << "MODEL_NAME from the environment renamed the pack to #{model_id(p2).inspect}" unless model_id(p2) == NAME
    f << "MODEL_PATH / MAX_T from the environment changed the replies" unless replies(p2) == ref
  else
    f << "with MODEL_PATH/MODEL_NAME/MAX_T/PORT set the artifact exited #{r}"
  end
  p3 = free_port
  r = boot([art, "--port", p3.to_s], { "PORT" => p2.to_s }, p3, chdir: EMPTY, clean: true)
  f << "--port #{p3} with PORT=#{p2}: #{r == true ? 'ok' : "exited #{r}"}" unless r == true
  stop_server
  if f.empty?
    ok.call("fixed", "MODEL_PATH, MODEL_NAME and MAX_T are ignored; PORT moves the port; --port beats PORT")
  else
    f.each { |m| failures << "fixed: #{m}" }
  end

  # ---- describe ----
  d = []
  text, _err, st = Open3.capture3(art, "--describe", chdir: EMPTY, unsetenv_others: true)
  sha = Digest::SHA256.file(GGUF).hexdigest
  toy_rev  = `git -C #{ROOT} rev-parse HEAD`.strip
  ggml_rev = `git -C #{File.join(ROOT, "vendor", "ggml")} rev-parse HEAD`.strip
  d << "--describe exited #{st.exitstatus}" unless st.success?
  d << "no model sha256 #{sha}" unless text.include?(sha)
  d << "no toy commit #{toy_rev}" unless text.include?(toy_rev)
  d << "no ggml commit #{ggml_rev}" unless text.include?(ggml_rev)
  d << "no card (Algorithm: line)" unless text.include?("Algorithm:")
  d << "--describe loaded the model" if text.include?("loading config")
  _o, st = Open3.capture2e(art, "--bogus", chdir: EMPTY, unsetenv_others: true)
  d << "an unknown argument exited #{st.exitstatus}, expected 2" unless st.exitstatus == 2
  if d.empty?
    ok.call("describe", "card, model sha256, toy and ggml commits; nothing loaded; unknown argument exits 2")
  else
    d.each { |m| failures << "describe: #{m}" }
  end

  # ---- damaged ----
  g = []
  size = File.size(art)
  { "a damaged trailer" => ->(fh) { fh.seek(size - 24); fh.write("XOYPACK1") },
    "a cut file"        => ->(fh) { fh.truncate(size - 1) } }.each do |what, damage|
    bad = File.join(TMP, "bad")
    FileUtils.cp(art, bad)
    File.open(bad, "r+b") { |fh| damage.call(fh) }
    bp = free_port
    r = boot([bad], { "PORT" => bp.to_s }, bp, chdir: EMPTY, clean: true)
    if r == true
      g << "#{what}: still serves #{model_id(bp).inspect}"
    else
      g << "#{what}: exited #{r}, expected 1" unless r == 1
      g << "#{what}: no 'damaged pack' message" unless File.read(@log).include?("damaged pack")
    end
    stop_server
    FileUtils.rm_f(bad)
  end
  # A KV entry made from another model: change the model fingerprint it
  # records ("<size>-<hash>", an ASCII string near the entry's start).
  bad = File.join(TMP, "bad")
  FileUtils.cp(art, bad)
  fp_at = nil
  File.open(bad, "r+b") do |fh|
    fh.seek(kv_entry[1])
    head = fh.read([kv_entry[2], 4096].min)
    m = head.match(/#{File.size(GGUF)}-[0-9a-f]{16}/n)
    if m
      fp_at = kv_entry[1] + m.begin(0)
      fh.seek(fp_at)
      fh.write("9")
    end
  end
  if fp_at.nil?
    g << "could not find the model fingerprint in the KV entry"
  else
    bp = free_port
    r = boot([bad], { "PORT" => bp.to_s }, bp, chdir: EMPTY, clean: true)
    g << "a KV entry from another model: #{r == true ? 'served anyway' : "exited #{r}, expected 1"}" unless r == 1
    g << "a KV entry from another model: no 'repack the artifact' message" unless File.read(@log).include?("repack the artifact")
    stop_server
  end
  FileUtils.rm_f(bad)
  if g.empty?
    ok.call("damaged", "a damaged or cut pack refuses to start; a KV entry from another model refuses to boot")
  else
    g.each { |m| failures << "damaged: #{m}" }
  end

  # ---- bare ----
  b = []
  bare = File.join(TMP, "bare")
  bport = free_port
  out, st = toy_pack(GGUF, "--name", NAME, "--device", DEVICE, "--port", bport.to_s, "-o", bare)
  if st.success?
    b << "a pack without --prefix has a kv entry" if pack_index(bare).any? { |e| e[0] == "kv" }
    r = boot([bare], {}, bport, chdir: EMPTY, clean: true)
    if r == true
      b << "a pack without --prefix reports a KV prefix" if File.read(@log).include?("KV prefix:")
      b << "a pack without --prefix gives other replies" unless replies(bport) == ref
    else
      b << "a pack without --prefix exited #{r}"
    end
    stop_server
  else
    b << "`toy pack` without --prefix exited #{st.exitstatus}: #{out.lines.last.to_s.strip}"
  end
  FileUtils.rm_f(bare)
  if b.empty?
    ok.call("bare", "a pack without --prefix serves the same replies with no KV prefix")
  else
    b.each { |m| failures << "bare: #{m}" }
  end

  # ---- usage ----
  u = []
  _o, st = toy_pack(GGUF)
  u << "without -o: exit #{st.exitstatus}, expected 2" unless st.exitstatus == 2
  _o, st = toy_pack(File.join(TMP, "absent.gguf"), "-o", File.join(TMP, "x"))
  u << "with a missing model: exit #{st.exitstatus}, expected 2" unless st.exitstatus == 2
  u << "a failed pack left an artifact behind" if File.exist?(File.join(TMP, "x"))
  if u.empty?
    ok.call("usage", "`toy pack` without -o or with a missing model exits 2 and writes nothing")
  else
    u.each { |m| failures << "usage: #{m}" }
  end
rescue StandardError => e
  failures << "gate aborted: #{e.message.lines.first.to_s.strip}"
end

if failures.empty?
  puts "GATE PASS [#{LABEL}]: one executable carries the model, config and KV prefix and serves alone, on #{DEVICE}"
  exit 0
end
failures.each { |m| puts "  FAIL: #{m}" }
puts "GATE FAIL [#{LABEL}]: #{failures.length} failure(s)"
exit 1
