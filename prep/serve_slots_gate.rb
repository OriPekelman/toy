#!/usr/bin/env ruby
# prep/serve_slots_gate.rb — toy#217 gate: `toy serve --slots N` decodes
# several requests as one batch.
#
# Boots the serve runner (CPU by default, `--device cuda` for
# libexec/toy-serve-cuda) on data/smollm2-135m-tok.gguf (F32, embedded
# tokenizer) and checks over real HTTP. The REFERENCE is a one-slot server
# answering the same requests one after another.
#
#   same      8 requests sent at once to a 4-slot server return the
#             reference ids, and so do the same 8 sent one at a time
#   batched   those requests really shared decode steps: every request
#             event of the concurrent round reports shared_steps > 0, more
#             than one slot was used, and the one-at-a-time round reports
#             shared_steps 0 (the client seeing overlapping requests proves
#             nothing: a one-slot server queues them)
#   queue     8 requests at once on 2 slots are all answered with the
#             reference ids (the extra ones wait for a slot)
#   stream    4 streamed chat requests at once: each stream's content is
#             the reply the request gives alone, and ends with [DONE]
#   options   stop strings and logprobs are per request: two requests at
#             once, one with a stop string and logprobs, the other plain,
#             each give what they give alone
#   prefix    with a boot prefix (--prefix-file) every concurrent request
#             reads the shared prefix cells (cached_tokens >= the prefix
#             length) and the replies equal the reference server's, which
#             has no prefix
#   fault     a request that raises among concurrent ones is HTTP 500;
#             the others are answered with the reference ids
#   refuse    --slots 2 on a per-head cache (KV_FUSED=0) exits with a
#             message instead of serving
#
#   ruby prep/serve_slots_gate.rb [--device cuda]   # exit 0 pass, 1 fail, 2 unusable

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
GGUF    = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
LABEL   = DEVICE == "cuda" ? "serve-slots-cuda" : "serve-slots"
MAX_TOKENS = 20

unless %w[cpu cuda].include?(DEVICE)
  warn "serve_slots_gate: --device must be cpu or cuda"
  exit 2
end
unless File.file?(GGUF)
  warn "serve_slots_gate: missing fixture #{GGUF} (never skipped silently)"
  exit 2
end
unless ENV["TOY_GATE_RUNNER"]
  out, st = Open3.capture2e("make", "-C", ROOT, TARGET)
  unless st.success?
    warn "serve_slots_gate: `make #{TARGET}` failed:\n#{out.lines.last(15).join}"
    exit 2
  end
end

STDBUF_BIN = %w[/usr/bin/stdbuf /bin/stdbuf /opt/homebrew/bin/gstdbuf].find { |b| File.executable?(b) }
STDBUF = STDBUF_BIN ? [STDBUF_BIN, "-oL"].freeze : [].freeze
WORK = Dir.mktmpdir("serve_slots_gate_")

# ---- server lifecycle (one at a time; always torn down) --------------------

@pid = nil
@logf = nil
@run_dir = nil

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

at_exit do
  stop_server
  FileUtils.rm_rf(WORK)
end
%w[INT TERM].each { |s| trap(s) { exit 130 } }

# Boots the runner; returns the port, or raises with the log tail. Each
# server gets its own run directory, so its request events can be read.
def start_server(extra_env = {})
  stop_server
  sock = TCPServer.new("127.0.0.1", 0)
  port = sock.addr[1]
  sock.close
  @run_dir = File.join(WORK, "run_#{port}")
  FileUtils.mkdir_p(@run_dir)
  @logf = File.join(WORK, "serve_#{port}.log")
  env = { "MODEL_PATH" => GGUF, "MODEL_NAME" => "gate", "PORT" => port.to_s,
          "TOY_RUN_DIR" => @run_dir, "TOY_RUN_ID" => "slots-gate" }.merge(extra_env)
  File.open(@logf, "w") do |lf|
    @pid = Process.spawn(env, *STDBUF, RUNNER, chdir: ROOT, pgroup: true, out: lf, err: lf, in: File::NULL)
  end
  deadline = Time.now + 180
  while Time.now < deadline
    if Process.waitpid(@pid, Process::WNOHANG)
      @pid = nil
      raise "server exited before ready:\n#{File.read(@logf).lines.last(8).join}"
    end
    begin
      return port if Net::HTTP.get_response(URI("http://127.0.0.1:#{port}/health")).code == "200"
    rescue StandardError
      # not listening yet
    end
    sleep 0.2
  end
  raise "server not ready after 180 s:\n#{File.read(@logf).lines.last(8).join}"
end

def post(port, path, body)
  res = Net::HTTP.start("127.0.0.1", port, read_timeout: 300) do |h|
    h.post(path, JSON.generate(body), "Content-Type" => "application/json")
  end
  [res.code.to_i, res.body]
end

def chatml(user, system = nil)
  s = +""
  s << "<|im_start|>system\n#{system}<|im_end|>\n" if system
  s << "<|im_start|>user\n#{user}<|im_end|>\n<|im_start|>assistant\n"
  s
end

# /v1/completions with a text prompt: the generated ids, or the HTTP status.
def ids_of(port, prompt, extra = {})
  code, body = post(port, "/v1/completions", { model: "gate", prompt: prompt, max_tokens: MAX_TOKENS }.merge(extra))
  return code unless code == 200
  JSON.parse(body).dig("choices", 0, "ids")
end

def all_at_once(items)
  items.map { |it| Thread.new { yield it } }.map(&:value)
end

# The request events of the CURRENT server, in order.
def request_events
  f = File.join(@run_dir, "events.jsonl")
  return [] unless File.file?(f)
  File.readlines(f).map { |l| JSON.parse(l) }.select { |e| e["name"] == "request" }.map { |e| e["extra"] }
end

USERS = ["What is 2+2?", "Name three colours.", "Tell me about the sea.", "Why is the sky blue?",
         "Write a haiku about rain.", "What is the capital of France?", "Count from one to five.",
         "Who wrote Hamlet?"].freeze
PROMPTS = USERS.map { |u| chatml(u) }.freeze
SYSTEM  = "You are a careful assistant. Answer in one short sentence and never use lists."
SYS_PROMPTS = USERS.map { |u| chatml(u, SYSTEM) }.freeze
STOP_PROMPT = PROMPTS[0]

failures = []
ok = ->(leg, msg) { puts "  ok: #{leg} — #{msg}" }

begin
  # ---- reference: one slot, one request at a time ----
  port = start_server
  ref     = PROMPTS.map { |p| ids_of(port, p) }
  ref_sys = SYS_PROMPTS.map { |p| ids_of(port, p) }
  ref_chat = USERS.first(4).map do |u|
    _, body = post(port, "/v1/chat/completions", { model: "gate", messages: [{ role: "user", content: u }], max_tokens: MAX_TOKENS })
    JSON.parse(body).dig("choices", 0, "message", "content")
  end
  # A stop string taken from inside the first reply, and that reply's logprobs.
  _, body = post(port, "/v1/completions", { model: "gate", prompt: STOP_PROMPT, max_tokens: MAX_TOKENS })
  ref_text = JSON.parse(body).dig("choices", 0, "text")
  stop_str = ref_text[ref_text.length / 2, 3]
  _, body = post(port, "/v1/completions", { model: "gate", prompt: STOP_PROMPT, max_tokens: MAX_TOKENS, stop: stop_str, logprobs: 2 })
  ref_opt = JSON.parse(body).dig("choices", 0)
  unless ref.all? { |r| r.is_a?(Array) && r.length == MAX_TOKENS } && ref.uniq.length > 4
    raise "reference server: expected #{MAX_TOKENS} ids per prompt and distinct replies, got #{ref.inspect[0, 200]}"
  end
  unless ref_opt["finish_reason"] == "stop" && ref_opt["text"].length < ref_text.length && ref_opt.dig("logprobs", "content").is_a?(Array)
    raise "reference server: the stop-string request did not stop early: #{ref_opt.inspect[0, 200]}"
  end

  # ---- same + batched: 4 slots ----
  port = start_server("SLOTS" => "4")
  one_by_one = PROMPTS.map { |p| ids_of(port, p) }
  ev_seq = request_events
  at_once = all_at_once(PROMPTS) { |p| ids_of(port, p) }
  ev_con = request_events.drop(ev_seq.length)
  if one_by_one != ref
    bad = one_by_one.each_index.reject { |k| one_by_one[k] == ref[k] }
    failures << "same: one at a time on 4 slots, prompts #{bad.inspect} differ from the one-slot reference"
  elsif at_once != ref
    bad = at_once.each_index.reject { |k| at_once[k] == ref[k] }
    failures << "same: sent at once on 4 slots, prompts #{bad.inspect} differ from the reference " \
                "(first: #{at_once[bad[0]].inspect} vs #{ref[bad[0]].inspect})"
  else
    ok.call("same", "#{PROMPTS.length} requests at once on 4 slots return the one-slot ids (#{MAX_TOKENS} each)")
  end
  slots_used = ev_con.map { |e| e["slot"] }.uniq
  if ev_seq.length != PROMPTS.length || ev_con.length != PROMPTS.length
    failures << "batched: expected #{PROMPTS.length} request events per round, got #{ev_seq.length} and #{ev_con.length}"
  elsif ev_seq.any? { |e| e["shared_steps"] != 0 }
    failures << "batched: one-at-a-time requests report shared decode steps: #{ev_seq.map { |e| e["shared_steps"] }.inspect}"
  elsif ev_con.any? { |e| !e["shared_steps"].is_a?(Integer) || e["shared_steps"] <= 0 } || slots_used.length < 2
    failures << "batched: concurrent requests did not share decode steps: shared_steps " \
                "#{ev_con.map { |e| e["shared_steps"] }.inspect}, slots #{slots_used.inspect}"
  else
    ok.call("batched", "every concurrent request shared decode steps (#{ev_con.map { |e| e["shared_steps"] }.minmax.join("-")} " \
                       "each, slots #{slots_used.sort.inspect}); none did one at a time")
  end

  # ---- stream + options, same server ----
  streams = all_at_once(USERS.first(4)) do |u|
    code, body = post(port, "/v1/chat/completions", { model: "gate", stream: true, max_tokens: MAX_TOKENS,
                                                      messages: [{ role: "user", content: u }] })
    frames = body.split("\n\n").map { |f| f.sub(/\Adata: /, "") }.reject(&:empty?)
    done = frames.last == "[DONE]"
    text = frames.reject { |f| f == "[DONE]" }.map { |f| JSON.parse(f).dig("choices", 0, "delta", "content").to_s }.join
    [code, done, text]
  end
  if streams.any? { |c, d, _| c != 200 || !d }
    failures << "stream: a concurrent stream was not 200 or did not end with [DONE]: #{streams.map { |c, d, _| [c, d] }.inspect}"
  elsif streams.map(&:last) != ref_chat
    failures << "stream: concurrent streams differ from the replies alone: #{streams.map(&:last).inspect} vs #{ref_chat.inspect}"
  else
    ok.call("stream", "4 streams at once each carry the reply their request gives alone")
  end

  pair = all_at_once([true, false]) do |with|
    extra = with ? { stop: stop_str, logprobs: 2 } : {}
    _, body = post(port, "/v1/completions", { model: "gate", prompt: STOP_PROMPT, max_tokens: MAX_TOKENS }.merge(extra))
    JSON.parse(body).dig("choices", 0)
  end
  lp_ids = ->(c) { c.dig("logprobs", "content").map { |e| [e["id"], e["top_logprobs"].map { |a| a["id"] }] } }
  if pair[0]["text"] != ref_opt["text"] || pair[0]["finish_reason"] != "stop" || lp_ids.call(pair[0]) != lp_ids.call(ref_opt)
    failures << "options: the stop-string + logprobs request changed next to a plain one: #{pair[0]["text"].inspect} vs #{ref_opt["text"].inspect}"
  elsif pair[1]["text"] != ref_text || pair[1].key?("logprobs")
    failures << "options: the plain request took the other request's options: #{pair[1].inspect[0, 200]}"
  else
    ok.call("options", "a stop string and logprobs apply to their own request only (#{pair[0]["text"].length} vs #{pair[1]["text"].length} characters)")
  end

  # ---- fault: one of four raises ----
  port = start_server("SLOTS" => "4", "SERVE_TEST_FAULT_AT" => "7")
  fault_prompt = (1..7).to_a   # exactly 7 token ids
  mixed = all_at_once([0, 1, :fault, 2]) do |k|
    k == :fault ? ids_of(port, fault_prompt) : ids_of(port, PROMPTS[k])
  end
  after = ids_of(port, PROMPTS[3])
  if mixed[2] != 500
    failures << "fault: the failing request answered #{mixed[2].inspect}, expected HTTP 500"
  elsif [mixed[0], mixed[1], mixed[3]] != [ref[0], ref[1], ref[2]] || after != ref[3]
    failures << "fault: requests next to (or after) the failing one differ from the reference"
  else
    ok.call("fault", "one request raising among four is HTTP 500; the other three and the next one are unaffected")
  end

  # ---- queue: more requests than slots ----
  port = start_server("SLOTS" => "2")
  queued = all_at_once(PROMPTS) { |p| ids_of(port, p) }
  ev = request_events
  if queued != ref
    bad = queued.each_index.reject { |k| queued[k] == ref[k] }
    failures << "queue: #{PROMPTS.length} requests on 2 slots, prompts #{bad.inspect} differ (#{queued[bad[0]].inspect[0, 80]})"
  elsif ev.map { |e| e["slot"] }.uniq.sort != [0, 1]
    failures << "queue: expected both slots in use, got #{ev.map { |e| e["slot"] }.inspect}"
  else
    ok.call("queue", "#{PROMPTS.length} requests at once on 2 slots are all answered with the reference ids")
  end

  # ---- prefix: the boot prefix is shared by every slot ----
  pfile = File.join(WORK, "system.txt")
  File.write(pfile, SYSTEM + "\n")
  port = start_server("SLOTS" => "4", "PREFIX_FILE" => pfile)
  n_prefix = File.read(@logf)[/KV prefix: evaluated (\d+) tokens/, 1].to_i
  shared = all_at_once(SYS_PROMPTS) { |p| ids_of(port, p) }
  ev = request_events
  if n_prefix < 10
    failures << "prefix: the server did not report evaluating the boot prefix"
  elsif shared != ref_sys
    bad = shared.each_index.reject { |k| shared[k] == ref_sys[k] }
    failures << "prefix: with a shared boot prefix, prompts #{bad.inspect} differ from the reference without one"
  elsif ev.length != SYS_PROMPTS.length || ev.any? { |e| e["cached_tokens"] < n_prefix }
    failures << "prefix: cached_tokens #{ev.map { |e| e["cached_tokens"] }.inspect}; expected at least the #{n_prefix}-token prefix each"
  else
    ok.call("prefix", "#{SYS_PROMPTS.length} concurrent requests each read the #{n_prefix} shared prefix cells and return the reference ids")
  end

  # ---- refuse: several slots need the fused cache ----
  stop_server
  sock = TCPServer.new("127.0.0.1", 0)
  rport = sock.addr[1]
  sock.close
  out, st = Timeout.timeout(120) do
    Open3.capture2e({ "MODEL_PATH" => GGUF, "MODEL_NAME" => "gate", "PORT" => rport.to_s,
                      "SLOTS" => "2", "KV_FUSED" => "0" }, RUNNER, chdir: ROOT, stdin_data: "")
  end
  if st.success? || out !~ /--slots 2 needs the fused decode graph/
    failures << "refuse: SLOTS=2 with KV_FUSED=0 should exit with a message; status #{st.exitstatus.inspect}, tail #{out.lines.last(2).join.inspect}"
  else
    ok.call("refuse", "--slots 2 on a per-head cache exits #{st.exitstatus} with a message")
  end
rescue StandardError => e
  failures << "gate error: #{e.class}: #{e.message}"
end

stop_server
if failures.empty?
  puts "GATE PASS [#{LABEL}]: toy serve --slots decodes concurrent requests as one batch, on #{DEVICE}"
  exit 0
end
failures.each { |f| puts "  FAIL: #{f}" }
puts "GATE FAIL [#{LABEL}]"
exit 1
