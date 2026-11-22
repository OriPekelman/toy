#!/usr/bin/env ruby
# prep/serve_chat_gate.rb — toy#209 gate: `toy serve` as a text endpoint.
#
# Boots the serve runner (CPU by default, `--device cuda` for
# libexec/toy-serve-cuda) on data/smollm2-135m-tok.gguf, which embeds a
# tokenizer, and checks over real HTTP:
#
#   text      /v1/completions returns decoded `text` next to `ids`, and the
#             text re-encodes to exactly those ids (independent encoder run)
#   long      max_tokens 300 yields 300 tokens (the ceiling was a literal 256)
#   chat      /v1/chat/completions is 200 with content; usage.prompt_tokens
#             equals the id count of the ChatML text with each special token
#             as ONE id
#   nonascii  a message with non-ASCII content answers (tep read the body by
#             character count and stalled) and counts its prompt correctly
#   escapes   the same message with non-ASCII written as \\uXXXX escapes
#             (surrogate pair included) is the same request (toy#215: the
#             JSON decoder turned each escape into "?")
#   stream    "stream":true frames concatenate to the non-streamed content,
#             every frame is valid JSON, and the stream ends with [DONE]
#   stop      with STOP_IDS the same request ends early with finish_reason
#             "stop" and the stop id is not in the reply; without it the
#             request runs to max_tokens with "length"
#   empty     a zero-token reply (first id is a stop id) is answered and the
#             server keeps serving (toy#214: decoding nothing raised
#             FrozenError and ended the process)
#   fault     a request that raises inside generation is answered with
#             HTTP 500 and an error object; the next request is unaffected
#   logprobs  (toy#216) "logprobs":true,"top_logprobs":n gives one entry per
#             reply token: the token picked is the most likely alternative,
#             the alternatives are sorted, their probabilities sum to at
#             most 1, the bytes spell the content, and the content is the
#             one the request gives without logprobs
#   prefix    (toy#216) a trailing partial assistant message is continued:
#             the first 4 tokens of a reply, sent back as that message,
#             give the rest of the reply, and the prompt is 4 tokens longer
#   textprompt (toy#216) /v1/completions with the ChatML text as "prompt"
#             gives the chat reply and the chat prompt token count
#   stopstr   (toy#216) a "stop" string cuts the reply where it begins with
#             finish_reason "stop", streamed or not; one that never occurs
#             changes nothing
#   errors    (toy#216) empty messages / empty prompt are HTTP 400 with an
#             error object, and chat finish_reason is the backend's
#             ("length" at max_tokens)
#   idsonly   a GGUF with no tokenizer: text "", chat 501 (unchanged)
#   qwen      (TOY_GATE_QWEN_GGUF=<file>) a real chat model stops at its own
#             end-of-turn token before max_tokens
#
#   ruby prep/serve_chat_gate.rb [--device cuda]   # exit 0 pass, 1 fail, 2 unusable
#
# The base SmolLM2 fixture never emits an end-of-turn token by itself, which
# is why `stop` uses STOP_IDS and the real-EOS leg is the optional one.

require "net/http"
require "json"
require "socket"
require "timeout"
require "tmpdir"
require "open3"

ROOT    = File.expand_path("..", __dir__)
DEVICE  = (i = ARGV.index("--device")) ? ARGV[i + 1].to_s : "cpu"
TARGET  = DEVICE == "cuda" ? "libexec/toy-serve-cuda" : "libexec/toy-serve"
RUNNER  = ENV["TOY_GATE_RUNNER"] || File.join(ROOT, TARGET)
TOKENIZE = File.join(ROOT, "prep", "smokes", "smoke_tokenize")
TOK_GGUF = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
IDS_GGUF = File.join(ROOT, "data", "smollm2-135m-f32.gguf")   # no tokenizer
QWEN     = ENV["TOY_GATE_QWEN_GGUF"].to_s
LABEL    = DEVICE == "cuda" ? "serve-chat-cuda" : "serve-chat"

unless %w[cpu cuda].include?(DEVICE)
  warn "serve_chat_gate: --device must be cpu or cuda"
  exit 2
end
[TOK_GGUF, IDS_GGUF].each do |f|
  next if File.file?(f)
  warn "serve_chat_gate: missing fixture #{f} (never skipped silently)"
  exit 2
end
targets = ["prep/smokes/smoke_tokenize"]
targets << TARGET unless ENV["TOY_GATE_RUNNER"]
targets.each do |t|
  out, st = Open3.capture2e("make", "-C", ROOT, t)
  next if st.success?
  warn "serve_chat_gate: `make #{t}` failed:\n#{out.lines.last(15).join}"
  exit 2
end

# The runner's stdout is block-buffered into a file; line-buffer it when
# stdbuf is there so the `fault` leg can read the server's log line.
STDBUF_BIN = %w[/usr/bin/stdbuf /bin/stdbuf /opt/homebrew/bin/gstdbuf].find { |b| File.executable?(b) }
STDBUF = STDBUF_BIN ? [STDBUF_BIN, "-oL"].freeze : [].freeze

# ---- server lifecycle (one at a time; always torn down) --------------------

@pid = nil
@logf = nil

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
ensure
  File.delete(@logf) if @logf && File.file?(@logf)
end

at_exit { stop_server }
%w[INT TERM].each { |s| trap(s) { stop_server; exit 130 } }

# Boots the runner on `gguf`; returns the port, or raises with the log tail.
def start_server(gguf, extra_env = {})
  stop_server
  sock = TCPServer.new("127.0.0.1", 0)
  port = sock.addr[1]
  sock.close
  @logf = File.join(Dir.tmpdir, "serve_chat_gate_#{Process.pid}_#{port}.log")
  env = { "MODEL_PATH" => gguf, "MODEL_NAME" => "gate", "PORT" => port.to_s }.merge(extra_env)
  # One descriptor for both streams, and no inherited stdin (tep#283).
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
      r = Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 5) { |h| h.get("/health") }
      if r.code == "200"
        if DEVICE == "cuda" && !File.read(@logf).include?("ggml_cuda_init: found")
          raise "the CUDA runner did not initialise a CUDA device"
        end
        return port
      end
    rescue Errno::ECONNREFUSED, Errno::ECONNRESET, EOFError, Timeout::Error, SocketError,
           Net::OpenTimeout, Net::ReadTimeout
    end
    sleep 0.25
  end
  raise "server never became ready"
end

def post(port, path, payload, timeout = 120)
  Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: timeout) do |h|
    req = Net::HTTP::Post.new(path, "Content-Type" => "application/json")
    req.body = JSON.dump(payload)
    h.request(req)
  end
end

# Ids of `text` from the standalone encoder (not the server).
def encode(gguf, text, plain = false)
  env = { "GGUF" => gguf, "TEXT" => text }
  env["PLAIN"] = "1" if plain
  out, st = Open3.capture2e(env, TOKENIZE)
  raise "smoke_tokenize failed: #{out.lines.last}" unless st.success?
  out[/^ids: (.*)$/, 1].to_s.split.map(&:to_i)
end

def chatml(messages)
  messages.map { |m| "<|im_start|>#{m["role"]}\n#{m["content"]}<|im_end|>\n" }.join + "<|im_start|>assistant\n"
end

failures = []
ok = ->(leg, what) { puts "  ok: #{leg} — #{what}" }

# A prompt whose next tokens have a clear margin (see docs/models.md on
# quantized near-ties; this fixture is F32 anyway).
PROMPT = [2020, 719, 253, 655, 28, 665].freeze

begin
  port = start_server(TOK_GGUF)

  # ---- text ----
  r = post(port, "/v1/completions", "model" => "gate", "prompt" => PROMPT, "max_tokens" => 10)
  c = r.code == "200" ? JSON.parse(r.body)["choices"][0] : {}
  ids, text = c["ids"].to_a, c["text"].to_s
  if r.code != "200"
    failures << "text: /v1/completions returned #{r.code}"
  elsif ids.length != 10 || text.empty?
    failures << "text: expected 10 ids and a decoded text, got #{ids.length} ids and #{text.inspect}"
  elsif encode(TOK_GGUF, text, true) != ids
    failures << "text: the returned text does not re-encode to the returned ids (#{text.inspect})"
  elsif c["finish_reason"] != "length"
    failures << "text: finish_reason is #{c["finish_reason"].inspect}, expected \"length\""
  else
    ok.call("text", "ids and text agree (#{text.inspect})")
  end
  plain_ids = ids

  # ---- chat ----
  msgs = [{ "role" => "system", "content" => "Be brief." },
          { "role" => "user", "content" => "What is 2+2?" }]
  r = post(port, "/v1/chat/completions", "model" => "gate", "messages" => msgs, "max_tokens" => 12)
  j = r.code == "200" ? JSON.parse(r.body) : {}
  want_prompt = encode(TOK_GGUF, chatml(msgs)).length
  chat_content = j.dig("choices", 0, "message", "content").to_s
  if r.code != "200"
    failures << "chat: /v1/chat/completions returned #{r.code}: #{r.body.to_s[0, 160]}"
  elsif chat_content.empty? || j.dig("choices", 0, "message", "role") != "assistant"
    failures << "chat: no assistant content in #{r.body.to_s[0, 200]}"
  elsif j.dig("usage", "prompt_tokens") != want_prompt
    failures << "chat: prompt_tokens #{j.dig("usage", "prompt_tokens")}, the ChatML text is #{want_prompt} ids"
  elsif j.dig("usage", "completion_tokens") != 12
    failures << "chat: completion_tokens #{j.dig("usage", "completion_tokens")}, expected 12"
  else
    ok.call("chat", "#{want_prompt} prompt tokens (specials as single ids), content #{chat_content.inspect[0, 50]}")
  end

  # ---- nonascii ----
  umsgs = [{ "role" => "user", "content" => "Répète: 日本語 ☃ é" }]
  begin
    r = post(port, "/v1/chat/completions", { "model" => "gate", "messages" => umsgs, "max_tokens" => 6 }, 30)
    j = JSON.parse(r.body)
    want = encode(TOK_GGUF, chatml(umsgs)).length
    if r.code != "200"
      failures << "nonascii: returned #{r.code}"
    elsif j.dig("usage", "prompt_tokens") != want
      failures << "nonascii: prompt_tokens #{j.dig("usage", "prompt_tokens")}, expected #{want}"
    elsif !j.dig("choices", 0, "message", "content").to_s.valid_encoding?
      failures << "nonascii: reply content is not valid UTF-8"
    else
      ok.call("nonascii", "non-ASCII message answered, #{want} prompt tokens")
    end
  rescue Net::ReadTimeout
    failures << "nonascii: request with a non-ASCII body stalled (tep reads the body by character count before 0.11.9)"
    port = start_server(TOK_GGUF)
  end

  # ---- escapes ----
  # The same body with every non-ASCII character written as a \\uXXXX
  # escape (what Python's json.dumps sends by default; 😀 is a surrogate
  # pair) must be the same request.
  emsgs = [{ "role" => "user", "content" => "Répète: été cœur 日本語 ☃ 😀" }]
  ebody = { "model" => "gate", "messages" => emsgs, "max_tokens" => 6 }
  want = encode(TOK_GGUF, chatml(emsgs)).length
  got = [JSON.generate(ebody), JSON.generate(ebody, ascii_only: true)].map do |raw|
    r = Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 30) do |h|
      req = Net::HTTP::Post.new("/v1/chat/completions", "Content-Type" => "application/json")
      req.body = raw
      h.request(req)
    end
    j = r.code == "200" ? JSON.parse(r.body) : {}
    [j.dig("usage", "prompt_tokens"), j.dig("choices", 0, "message", "content")]
  end
  if JSON.generate(ebody, ascii_only: true) !~ /\\ud83d\\ude00/i
    failures << "escapes: the escaped body does not contain a surrogate pair (gate bug)"
  elsif got[0][0] != want
    failures << "escapes: raw body gave #{got[0][0].inspect} prompt tokens, the ChatML text is #{want}"
  elsif got[1] != got[0]
    failures << "escapes: the \\u-escaped body gave #{got[1][0].inspect} prompt tokens and #{got[1][1].inspect}, " \
                "the raw body #{got[0][0]} and #{got[0][1].inspect}"
  else
    ok.call("escapes", "a \\u-escaped body is the same request as raw UTF-8 (#{want} prompt tokens)")
  end

  # ---- logprobs (toy#216) ----
  chat_req = { "model" => "gate", "messages" => msgs, "max_tokens" => 12 }
  r = post(port, "/v1/chat/completions", chat_req.merge("logprobs" => true, "top_logprobs" => 5))
  j = r.code == "200" ? JSON.parse(r.body) : {}
  lp = j.dig("choices", 0, "logprobs", "content").to_a
  e = []
  e << "HTTP #{r.code}" unless r.code == "200"
  e << "content #{j.dig("choices", 0, "message", "content").inspect} differs from the reply without logprobs" \
    unless j.dig("choices", 0, "message", "content") == chat_content
  e << "#{lp.length} entries for 12 tokens" unless lp.length == 12
  if lp.length == 12
    e << "the entries' bytes do not spell the content" \
      unless lp.flat_map { |t| t["bytes"] }.pack("C*").force_encoding("UTF-8") == chat_content
    lp.each_with_index do |t, i|
      alts = t["top_logprobs"].to_a
      vals = alts.map { |a| a["logprob"] }
      e << "token #{i}: #{alts.length} alternatives, expected 5" unless alts.length == 5
      e << "token #{i}: the token picked (id #{t["id"]}) is not the first alternative" \
        unless alts.first && alts.first["id"] == t["id"] && alts.first["logprob"] == t["logprob"]
      e << "token #{i}: alternatives not sorted or not <= 0: #{vals.inspect}" \
        unless vals == vals.sort.reverse && vals.all? { |v| v.is_a?(Numeric) && v <= 0 }
      mass = vals.sum { |v| Math.exp(v) }
      e << "token #{i}: alternatives' probabilities sum to #{mass.round(4)}" unless mass > 0.01 && mass <= 1.0 + 1e-9
    end
  end
  r0 = post(port, "/v1/chat/completions", chat_req)
  e << "the plain reply carries a logprobs field" if r0.body.include?("\"logprobs\"")
  if e.empty?
    ok.call("logprobs", "12 tokens, each with its log-probability and 5 sorted alternatives (first token p=#{Math.exp(lp[0]["logprob"]).round(3)})")
  else
    e.first(4).each { |m| failures << "logprobs: #{m}" }
  end

  # ---- prefix (toy#216) ----
  if lp.length == 12
    head = lp.first(4).flat_map { |t| t["bytes"] }.pack("C*").force_encoding("UTF-8")
    r = post(port, "/v1/chat/completions",
             chat_req.merge("messages" => msgs + [{ "role" => "assistant", "content" => head }], "max_tokens" => 8))
    j = r.code == "200" ? JSON.parse(r.body) : {}
    rest = j.dig("choices", 0, "message", "content")
    if r.code != "200"
      failures << "prefix: HTTP #{r.code}: #{r.body.to_s[0, 160]}"
    elsif head + rest.to_s != chat_content
      failures << "prefix: #{head.inspect} continued with #{rest.inspect}, expected the rest of #{chat_content.inspect}"
    elsif j.dig("usage", "prompt_tokens") != want_prompt + 4
      failures << "prefix: prompt_tokens #{j.dig("usage", "prompt_tokens")}, expected #{want_prompt + 4} (the prompt plus the 4-token prefix)"
    else
      ok.call("prefix", "a partial assistant message #{head.inspect} is continued with the rest of the reply")
    end
  else
    failures << "prefix: not run (no logprobs to take a token boundary from)"
  end

  # ---- textprompt (toy#216) ----
  r = post(port, "/v1/completions", "model" => "gate", "prompt" => chatml(msgs), "max_tokens" => 12)
  j = r.code == "200" ? JSON.parse(r.body) : {}
  if r.code != "200"
    failures << "textprompt: HTTP #{r.code}: #{r.body.to_s[0, 160]}"
  elsif j.dig("choices", 0, "text") != chat_content
    failures << "textprompt: text #{j.dig("choices", 0, "text").inspect}, the chat reply is #{chat_content.inspect}"
  elsif j.dig("usage", "prompt_tokens") != want_prompt
    failures << "textprompt: prompt_tokens #{j.dig("usage", "prompt_tokens")}, expected #{want_prompt}"
  else
    ok.call("textprompt", "a text prompt on /v1/completions gives the chat reply (#{want_prompt} prompt tokens)")
  end

  # ---- stopstr (toy#216) ----
  if lp.length == 12
    # A stop string that begins inside the reply: tokens 6..8 of it.
    stop_s = lp[6, 3].flat_map { |t| t["bytes"] }.pack("C*").force_encoding("UTF-8")
    at = chat_content.index(stop_s)
    want_cut = chat_content[0, at]
    e = []
    r = post(port, "/v1/chat/completions", chat_req.merge("stop" => ["\u0000never", stop_s]))
    j = r.code == "200" ? JSON.parse(r.body) : {}
    e << "content #{j.dig("choices", 0, "message", "content").inspect} with stop #{stop_s.inspect}, expected #{want_cut.inspect}" \
      unless j.dig("choices", 0, "message", "content") == want_cut
    e << "finish_reason #{j.dig("choices", 0, "finish_reason").inspect}, expected \"stop\"" unless j.dig("choices", 0, "finish_reason") == "stop"
    e << "completion_tokens #{j.dig("usage", "completion_tokens")}, expected fewer than 12" unless j.dig("usage", "completion_tokens").to_i.between?(1, 11)
    r = post(port, "/v1/chat/completions", chat_req.merge("stop" => "no such text \u2603"))
    j = r.code == "200" ? JSON.parse(r.body) : {}
    e << "a stop string that never occurs changed the reply" \
      unless j.dig("choices", 0, "message", "content") == chat_content && j.dig("choices", 0, "finish_reason") == "length"
    sbuf = +""
    Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 120) do |h|
      req = Net::HTTP::Post.new("/v1/chat/completions", "Content-Type" => "application/json")
      req.body = JSON.dump(chat_req.merge("stream" => true, "stop" => stop_s))
      h.request(req) { |resp| resp.read_body { |chunk| sbuf << chunk } }
    end
    sgot = sbuf.split("\n\n").map { |ev| ev.sub(/\Adata: /, "") }.reject { |d| d.strip.empty? || d.strip == "[DONE]" }
               .map { |d| JSON.parse(d).dig("choices", 0, "delta", "content").to_s }.join
    e << "streamed with the stop string: #{sgot.inspect}, expected #{want_cut.inspect}" unless sgot == want_cut
    if at.nil? || at == 0
      failures << "stopstr: could not pick a stop string from #{chat_content.inspect}"
    elsif e.empty?
      ok.call("stopstr", "stop #{stop_s.inspect} cuts the reply to #{want_cut.inspect}, streamed too; an absent one changes nothing")
    else
      e.each { |m| failures << "stopstr: #{m}" }
    end
  else
    failures << "stopstr: not run (no logprobs to take a token boundary from)"
  end

  # ---- errors (toy#216) ----
  e = []
  r = post(port, "/v1/chat/completions", "model" => "gate", "messages" => [])
  e << "empty messages: HTTP #{r.code}" unless r.code == "400" && (JSON.parse(r.body).dig("error", "type") rescue nil) == "invalid_request_error"
  r = post(port, "/v1/completions", "model" => "gate", "prompt" => [])
  e << "empty prompt: HTTP #{r.code}" unless r.code == "400" && (JSON.parse(r.body).dig("error", "type") rescue nil) == "invalid_request_error"
  e << "chat finish_reason at max_tokens is #{JSON.parse(r0.body).dig("choices", 0, "finish_reason").inspect}, expected \"length\"" \
    unless JSON.parse(r0.body).dig("choices", 0, "finish_reason") == "length"
  r = post(port, "/v1/chat/completions", chat_req)
  e << "the server did not answer normally after the error replies" \
    unless r.code == "200" && JSON.parse(r.body).dig("choices", 0, "message", "content") == chat_content
  if e.empty?
    ok.call("errors", "empty messages and an empty prompt are HTTP 400 with an error object; chat reports \"length\" at max_tokens")
  else
    e.each { |m| failures << "errors: #{m}" }
  end

  # ---- stream ----
  frames = []
  done = false
  Net::HTTP.start("127.0.0.1", port, open_timeout: 2, read_timeout: 120) do |h|
    req = Net::HTTP::Post.new("/v1/chat/completions", "Content-Type" => "application/json")
    req.body = JSON.dump("model" => "gate", "messages" => msgs, "max_tokens" => 12, "stream" => true)
    buf = +""
    h.request(req) { |resp| resp.read_body { |chunk| buf << chunk } }
    buf.split("\n\n").each do |ev|
      data = ev.sub(/\Adata: /, "")
      if data.strip == "[DONE]"
        done = true
      elsif !data.strip.empty?
        frames << JSON.parse(data)
      end
    end
  end
  streamed = frames.map { |f| f.dig("choices", 0, "delta", "content").to_s }.join
  if !done
    failures << "stream: no [DONE]"
  elsif frames.empty? || frames.first.dig("choices", 0, "delta", "role") != "assistant"
    failures << "stream: no assistant role prelude"
  elsif streamed != chat_content
    failures << "stream: streamed content #{streamed.inspect} differs from #{chat_content.inspect}"
  elsif frames.last.dig("choices", 0, "finish_reason").nil?
    failures << "stream: last frame carries no finish_reason"
  else
    ok.call("stream", "#{frames.length} frames, content equals the non-streamed reply")
  end

  # ---- stop ----
  # Stop on the 4th generated id; `cut` is where it FIRST occurs.
  stop_id = plain_ids[3]
  cut     = plain_ids.index(stop_id)
  port = start_server(TOK_GGUF, "STOP_IDS" => stop_id.to_s)
  r = post(port, "/v1/completions", "model" => "gate", "prompt" => PROMPT, "max_tokens" => 10)
  c = r.code == "200" ? JSON.parse(r.body)["choices"][0] : {}
  got = c["ids"].to_a
  if cut.nil? || cut == 0
    failures << "stop: could not pick a stop id from #{plain_ids.inspect}"
  elsif c["finish_reason"] != "stop"
    failures << "stop: finish_reason #{c["finish_reason"].inspect} with STOP_IDS=#{stop_id}, expected \"stop\""
  elsif got != plain_ids[0, cut]
    failures << "stop: ids #{got.inspect}, expected #{plain_ids[0, cut].inspect} (cut before #{stop_id})"
  else
    ok.call("stop", "ends before id #{stop_id} after #{cut} of 10 tokens, finish_reason \"stop\"")
  end

  # ---- empty (toy#214) ----
  # A reply of zero tokens: the first generated id is a stop id. Decoding
  # nothing used to raise FrozenError and take the whole server down.
  port = start_server(TOK_GGUF, "STOP_IDS" => plain_ids[0].to_s)
  r = post(port, "/v1/completions", "model" => "gate", "prompt" => PROMPT, "max_tokens" => 10)
  c = r.code == "200" ? JSON.parse(r.body)["choices"][0] : {}
  e = []
  e << "completion: HTTP #{r.code}" unless r.code == "200"
  e << "completion: ids #{c["ids"].inspect} text #{c["text"].inspect} finish #{c["finish_reason"].inspect}, " \
       "expected [] \"\" \"stop\"" unless c["ids"].to_a == [] && c["text"].to_s == "" && c["finish_reason"] == "stop"
  first_chat = post(port, "/v1/chat/completions", "model" => "gate", "messages" => msgs, "max_tokens" => 1)
  e << "chat after an empty reply: HTTP #{first_chat.code}" unless first_chat.code == "200"
  if e.empty?
    ok.call("empty", "a zero-token reply is answered (ids [], text \"\", \"stop\") and the server keeps serving")
  else
    e.each { |m| failures << "empty: #{m}" }
  end

  # ---- fault (toy#214) ----
  # An exception inside one request becomes an HTTP 500 with an error
  # object (toy#216; it was a 200 with finish_reason "error"), is logged,
  # and the next request gets the same answer as on a fresh server.
  port = start_server(TOK_GGUF, "SERVE_TEST_FAULT_AT" => PROMPT.length.to_s)
  r = post(port, "/v1/completions", "model" => "gate", "prompt" => PROMPT, "max_tokens" => 10)
  err = (JSON.parse(r.body)["error"] rescue nil) || {}
  f = []
  f << "faulted request: HTTP #{r.code}, error #{err.inspect}; expected 500 with type \"server_error\" " \
       "and the exception's message" unless r.code == "500" && err["type"] == "server_error" &&
                                            err["message"].to_s.include?("injected test fault")
  if STDBUF_BIN && !File.read(@logf).include?("request failed: injected test fault")
    f << "no 'request failed' line in the server log"
  end
  r2 = post(port, "/v1/chat/completions", "model" => "gate", "messages" => msgs, "max_tokens" => 12)
  after = r2.code == "200" ? JSON.parse(r2.body).dig("choices", 0, "message", "content") : nil
  f << "the request after the fault gave #{after.inspect}, expected #{chat_content.inspect}" unless after == chat_content
  if f.empty?
    ok.call("fault", "a request that raises is answered with HTTP 500 and an error object, logged, and the next request is unaffected")
  else
    f.each { |m| failures << "fault: #{m}" }
  end

  # ---- idsonly + long ----
  port = start_server(IDS_GGUF)
  r = post(port, "/v1/completions", "model" => "gate", "prompt" => PROMPT, "max_tokens" => 300)
  c = r.code == "200" ? JSON.parse(r.body)["choices"][0] : {}
  rc = post(port, "/v1/chat/completions", "model" => "gate", "messages" => msgs, "max_tokens" => 4)
  if c["ids"].to_a.length != 300
    failures << "long: max_tokens 300 gave #{c["ids"].to_a.length} tokens"
  else
    ok.call("long", "max_tokens 300 gives 300 tokens")
  end
  if c["text"] != "" || c["finish_reason"] != "length"
    failures << "idsonly: a tokenizer-less GGUF returned text #{c["text"].inspect} / #{c["finish_reason"].inspect}"
  elsif c["ids"].to_a[0, 10] != plain_ids
    failures << "idsonly: ids differ from the same weights served with a tokenizer"
  elsif rc.code != "501"
    failures << "idsonly: chat on a tokenizer-less GGUF returned #{rc.code}, expected 501"
  else
    ok.call("idsonly", "no tokenizer: same ids, text \"\", chat 501")
  end

  # ---- qwen (optional) ----
  if QWEN.empty?
    puts "  not run: qwen — set TOY_GATE_QWEN_GGUF=<chat model.gguf> for the real end-of-turn leg"
  elsif !File.file?(QWEN)
    failures << "qwen: TOY_GATE_QWEN_GGUF=#{QWEN} does not exist"
  else
    port = start_server(QWEN, "MAX_T" => "512")
    qm = [{ "role" => "system", "content" => "You answer in one short sentence." },
          { "role" => "user", "content" => "What is 2+2?" }]
    r = post(port, "/v1/chat/completions", { "model" => "gate", "messages" => qm, "max_tokens" => 120 }, 600)
    j = r.code == "200" ? JSON.parse(r.body) : {}
    n = j.dig("usage", "completion_tokens").to_i
    content = j.dig("choices", 0, "message", "content").to_s
    if r.code != "200"
      failures << "qwen: returned #{r.code}"
    elsif n <= 0 || n >= 120
      failures << "qwen: #{n} completion tokens of 120 — did not stop at end of turn"
    elsif content.include?("<|im_end|>") || content.include?("<|im_start|>")
      failures << "qwen: control tokens leaked into the content: #{content.inspect[0, 120]}"
    else
      ok.call("qwen", "stopped at end of turn after #{n} tokens: #{content.inspect[0, 60]}")
    end
  end
rescue StandardError => e
  failures << "gate aborted: #{e.message.lines.first.to_s.strip} #{e.message.lines[1..].to_a.join.strip[0, 400]}"
ensure
  stop_server
end

if failures.empty?
  puts "GATE PASS [#{LABEL}]: toy serve takes chat messages and returns text, on #{DEVICE}"
  exit 0
end
failures.each { |m| puts "  FAIL: #{m}" }
puts "GATE FAIL [#{LABEL}]: #{failures.length} failure(s)"
exit 1
