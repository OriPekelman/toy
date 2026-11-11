#!/usr/bin/env ruby
# prep/gguf_relabel.rb — rewrite a GGUF's METADATA the way a different
# converter would have labelled it, leaving every tensor byte untouched.
#
#   ruby prep/gguf_relabel.rb IN.gguf OUT.gguf --prefix qwen2 [--drop KEY]...
#
# Renames every "<old>.*" architecture key to "<prefix>.*" (old = the
# file's general.architecture), sets general.architecture = <prefix>, and
# drops any --drop keys. Used by prep/llamacpp_gguf_gate.rb (toy#203) to
# manufacture, from a small toy-native fixture, exactly the file shape
# llama.cpp's convert_hf_to_gguf.py emits — "qwen2.*" keys, no
# toy.ggml_native — without committing a second model or depending on
# llama.cpp being installed.
#
# Pure CRuby, no gems. GGUF v2/v3, little-endian.

SCALAR = { 0 => 1, 1 => 1, 2 => 2, 3 => 2, 4 => 4, 5 => 4, 6 => 4, 7 => 1,
           10 => 8, 11 => 8, 12 => 8 }.freeze
T_STRING = 8
T_ARRAY  = 9

def rd_str(io)
  n = io.read(8).unpack1("Q<")
  io.read(n)
end

def wr_str(s)
  [s.bytesize].pack("Q<") + s
end

# Returns the RAW bytes of one value of type `t` (re-emitted verbatim).
def rd_value_raw(io, t)
  if t == T_STRING
    wr_str(rd_str(io))
  elsif t == T_ARRAY
    head = io.read(12)
    et, n = head.unpack("L<Q<")
    if et == T_STRING
      body = +""
      n.times { body << wr_str(rd_str(io)) }
      head + body
    else
      sz = SCALAR[et] or abort "gguf_relabel: unsupported array element type #{et}"
      head + io.read(sz * n)
    end
  else
    sz = SCALAR[t] or abort "gguf_relabel: unsupported value type #{t}"
    io.read(sz)
  end
end

src, dst = ARGV[0], ARGV[1]
prefix = nil
drops = []
i = 2
while i < ARGV.length
  case ARGV[i]
  when "--prefix" then prefix = ARGV[i + 1]; i += 2
  when "--drop"   then drops << ARGV[i + 1]; i += 2
  else abort "gguf_relabel: unknown argument #{ARGV[i]}"
  end
end
abort "usage: gguf_relabel.rb IN OUT --prefix NAME [--drop KEY]..." unless src && dst && prefix

File.open(src, "rb") do |io|
  abort "gguf_relabel: #{src} is not a GGUF" unless io.read(4) == "GGUF"
  version, n_tensors, n_kv = io.read(20).unpack("L<Q<Q<")
  kvs = []
  n_kv.times do
    key = rd_str(io)
    t = io.read(4).unpack1("L<")
    kvs << [key, t, rd_value_raw(io, t)]
  end
  infos_start = io.pos
  n_tensors.times do
    rd_str(io)
    nd = io.read(4).unpack1("L<")
    io.read(8 * nd + 12)
  end
  infos = io.pread(io.pos - infos_start, infos_start)

  arch = kvs.find { |k, _, _| k == "general.architecture" }
  abort "gguf_relabel: no general.architecture in #{src}" unless arch
  old = arch[2][8..]
  align_kv = kvs.find { |k, _, _| k == "general.alignment" }
  align = align_kv ? align_kv[2].unpack1("L<") : 32

  out = kvs.reject { |k, _, _| drops.include?(k) }.map do |k, t, raw|
    if k == "general.architecture"
      [k, t, wr_str(prefix)]
    elsif k.start_with?(old + ".")
      [prefix + k[old.length..], t, raw]
    else
      [k, t, raw]
    end
  end

  # Tensor offsets are relative to the (aligned) start of the data
  # section, so a header of a different length only needs re-padding.
  data_start = (io.pos + align - 1) / align * align
  File.open(dst, "wb") do |o|
    o.write("GGUF")
    o.write([version, n_tensors, out.length].pack("L<Q<Q<"))
    out.each { |k, t, raw| o.write(wr_str(k)); o.write([t].pack("L<")); o.write(raw) }
    o.write(infos)
    pad = (align - o.pos % align) % align
    o.write("\0" * pad)
    io.seek(data_start)
    IO.copy_stream(io, o)
  end
end
