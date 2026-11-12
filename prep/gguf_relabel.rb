#!/usr/bin/env ruby
# prep/gguf_relabel.rb — rewrite a GGUF's METADATA the way a different
# converter would have labelled it, leaving every tensor byte untouched.
#
#   ruby prep/gguf_relabel.rb IN.gguf OUT.gguf --prefix qwen2 [--drop KEY]...
#                             [--q8-last TENSOR]
#
# Renames every "<old>.*" architecture key to "<prefix>.*" (old = the
# file's general.architecture), sets general.architecture = <prefix>, and
# drops any --drop keys. Used by prep/llamacpp_gguf_gate.rb (toy#203) to
# manufacture, from a small toy-native fixture, exactly the file shape
# llama.cpp's convert_hf_to_gguf.py emits — "qwen2.*" keys, no
# toy.ggml_native — without committing a second model or depending on
# llama.cpp being installed.
#
# --q8-last TENSOR is the one option that DOES touch tensor bytes: it
# moves that (F32) tensor to the end of the file and re-encodes it as
# Q8_0, so the file ends on the last byte of a quantized tensor. Every
# other tensor is copied verbatim. Used by prep/q8_tail_gate.rb (toy#205)
# to manufacture the shape that aborted on CUDA — a quantized tensor
# whose row width is not a multiple of 512 sitting last in the file,
# where ggml-cuda's row padding runs past EOF — from a small F32 fixture.
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

# IEEE half-precision bits for a non-negative Float (round to nearest).
def half_bits(f)
  return 0 if f <= 0.0
  m, e = Math.frexp(f)            # f = m * 2**e, 0.5 <= m < 1
  e -= 1                          # f = (2m) * 2**e, 1 <= 2m < 2
  return 0x7C00 if e > 15
  if e < -14                      # subnormal
    return (f / 2.0**-24).round
  end
  frac = ((m * 2.0 - 1.0) * 1024).round
  if frac == 1024
    frac = 0
    e += 1
    return 0x7C00 if e > 15
  end
  ((e + 15) << 10) | frac
end

# ggml's Q8_0: per 32 values, a half-float scale d = max|x| / 127 then
# 32 int8 = round(x / d).
def q8_0(f32_bytes)
  out = +"".b
  vals = f32_bytes.unpack("e*")
  abort "gguf_relabel: --q8-last needs a multiple of 32 values" unless (vals.length % 32).zero?
  vals.each_slice(32) do |blk|
    amax = blk.map(&:abs).max
    d = amax / 127.0
    id = d.zero? ? 0.0 : 1.0 / d
    out << [half_bits(d)].pack("S<") << blk.map { |x| (x * id).round.clamp(-127, 127) }.pack("c*")
  end
  out
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
q8_last = nil
i = 2
while i < ARGV.length
  case ARGV[i]
  when "--prefix" then prefix = ARGV[i + 1]; i += 2
  when "--drop"   then drops << ARGV[i + 1]; i += 2
  when "--q8-last" then q8_last = ARGV[i + 1]; i += 2
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
  tensors = []   # [name, dims, type, offset]
  n_tensors.times do
    name = rd_str(io)
    nd = io.read(4).unpack1("L<")
    dims = io.read(8 * nd).unpack("Q<*")
    type, off = io.read(12).unpack("L<Q<")
    tensors << [name, dims, type, off]
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

  # --q8-last: new tensor order (target last), new offsets, and the
  # target's Q8_0 bytes. ggml requires tensors to be contiguous in info
  # order, so the others are re-packed in their original order; each
  # keeps its own bytes, only its offset moves.
  plan = nil   # [[src_offset, nbytes] ...] for the verbatim tensors
  tail = nil   # Q8_0 bytes of the target
  if q8_last
    idx = tensors.index { |t| t[0] == q8_last } or abort "gguf_relabel: --q8-last: no tensor #{q8_last}"
    name, dims, type, off = tensors[idx]
    abort "gguf_relabel: --q8-last: #{name} is type #{type}, need F32" unless type == 0
    data_len = io.size - data_start
    ends = tensors.map { |t| t[3] }.sort + [data_len]
    extent = ->(o) { ends.find { |e| e > o } - o }
    rest = tensors.reject.with_index { |_, j| j == idx }
    plan = []
    cur = 0
    new_infos = +"".b
    rest.each do |n, d, ty, o|
      len = extent.call(o)
      plan << [o, len]
      new_infos << wr_str(n) << [d.length].pack("L<") << d.pack("Q<*") << [ty, cur].pack("L<Q<")
      cur = (cur + len + align - 1) / align * align
    end
    tail = q8_0(io.pread(4 * dims.inject(1) { |a, b| a * b }, data_start + off))
    new_infos << wr_str(name) << [dims.length].pack("L<") << dims.pack("Q<*") << [8, cur].pack("L<Q<")
    infos = new_infos
  end
  File.open(dst, "wb") do |o|
    o.write("GGUF")
    o.write([version, n_tensors, out.length].pack("L<Q<Q<"))
    out.each { |k, t, raw| o.write(wr_str(k)); o.write([t].pack("L<")); o.write(raw) }
    o.write(infos)
    pad = (align - o.pos % align) % align
    o.write("\0" * pad)
    if plan
      base = o.pos
      plan.each do |src_off, len|
        io.seek(data_start + src_off)
        IO.copy_stream(io, o, len)
        o.write("\0" * ((align - (o.pos - base) % align) % align))
      end
      o.write(tail)
    else
      io.seek(data_start)
      IO.copy_stream(io, o)
    end
  end
end
