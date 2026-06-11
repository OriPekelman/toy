#!/usr/bin/env ruby
# Auto-generate docs/coverage.md from the source tree.
#
# Canonical sources:
#   vendor/ggml/include/ggml.h          : enum ggml_op (all ops in the library)
#   tinynn/tinynn_ggml.{h,c}            : our C-level wrappers
#   lib/toy/ffi/tinynn.rb / lib/toy/ffi/tinynn_cuda.rb  : Ruby FFI bindings on each backend
#
# For each GGML_OP_X:
#   1. Derive the natural C name (ggml_<x_lower>).
#   2. Grep tinynn_ggml.c for calls to ggml_<x_lower>(. The enclosing
#      function (heuristic: walk back to the previous `^void *tnn_…(` or
#      `^int tnn_…(` declaration) is the tnn_ wrapper.
#   3. Look up that wrapper in lib/toy/ffi/tinynn.rb and lib/toy/ffi/tinynn_cuda.rb
#      to check FFI binding presence on each backend.
#   4. Backward: if GGML_OP_X_BACK is in the enum, check whether
#      ggml_<x>_back appears in any tnn_ wrapper body.
#
# Output is single-file, deterministic, diff-friendly markdown.
#
#   ruby prep/gen_coverage.rb              # writes docs/coverage.md
#   ruby prep/gen_coverage.rb --check      # exits non-zero if stale

require "set"

ROOT = File.expand_path("..", __dir__)

# ---------------------------------------------------------------------------
# 1. ggml ops — parse enum ggml_op
# ---------------------------------------------------------------------------

def parse_ggml_ops(path)
  text  = File.read(path)
  m     = text.match(/enum\s+ggml_op\s*\{(.+?)\};/m)
  raise "couldn't find enum ggml_op in #{path}" unless m
  ops   = []
  m[1].each_line do |line|
    line.strip!
    next if line.empty? || line.start_with?("//", "/*")
    name = line[/GGML_OP_[A-Z_0-9]+/]
    next if name.nil?
    next if name == "GGML_OP_NONE" || name == "GGML_OP_COUNT"
    ops << name unless ops.include?(name)
  end
  ops
end

# ---------------------------------------------------------------------------
# 2. tnn wrappers — scan tinynn_ggml.c for tnn_* impls and which ggml_* they
#    call. Returns a hash: tnn_name => Set[ggml_func_called]
# ---------------------------------------------------------------------------

def parse_tnn_wrappers(path)
  text     = File.read(path)
  wrappers = {}
  current  = nil
  text.each_line do |line|
    # Match a function definition opening at column 0. Cover the
    # return-type variants in the C source: `void *`, `int`, `long`,
    # `size_t`, `double`, `float`, `const char *`. The pointer star
    # may or may not have a space before the identifier.
    if line =~ /^(?:void|int|long|size_t|double|float|const\s+char)\s*\*?\s*(tnn_[a-z_0-9]+)\s*\(/
      current = $1
      wrappers[current] ||= Set.new
    elsif current && line =~ /^\}/
      current = nil
    elsif current
      line.scan(/\bggml_[a-z_0-9]+\b/).each do |sym|
        # Filter out struct names / type tokens — keep only function-call
        # forms (followed by `(`).  This regex peeks one char.
        wrappers[current] << sym if line[/#{Regexp.escape(sym)}\s*\(/]
      end
    end
  end
  wrappers
end

# ---------------------------------------------------------------------------
# 3. FFI binding presence
# ---------------------------------------------------------------------------

def parse_ffi_funcs(path)
  set = Set.new
  File.foreach(path) do |line|
    if line =~ /ffi_func\s+:(tnn_[a-z_0-9]+)/
      set << $1
    end
  end
  set
end

# ---------------------------------------------------------------------------
# 4. Map a GGML_OP_X to its C-function "stem"
# ---------------------------------------------------------------------------

def op_stem(op_name)
  # GGML_OP_MUL_MAT  -> "mul_mat"
  # GGML_OP_RMS_NORM -> "rms_norm"
  op_name.sub(/^GGML_OP_/, "").downcase
end

# Some ggml C functions don't match the natural stem (or we have aliases).
# Express the exceptions here, in C-function-name form. Each entry is a
# list of acceptable C function names for that op (any one being called
# from a tnn_ wrapper counts as "bound").
SPECIAL_STEMS = {
  "ROPE"             => %w[ggml_rope ggml_rope_ext ggml_rope_multi],
  "ROPE_BACK"        => %w[ggml_rope_back ggml_rope_ext_back],
  "VIEW"             => %w[ggml_view_1d ggml_view_2d ggml_view_3d ggml_view_4d],
  "RESHAPE"          => %w[ggml_reshape_1d ggml_reshape_2d ggml_reshape_3d ggml_reshape_4d],
  "PERMUTE"          => %w[ggml_permute],
  "TRANSPOSE"        => %w[ggml_transpose],
  "CONT"             => %w[ggml_cont ggml_cont_1d ggml_cont_2d ggml_cont_3d],
  "CPY"              => %w[ggml_cpy],
  "GET_ROWS"         => %w[ggml_get_rows],
  "GET_ROWS_BACK"    => %w[ggml_get_rows_back],
  "SET_ROWS"         => %w[ggml_set_rows],
  "ADD_ID"           => %w[ggml_add_id],
  "MUL_MAT"          => %w[ggml_mul_mat],
  "MUL_MAT_ID"       => %w[ggml_mul_mat_id],
  "OUT_PROD"         => %w[ggml_out_prod],
  "DIAG_MASK_INF"    => %w[ggml_diag_mask_inf],
  "SOFT_MAX"         => %w[ggml_soft_max ggml_soft_max_ext],
  "SOFT_MAX_BACK"    => %w[ggml_soft_max_back ggml_soft_max_ext_back],
  "SILU_BACK"        => %w[ggml_silu_back],
  "RMS_NORM"         => %w[ggml_rms_norm],
  "RMS_NORM_BACK"    => %w[ggml_rms_norm_back],
  "FLASH_ATTN_EXT"   => %w[ggml_flash_attn_ext],
  "FLASH_ATTN_BACK"  => %w[ggml_flash_attn_back],
  # UNARY is a meta-op; the unary fn (silu/gelu/relu) is its own ggml_<u>.
  "UNARY"            => %w[ggml_unary],
  "GLU"              => %w[ggml_glu],
  # Internal-only enum cases — no public ggml_<x>() constructor.
  "MAP_CUSTOM1"      => [],
  "MAP_CUSTOM2"      => [],
  "MAP_CUSTOM3"      => [],
  "CUSTOM"           => [],
}

def candidate_c_funcs(op_name)
  return SPECIAL_STEMS[op_name.sub(/^GGML_OP_/, "")] if SPECIAL_STEMS.key?(op_name.sub(/^GGML_OP_/, ""))
  ["ggml_" + op_stem(op_name)]
end

# Wrapper-name overrides for ops where our tnn_<wrapper> doesn't
# follow the tnn_<ggml_stem> naming convention. Without this, the
# matching heuristic falls back to `via` (incidental composition)
# even when a dedicated wrapper exists under a different name.
PRIMARY_WRAPPER = {
  "GGML_OP_MUL_MAT" => "tnn_matmul",   # not tnn_mul_mat
  "GGML_OP_SOFT_MAX" => "tnn_softmax", # not tnn_soft_max
}

# Unary sub-ops (SILU, GELU, RELU, etc.) live under GGML_OP_UNARY in
# the dispatch enum but each has its own ggml_<x>() C constructor and
# its own enum value in `enum ggml_unary_op`. Surface them as virtual
# rows so the coverage table reflects their bound status individually.
UNARY_OPS = %w[GELU SILU RELU GELU_QUICK TANH ELU SIGMOID HARDSIGMOID HARDSWISH ABS SGN NEG STEP EXP]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ggml_h          = File.join(ROOT, "vendor/ggml/include/ggml.h")
tinynn_c        = File.join(ROOT, "tinynn/tinynn_ggml.c")
tinynn_rb       = File.join(ROOT, "lib/toy/ffi/tinynn.rb")
tinynn_cuda_rb  = File.join(ROOT, "lib/toy/ffi/tinynn_cuda.rb")
tinynn_metal_rb = File.join(ROOT, "lib/toy/ffi/tinynn_metal.rb")
out_path        = File.join(ROOT, "docs/coverage.md")

ops       = parse_ggml_ops(ggml_h)
wrappers  = parse_tnn_wrappers(tinynn_c)
cpu_ffi   = parse_ffi_funcs(tinynn_rb)
cuda_ffi  = parse_ffi_funcs(tinynn_cuda_rb)
# Metal mirror is intentionally a thin subset (issue #2). Whatever
# tnn_* bindings it declares are reported here; ops absent from the
# mirror but present in the C wrapper are not "drift" — they're just
# not yet wired into the Ruby Metal surface. See the Metal column note
# under "Parity drift" below.
metal_ffi = File.exist?(tinynn_metal_rb) ? parse_ffi_funcs(tinynn_metal_rb) : Set.new

# Reverse index: ggml_func -> [tnn_wrappers that call it]
ggml_to_tnn = Hash.new { |h, k| h[k] = [] }
wrappers.each do |tnn_name, ggml_funcs|
  ggml_funcs.each { |g| ggml_to_tnn[g] << tnn_name }
end

def row_for(op_name, ggml_to_tnn, cpu_ffi, cuda_ffi, metal_ffi, ops_set)
  cands = candidate_c_funcs(op_name)
  if cands.empty?
    return {
      op: op_name, wrapper: nil,
      cpu: "n/a", cuda: "n/a", metal: "n/a", backward: "—",
      note: "internal dispatch op (no public ggml_<x>() constructor)",
    }
  end

  # Find the *primary* tnn_ wrapper for this op: a wrapper named after
  # the ggml stem (so ggml_view_2d → tnn_view_2d). Composing wrappers
  # that incidentally call ggml_cont / ggml_view inside don't count as
  # bindings for those ops — they're tracked by the wrappers they
  # belong to (e.g. tnn_transpose composes ggml_cont but VIEW/CONT
  # binding status should reflect whether there's a *dedicated*
  # wrapper). Variant suffixes like _1d/_2d are accepted (any one
  # variant being bound counts as the op being bound).
  hit_tnn         = nil
  incidental_only = false
  override        = PRIMARY_WRAPPER[op_name]
  cands.each do |c|
    next unless ggml_to_tnn.key?(c) && !ggml_to_tnn[c].empty?
    # First try the explicit override (e.g. ggml_mul_mat → tnn_matmul).
    if override && ggml_to_tnn[c].include?(override)
      hit_tnn = override
      break
    end
    expected_tnn = c.sub(/^ggml_/, "tnn_")
    if ggml_to_tnn[c].include?(expected_tnn)
      hit_tnn = expected_tnn
      break
    end
    # Remember that at least one wrapper composes this op but no
    # primary wrapper exists.
    incidental_only = true
  end

  if hit_tnn.nil?
    note = incidental_only ?
      "composed inside other wrappers (no dedicated tnn_)" :
      "no tnn_ wrapper"
    return {
      op: op_name, wrapper: nil,
      cpu:   incidental_only ? "via" : "missing",
      cuda:  incidental_only ? "via" : "missing",
      metal: incidental_only ? "via" : "missing",
      backward: "—",
      note: note,
    }
  end

  cpu_present   = cpu_ffi.include?(hit_tnn)
  cuda_present  = cuda_ffi.include?(hit_tnn)
  metal_present = metal_ffi.include?(hit_tnn)

  # Backward column semantics:
  #   "yes"      — explicit tnn_<x>_back wrapper exists (we call it
  #                from a manually-built backward sub-graph)
  #   "autodiff" — GGML_OP_<X>_BACK is in the enum, ggml's
  #                ggml_build_backward_expand emits it automatically.
  #                No wrapper needed for training to work.
  #   "—"        — op has no _BACK enum case at all.
  back_status = "—"
  back_op     = op_name + "_BACK"
  if ops_set.include?(back_op)
    back_cands = candidate_c_funcs(back_op)
    back_hit   = back_cands.any? { |c| ggml_to_tnn.key?(c) && !ggml_to_tnn[c].empty? }
    back_status = back_hit ? "yes" : "autodiff"
  end

  {
    op: op_name, wrapper: hit_tnn,
    cpu:   cpu_present   ? "yes" : "no",
    cuda:  cuda_present  ? "yes" : "no",
    metal: metal_present ? "yes" : "no",
    backward: back_status, note: "",
  }
end

ops_set = Set.new(ops)
rows = ops.flat_map do |op|
  # Replace the meta-row GGML_OP_UNARY with one virtual row per
  # ggml_unary_op (GELU / SILU / RELU / …). Each has its own C
  # constructor; they're individually bound or missing.
  if op == "GGML_OP_UNARY"
    UNARY_OPS.map do |u|
      virt = "GGML_OP_" + u
      r = row_for(virt, ggml_to_tnn, cpu_ffi, cuda_ffi, metal_ffi, ops_set)
      r[:note] = "unary sub-op (dispatches via GGML_OP_UNARY)" if r[:note] == "" || r[:note] == "no tnn_ wrapper"
      r
    end
  else
    [row_for(op, ggml_to_tnn, cpu_ffi, cuda_ffi, metal_ffi, ops_set)]
  end
end

# Filter: drop the *_BACK enum cases from the main table — they're rolled
# into the parent row's "Backward" column.
back_set = ops.select { |o| o =~ /_BACK$/ }.to_set
main_rows = rows.reject { |r| back_set.include?(r[:op]) }

# Counts for the summary.
bound_count    = main_rows.count { |r| r[:cpu] == "yes" }
via_count      = main_rows.count { |r| r[:cpu] == "via" }
missing_count  = main_rows.count { |r| r[:cpu] == "missing" }
notapp_count   = main_rows.count { |r| r[:cpu] == "n/a" }
parity_drift       = main_rows.count { |r| r[:cpu] == "yes" && r[:cuda]  != "yes" }
parity_drift_metal = main_rows.count { |r| r[:cpu] == "yes" && r[:metal] != "yes" }
back_count         = main_rows.count { |r| r[:backward] == "yes" }
back_auto          = main_rows.count { |r| r[:backward] == "autodiff" }
back_possible      = main_rows.count { |r| r[:backward] != "—" }

# ---------------------------------------------------------------------------
# Emit markdown
# ---------------------------------------------------------------------------

require "stringio"
md = StringIO.new
md << <<~MD
  # ggml op coverage

  Auto-generated by `prep/gen_coverage.rb` from the source tree. Don't
  hand-edit — run `make coverage` to regenerate.

  This is the canonical "what we support" matrix. One row per ggml op
  (`enum ggml_op` in `vendor/ggml/include/ggml.h`), with columns for the
  CPU + CUDA + Metal FFI bindings and backward support. Models built on
  top of these ops are tracked separately in `docs/models-verified.md`.

  ## Summary

  - **#{bound_count} of #{main_rows.size}** ops have a dedicated `tnn_*` wrapper bound on the CPU side (#{(100.0 * bound_count / main_rows.size).round}%).
  - **#{via_count}** more are composed inside other wrappers (status `via`) — usable in graphs we build but no standalone entry point.
  - **#{parity_drift}** bound ops are CPU-only (no CUDA binding) — see "Parity drift" below.
  - **#{parity_drift_metal}** bound ops are not bound on Metal — the Metal mirror (`lib/toy/ffi/tinynn_metal.rb`) is an intentionally thin smoke surface today, see "Metal mirror" below.
  - **#{back_count}** ops with a `*_BACK` enum case have an explicit backward wrapper; **#{back_auto}** more are emitted automatically by ggml's autodiff (`ggml_build_backward_expand`) without needing a wrapper.
  - **#{missing_count}** ops have no wrapper at all (and aren't composed by another).
  - **#{notapp_count}** ops are dispatch-internal (no public `ggml_<x>()` constructor — e.g. `UNARY`, `MAP_CUSTOM1`).

  Backend caveats not captured by the table:

  - **`MUL_MAT_ID` × Q4_K / Q6_K source weights**: ggml's mul_mat_id
    kernel produces garbage on K-quantized expert tensors. Q8_0 / F16 /
    F32 sources work. Tracked in `docs/notes/mul_mat_id_quant.md` once
    we write it; for now see commit `42b8609` (M2.3) for the
    repro/workaround.
  - **`FLASH_ATTN_BACK`**: ggml's backward kernel calls `GGML_ABORT` at
    runtime (see `vendor/ggml/src/ggml.c::ggml_flash_attn_back`). The
    op is "missing" for us, but it's also broken upstream — wait for
    upstream fix before binding.

  ## Op coverage

  | Op | tnn wrapper | CPU | CUDA | Metal | Back | Note |
  |----|-------------|:---:|:----:|:-----:|:----:|------|
MD

main_rows.each do |r|
  md << "| `#{r[:op]}` | "
  md << (r[:wrapper] ? "`#{r[:wrapper]}`" : "—") << " | "
  md << r[:cpu]      << " | "
  md << r[:cuda]     << " | "
  md << r[:metal]    << " | "
  md << r[:backward] << " | "
  md << r[:note]     << " |\n"
end

md << <<~MD

  ## Parity drift

  Ops where CPU has an FFI binding but CUDA does not. Drift > 0 is a
  regression — run `make coverage-check` in CI to catch it.

MD
drift_rows = main_rows.select { |r| r[:cpu] == "yes" && r[:cuda] != "yes" }
if drift_rows.empty?
  md << "_None — full CPU/CUDA parity on bound ops._\n"
else
  md << "| Op | tnn wrapper |\n|----|-------------|\n"
  drift_rows.each { |r| md << "| `#{r[:op]}` | `#{r[:wrapper]}` |\n" }
end

md << <<~MD

  ## Metal mirror

  The Metal FFI surface (`lib/toy/ffi/tinynn_metal.rb`) is intentionally a thin
  smoke binding today — just enough to prove `ggml_backend_metal_init`
  + a forward matmul on Apple Silicon. Op count below is *not* a
  regression signal; it's the to-do list for whoever wires real model
  inference on Metal (issue #2 follow-up).

MD
metal_drift = main_rows.select { |r| r[:cpu] == "yes" && r[:metal] != "yes" }
md << "_Metal-bound ops_: **#{main_rows.count { |r| r[:metal] == 'yes' }}** of #{main_rows.count { |r| r[:cpu] == 'yes' }} CPU-bound.\n"
md << "_Not yet on Metal_: #{metal_drift.size}.\n"

md << <<~MD

  ## Tnn surface not directly tied to a ggml op

  Wrappers in `tinynn/tinynn_ggml.c` that don't map 1:1 to a single
  `GGML_OP_*` — session lifecycle, GGUF reader, upload/download
  helpers, trace primitive, etc. Listed here for completeness; not
  covered by the table above.

MD

# Find tnn_ wrappers that don't appear in any GGML_OP row. Backward
# wrappers (tnn_*_back) live in the parent row's "Back" column; collect
# them here so they're not listed under "infrastructure".
tnn_in_table = main_rows.map { |r| r[:wrapper] }.compact.to_set
# Include explicit backward wrappers — anything matching tnn_*_back
# that calls a ggml_*_back function.
wrappers.each do |tnn_name, ggml_funcs|
  tnn_in_table << tnn_name if ggml_funcs.any? { |g| g.end_with?("_back") }
end
all_tnn      = wrappers.keys.to_set
infrastructure = (all_tnn - tnn_in_table).sort
md << "<details>\n<summary>#{infrastructure.size} infrastructure / helper wrappers</summary>\n\n"
infrastructure.each { |t| md << "- `#{t}`\n" }
md << "\n</details>\n"

# ---------------------------------------------------------------------------
# Write or check
# ---------------------------------------------------------------------------

if ARGV.include?("--check")
  existing = File.exist?(out_path) ? File.read(out_path) : ""
  if existing != md.string
    warn "coverage.md is stale; run `make coverage` to regenerate."
    exit 1
  end
  puts "coverage.md is up to date."
else
  FileUtils.mkdir_p(File.dirname(out_path)) if !File.exist?(File.dirname(out_path))
  File.write(out_path, md.string)
  puts "wrote #{out_path} (#{main_rows.size} ops, #{bound_count} bound, #{parity_drift} parity drift)"
end
