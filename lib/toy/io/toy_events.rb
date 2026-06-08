# lib/toy/io/toy_events.rb — Toy::Events, shared toy/v1 run_start provenance.
#
# WHY THIS EXISTS. Every runner/example run_start event carried the SAME three
# provenance sub-objects in the SAME order — host{name,os,arch}, backend{kind},
# git{sha,branch} — built inline (~12 lines + a Toy::Git.read) in 11 places. This
# folds that into one call:
#
#   rs = Toy::Json.new
#   rs.j_str("kind", "run_start"); rs.j_str("schema", "toy/v1")
#   rs.j_num("t", now); rs.j_str("started_at", iso); rs.j_str("run_id", rid)
#   rs.j_str("phase", "train")
#   Toy::Events.add_provenance(rs, host_name, host_os, host_arch, backend_kind)
# (git{} comes from SpinelKit::Git.read — toy#44.)
#   # ... caller appends model{} / config{} / schedule{} ...
#   TinyNN.tnn_events_emit(rs.j_dump)
#
# The host/backend values are PASSED IN: they come from the backend-specific
# TinyNN / TinyNNCuda / TinyNNMetal module the caller is bound to, which a shared
# helper can't name. git is backend-agnostic (pure file read) so it's read here.
# Key order is preserved exactly → byte-identical events for ASCII-clean values.
#
# Pure Ruby (no FFI) → compiles under Spinel, no mirror. Spinel naming discipline:
# the module method's params carry an `ev_` prefix so they can't widen an
# unrelated host_name/backend_kind elsewhere (landmines #12/#16).
require_relative "toy_json"
# SpinelKit::Git (toy#44) — git provenance, consolidated out of the former
# lib/toy/io/toy_git.rb (identical .git/HEAD reader; plain sha/branch accessors
# now that the Spinel gi_-prefix inference bug is fixed upstream). Vendored at
# build time via `make vendor-tep` (spinel-compat). Required by-path (NOT via
# vendor/spinel/deps) so the tep-free runners stay tep-free — deps.rb pulls tep
# too. toy_events is a runner/example dep, never part of toy.rb's compute
# surface, so this relative reach into vendor/spinel is build-local (same as
# serve.rb's vendor/spinel/deps require).
require_relative "../../../vendor/spinel/spinel_kit/lib/spinel_kit/git"

module Toy
  module Events
    # Append host{}, backend{}, git{} (the canonical run_start provenance) to an
    # in-progress Toy::Json run_start builder, in order. Mutates ev_rs.
    def self.add_provenance(ev_rs, ev_host_name, ev_host_os, ev_host_arch, ev_backend_kind)
      ev_host = Toy::Json.new
      ev_host.j_str("name", ev_host_name)
      ev_host.j_str("os",   ev_host_os)
      ev_host.j_str("arch", ev_host_arch)
      ev_rs.j_obj("host", ev_host)
      ev_backend = Toy::Json.new
      ev_backend.j_str("kind", ev_backend_kind)
      ev_rs.j_obj("backend", ev_backend)
      ev_gp = SpinelKit::Git.read
      ev_git = Toy::Json.new
      ev_git.j_str("sha",    ev_gp.sha)
      ev_git.j_str("branch", ev_gp.branch)
      ev_rs.j_obj("git", ev_git)
    end
  end
end
