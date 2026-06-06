# lib/toy/io/toy_git.rb — Toy::Git, git provenance read from .git/HEAD.
#
# WHY THIS EXISTS. Every tep-free runner and training example that emits a
# `run_start` event stamped its `git:{sha,branch}` provenance by inlining the
# SAME ~25-line .git/HEAD parse — copy-pasted across 9 files. This is that block,
# once. Pure Ruby (no FFI), so it compiles under Spinel and needs no mirror.
#
# Behaviour (unchanged from the inlined block): reads .git/HEAD; if it's a
# `ref: refs/heads/<branch>` pointer, the branch is the last path segment and the
# 40-char sha is read from the pointed-at ref file; if HEAD is detached (a raw
# sha), sha = that, branch = "HEAD". Anything missing/short → "unknown". Caller-
# facing default stays "unknown"/"unknown" so a non-repo checkout is non-fatal
# (Tao's acceptance only needs git.sha present).
#
# SPINEL NAMING DISCIPLINE: whole-program inference is keyed partly on method-
# and local-variable NAMES (landmines #12/#16). The reader members and every
# local carry a `gi_` prefix so they can't widen an unrelated `head`/`sha`/`pp`
# elsewhere in a compiled runner.
#
# USAGE (keeps the historical local-variable names so call sites are untouched):
#   gp = Toy::Git.read
#   git_sha    = gp.gi_sha
#   git_branch = gp.gi_branch
module Toy
  class Git
    def initialize(gi_sha, gi_branch)
      @gi_sha    = gi_sha
      @gi_branch = gi_branch
    end

    def gi_sha
      @gi_sha
    end

    def gi_branch
      @gi_branch
    end

    # Read provenance from ./.git/HEAD. Returns a Toy::Git (gi_sha/gi_branch).
    def self.read
      gi_s = "unknown"
      gi_b = "unknown"
      if File.exist?(".git/HEAD")
        gi_head = File.read(".git/HEAD")
        if gi_head.length > 0 && gi_head[gi_head.length - 1...gi_head.length] == "\n"
          gi_head = gi_head[0...gi_head.length - 1]
        end
        if gi_head.length > 5 && gi_head[0...5] == "ref: "
          gi_ref_rel = gi_head[5...gi_head.length]
          gi_pp = gi_ref_rel.split("/")
          if gi_pp.length >= 3
            gi_b = gi_pp[gi_pp.length - 1]
          end
          gi_ref_path = ".git/" + gi_ref_rel
          if File.exist?(gi_ref_path)
            gi_sha_raw = File.read(gi_ref_path)
            if gi_sha_raw.length >= 40
              gi_s = gi_sha_raw[0...40]
            end
          end
        else
          if gi_head.length >= 40
            gi_s = gi_head[0...40]
            gi_b = "HEAD"
          end
        end
      end
      Toy::Git.new(gi_s, gi_b)
    end
  end
end
