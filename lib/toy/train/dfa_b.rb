# lib/toy/train/dfa_b.rb — fixed random feedback matrices for DFA
# (toy#109 P2). One module, three distribution families, three scale
# rules — the experiment axes from the design doc §4b:
#
#   dist:  0 = gaussian (Box–Muller)   1 = uniform [-1,1]   2 = rademacher ±1
#   scale: 0 = inv_sqrt_fan (1/√fan)   1 = glorot (√(6/(fan_in+fan_out)))
#          2 = fixed (sigma passed through)
#
# Values are INT codes, not symbols — the policy travels as flat data
# (Spinel-safe, RecipeOptions-ready, no recompiles per config). Sampling
# is xorshift64 seeded per weight: pass a seed derived from
# (run_seed, stable_weight_id) so re-policying one segment never
# reshuffles another's B. Deterministic across runs and platforms that
# share IEEE doubles (the byte-repro gates pin this).
#
# fan_out here = the projected-INTO dimension (the weight's output dim),
# fan_in = the error dimension (vocab/d_out of e) — B maps e → δ_W.

module Toy
  module Train
    module DfaB
      DIST_GAUSSIAN   = 0
      DIST_UNIFORM    = 1
      DIST_RADEMACHER = 2
      SCALE_INV_SQRT_FAN = 0
      SCALE_GLOROT       = 1
      SCALE_FIXED        = 2

      # Resolve the scale rule to a sigma. fan_in = error dim, fan_out =
      # target dim. `fixed_sigma` is only read for SCALE_FIXED.
      def self.sigma_for(scale_code, fan_in, fan_out, fixed_sigma)
        if scale_code == SCALE_GLOROT
          return Math.sqrt(6.0 / (fan_in.to_f + fan_out.to_f))
        end
        if scale_code == SCALE_FIXED
          return fixed_sigma
        end
        1.0 / Math.sqrt(fan_in.to_f)
      end

      # Fill an n-element array from the coded distribution at `sigma`,
      # xorshift64 stream from `seed`.
      def self.fill(n, seed, dist_code, sigma)
        a = [0.0]; a.pop
        s = (seed * 2654435761) % 9007199254740881
        if s <= 0; s = seed + 7; end
        i = 0
        while i < n
          s ^= (s << 13) & 0xFFFFFFFFFFFF; s ^= (s >> 7); s ^= (s << 17) & 0xFFFFFFFFFFFF
          u1 = ((s % 1000003).to_f + 1.0) / 1000004.0
          if dist_code == DIST_UNIFORM
            a.push((u1 * 2.0 - 1.0) * sigma)
          elsif dist_code == DIST_RADEMACHER
            if u1 < 0.5
              a.push(0.0 - sigma)
            else
              a.push(sigma)
            end
          else
            s ^= (s << 13) & 0xFFFFFFFFFFFF; s ^= (s >> 7); s ^= (s << 17) & 0xFFFFFFFFFFFF
            u2 = ((s % 1000003).to_f + 1.0) / 1000004.0
            g = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(6.283185307179586 * u2)
            a.push(g * sigma)
          end
          i = i + 1
        end
        a
      end
    end
  end
end
