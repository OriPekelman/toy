# Probe landmine #9 — Hash[missing_key] semantics under Spinel.
#
# Original symptom (2026-05-22, T1.2 SmolLM2 tokenizer):
#   r = h[k]
#   if r != nil && r < best_rank  # missing-key returned 0, took the
#                                  # wrong branch every time
#
# The MRI/idiomatic pattern `v = h[k]; if v == nil` was broken because
# Spinel's int-valued hash returned 0 for missing keys instead of nil.
# The fix has been to guard with has_key? first.
#
# Recent Spinel commit 9d28a38 ("h[k] &&=/||= on int-valued hashes uses
# has_key, not _get return") is the closest in spirit. Test whether
# the plain `h[k] == nil` check now distinguishes missing from
# zero-valued entries.

h = {}
h["alpha"] = 0       # legitimate zero-value entry
h["beta"]  = 7
h["gamma"] = 99

# Missing key
mv = h["zeta"]
puts "missing 'zeta':   mv=" + mv.inspect + "  is_nil=" + (mv == nil).to_s

# Legitimate zero-value
av = h["alpha"]
puts "legit zero 'alpha': av=" + av.inspect + "  is_nil=" + (av == nil).to_s

# A non-zero entry
bv = h["beta"]
puts "non-zero 'beta': bv=" + bv.inspect + "  is_nil=" + (bv == nil).to_s

# Idiomatic gate the landmine documented
if mv != nil
  puts "FAIL: mv==nil should be true (missing key) but came back as non-nil"
else
  puts "PROBE-PASS-MISSING: missing-key correctly distinguishes from zero"
end

if av != nil
  puts "PROBE-PASS-ZERO: legitimate zero-value correctly non-nil"
else
  puts "FAIL: av (legit zero) appeared as nil — distinguishability broken"
end
