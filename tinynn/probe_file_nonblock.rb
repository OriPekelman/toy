# Probe: does the non-block File.open(path, "w") form actually write
# under current Spinel HEAD a03bb49? Landmine #15 said it silently
# no-ops; commit 39438d8 advertises a fix. Verify before removing
# block-form workarounds.

path = "/tmp/probe_file_nonblock.out"

# Non-block write
f = File.open(path, "w")
f.write("hello from non-block File.open\n")
f.write("line two\n")
f.close

puts "wrote " + path

# Read back
raw = File.read(path)
puts "read back " + raw.length.to_s + " bytes:"
puts raw
