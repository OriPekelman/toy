# Parity: Mat#matmul_t vs TinyNN.matmul_t, Mat#t_matmul vs TinyNN.t_matmul.
# These two cover the transformer's full matmul-variant set:
#   matmul    A * B     (FFN, projections)
#   matmul_t  A * B^T   (attention scores: Q * K^T)
#   t_matmul  A^T * B   (attention V backward, embedding backward chain)

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

# matmul_t: (3,4) and (5,4) -> (3,5)
mata = Mat.new(3, 4)
matb = Mat.new(5, 4)
i = 0
while i < 12
  mata.flat[i] = i.to_f * 0.1 - 0.5
  i = i + 1
end
i = 0
while i < 20
  matb.flat[i] = i.to_f * 0.07 - 0.4
  i = i + 1
end
native1 = mata.matmul_t(matb)
ffi1 = TinyNN.matmul_t(mata, matb)
ok1 = true
n = native1.nrows * native1.ncols
i = 0
while i < n
  d = native1.flat[i] - ffi1.flat[i]
  if d < 0
    d = -d
  end
  if d > 1.0e-4
    ok1 = false
  end
  i = i + 1
end
puts "matmul_t: shape " + native1.nrows.to_s + "x" + native1.ncols.to_s + " native[0]=" + native1.flat[0].to_s + " ffi[0]=" + ffi1.flat[0].to_s + " match=" + ok1.to_s

# t_matmul: (4,3) and (4,5) -> (3,5)
mata2 = Mat.new(4, 3)
matbt = Mat.new(4, 5)
i = 0
while i < 12
  mata2.flat[i] = i.to_f * 0.13 - 0.4
  i = i + 1
end
i = 0
while i < 20
  matbt.flat[i] = i.to_f * 0.09 - 0.3
  i = i + 1
end
native2 = mata2.t_matmul(matbt)
ffi2 = TinyNN.t_matmul(mata2, matbt)
ok2 = true
n = native2.nrows * native2.ncols
i = 0
while i < n
  d = native2.flat[i] - ffi2.flat[i]
  if d < 0
    d = -d
  end
  if d > 1.0e-4
    ok2 = false
  end
  i = i + 1
end
puts "t_matmul: shape " + native2.nrows.to_s + "x" + native2.ncols.to_s + " native[0]=" + native2.flat[0].to_s + " ffi[0]=" + ffi2.flat[0].to_s + " match=" + ok2.to_s
