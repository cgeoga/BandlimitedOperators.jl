
using BandlimitedOperators

# 1D:

pts1 = sort(rand(1000)).*3
pts2 = sort(rand(1000)).*3
x    = randn(length(pts1))

# The matrix we want to approximate, which corresponds to a kernel matrix whose
# kernel function's Fourier transform is supported on [-18.1, 18.1].
M = [sinc(2*18.1*(xj-xk)) for xj in pts1, xk in pts2]

# We approximate it with a FastBandlimited object. Note the Fourier transform of
# this kernel function is precisely ω ↦ 2*χ(|ω| ≤ 18.1)/18.1, so that is the
# third argument here.
fastM = FastBandlimited(pts1, pts2, x->inv(2*18.1), 18.1)
buf   = zeros(length(x))
mul!(buf, fastM, x)

@show maximum(abs, M*x - buf)

# adjoint apply:
mul!(buf, Adjoint(fastM), x)
@show maximum(abs, M'*x - buf)


# 2D:

using StaticArrays

pts1 = rand(SVector{2,Float64}, 1000).*3
pts2 = rand(SVector{2,Float64}, 1000).*3

sinc2d(t) = sinc(t[1])*sinc(t[2])
M = [sinc2d(2*3.3*(xj-xk)) for xj in pts1, xk in pts2]

fastM = FastBandlimited(pts1, pts2, x->inv((2*3.3)^2), 3.3)
mul!(buf, fastM, x)

@show maximum(abs, M*x - buf)

# adjoint apply:
mul!(buf, Adjoint(fastM), x)
@show maximum(abs, M'*x - buf)

