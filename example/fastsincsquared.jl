
using BandlimitedOperators


triangle(x, bw) = max(0.0, 1-abs(x)/bw)

# 1D:

pts1 = sort(rand(1000)).*3
pts2 = sort(rand(1000)).*3
x    = randn(length(pts1))

# The matrix we want to approximate, which corresponds to a kernel matrix whose
# kernel function's Fourier transform is supported on [-18.1, 18.1].
M = [sinc(18.1*(xj-xk))^2 for xj in pts1, xk in pts2]

# We approximate it with a FastBandlimited object. Note the Fourier transform of
# this kernel function is precisely ω ↦ triangle(w, 18.1)/18.1, so that is the third
# argument here.
#
# Unlike in the 1D fastsinc.jl demo, though, now we need to specify that the
# Fourier transform of our kernel function has a rough point at the origin. If
# you don't do this, the quadrature rule being used internally won't be accurate
# and you will get very few correct digits.
fastM = FastBandlimited(pts1, pts2, x->triangle(x, 18.1)/18.1, 18.1; quadn_add=10, roughpoints=(0.0,))
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

sincsquared2d(t) = (sinc(t[1])*sinc(t[2]))^2
M = [sincsquared2d(3.3*(xj-xk)) for xj in pts1, xk in pts2]

fastM = FastBandlimited(pts1, pts2, x->triangle(x[1], 3.3)*triangle(x[2], 3.3)/(3.3^2), 3.3;
                        quadn_add=30, roughpoints=(SA[0.0, 0.0],))
mul!(buf, fastM, x)

@show maximum(abs, M*x - buf)

# adjoint apply:
mul!(buf, Adjoint(fastM), x)
@show maximum(abs, M'*x - buf)

