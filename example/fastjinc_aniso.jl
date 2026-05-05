
using BandlimitedOperators, StaticArrays, Bessels

pts1 = [x + SA[10.0, 10.0] for x in rand(SVector{2,Float64}, 1000).*5.0]
pts2 = [x + SA[10.0, 10.0] for x in rand(SVector{2,Float64}, 1100).*0.1]

# The isotropic jinc function:
function iso_jinc(x::SVector{2,Float64}, y::SVector{2,Float64})
  nx = 2*pi*norm(x-y)
  iszero(nx) && return 0.5
  Bessels.besselj1(nx)/nx
end
iso_jinc_ft(ω) = inv(2*pi)

# The anisotropic version:
const A = @SMatrix [1.0 0.1
                    0.1 1.0]
const U = cholesky(A).U
aniso_jinc(x, y) = iso_jinc(U'\x, U'\y)
trueM = [aniso_jinc(x, y) for x in pts1, y in pts2]

# The fast equivalent:
fastM = FastBandlimited(pts1, pts2, iso_jinc_ft, 1.0; 
                        polar=true, warp=x->U'\x)

# forward apply:
x   = randn(length(pts2))
buf = zeros(length(pts1))
mul!(buf, fastM, x)
@show maximum(abs, trueM*x - buf)

# adjoint apply:
x   = randn(length(pts1))
buf = zeros(length(pts2))
mul!(buf, Adjoint(fastM), x)
@show maximum(abs, trueM'*x - buf)

