
using BandlimitedOperators, StaticArrays, Bessels

pts1 = [x + SA[10.0, 10.0] for x in rand(SVector{2,Float64}, 1000).*5.0]
pts2 = [x + SA[10.0, 10.0] for x in rand(SVector{2,Float64}, 1100).*0.1]

function jinc(x::SVector{2,Float64}, y::SVector{2,Float64}, bw)
  nx = 2*pi*bw*norm(x-y)
  iszero(nx) && return 0.5
  Bessels.besselj1(nx)/nx
end

trueM = [jinc(x, y, 1.0) for x in pts1, y in pts2]
fastM = FastBandlimited(pts1, pts2, x->inv(2*pi), 1.0; polar=true)

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

