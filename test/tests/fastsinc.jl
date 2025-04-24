
offsets    = (0.0, -3.0, 25.0)
bandwidths = (0.1, 1.0, 10.0, 45.2)

# centered on [-1/2, 1/2].
pts   = sort(rand(StableRNG(123), 1000)) .- 0.5
pts2D = rand(StableRNG(123), SVector{2,Float64}, 1000)

# random input vector:
v   = randn(StableRNG(124), length(pts))

# 1D:
for (oj, bk) in Iterators.product(offsets, bandwidths)
  ptsj = pts .+ oj
  fs   = FastBandlimited(ptsj, ptsj, x->inv(2*bk), bk)
  M    = [sinc(2*bk*(xj-xk)) for xj in ptsj, xk in ptsj]
  tmp  = similar(v)
  mul!(tmp, fs, v)
  @test maximum(abs, tmp-M*v) < 1e-11
end

#2D:
sinc2d(x) = sinc(x[1])*sinc(x[2])
for (oj, bk) in Iterators.product(offsets, bandwidths)
  ptsj = [x + SA[oj, oj] for x in pts2D]
  fs   = FastBandlimited(pts2D, pts2D, x->inv((2*bk)^2), bk)
  M    = [sinc2d(2*bk*(xj-xk)) for xj in ptsj, xk in ptsj]
  tmp  = similar(v)
  mul!(tmp, fs, v)
  @test maximum(abs, tmp-M*v) < 1e-11
end

