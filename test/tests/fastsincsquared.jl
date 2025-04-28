
offsets    = (0.0, -3.0, 25.0)
bandwidths = (0.1, 1.4, 10.3)

triangle(x, bw) = max(0.0, 1-abs(x)/bw)

# centered on [-1/2, 1/2].
pts   = sort(rand(StableRNG(123), 1000))
pts2D = rand(StableRNG(123), SVector{2,Float64}, 1000)

# random input vector:
v = randn(StableRNG(124), length(pts))

# 1D:
@testset "1D" begin
  for (oj, bk) in Iterators.product(offsets, bandwidths)
    ptsj = pts .+ oj
    fs   = FastBandlimited(ptsj, ptsj, x->triangle(x, bk)/bk, bk; roughpoints=(0.0,))
    M    = [sinc(bk*(xj-xk))^2 for xj in ptsj, xk in ptsj]
    tmp  = similar(v)
    mul!(tmp, fs, v)
    @test maximum(abs, tmp-M*v) < 1e-11
  end
end

#2D:
sincsqu2d(x) = (sinc(x[1])*sinc(x[2]))^2
@testset "2D" begin
  for (oj, bk) in Iterators.product(offsets, bandwidths)
    ptsj = [x + SA[oj, oj] for x in pts2D]
    fs   = FastBandlimited(pts2D, pts2D, x->(triangle(x[1], bk)*triangle(x[2], bk))/(bk^2), bk; 
                           roughpoints=(SA[0.0, 0.0],))
    M    = [sincsqu2d(bk*(xj-xk)) for xj in ptsj, xk in ptsj]
    tmp  = similar(v)
    mul!(tmp, fs, v)
    @test maximum(abs, tmp-M*v) < 1e-11
  end
end

