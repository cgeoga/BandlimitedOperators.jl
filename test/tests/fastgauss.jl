
offsets = (0.0, -3.0, 25.0)
alphas  = (5.0, 100.0, 1000.0)

pts1    = rand(StableRNG(1234), 1000).*10
pts2    = rand(StableRNG(1234), 1200).*10

@testset "1D" begin
  for (oj, alpha) in Iterators.product(offsets, alphas)

    gauss_kernel(t) = exp(-alpha*abs2(t))
    gauss_ft(omega) = sqrt(pi/alpha)*exp(-abs2(pi*omega)/alpha)
    bandwidth       = sqrt(-(alpha/(pi^2))*log(sqrt(alpha/pi)*1e-18))

    M     = [gauss_kernel(xj-xk) for xj in pts1, xk in pts2]
    fastM = FastBandlimited(pts1, pts2, gauss_ft, bandwidth)

    x    = randn(StableRNG(123), length(pts2))
    x2   = randn(StableRNG(123), length(pts1))
    buf  = zeros(length(pts1))
    buf2 = zeros(length(pts2))

    mul!(buf,  fastM,  x)
    mul!(buf2, fastM', x2)

    # in-place:
    @test maximum(abs, buf  - M*x)   < 1e-11 
    @test maximum(abs, buf2 - M'*x2) < 1e-11 

    # out-of-place:
    @test maximum(abs, fastM*x   - M*x)   < 1e-11 
    @test maximum(abs, fastM'*x2 - M'*x2) < 1e-11 

  end
end

