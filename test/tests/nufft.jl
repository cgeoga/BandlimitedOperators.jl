
for D in 1:3

  pts   = rand(SVector{D,Float64}, 20)
  fqs   = rand(SVector{D,Float64}, 30)

  F1    = BandlimitedOperators.nudftmatrix(fqs, pts, -1)

  F2    = BandlimitedOperators.NUFFT3(fqs.*(2*pi), pts, false)

  F2m = reduce(hcat, map(eachindex(pts)) do j
                 z    = zeros(ComplexF64, length(fqs))
                 c    = zeros(ComplexF64, length(pts))
                 c[j] = 1.0 + 0.0im
                 mul!(z, F2, c)
               end)

  @test maximum(abs, real(F1 - F2m)) < 1e-13
  @test maximum(abs, imag(F1 - F2m)) < 1e-13

end

