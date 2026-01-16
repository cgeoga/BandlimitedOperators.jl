
for D in 1:3

  pts   = rand(SVector{D,Float64}, 20)
  fqs   = rand(SVector{D,Float64}, 30)

  F1    = BandlimitedOperators.nudftmatrix(fqs, pts, -1; with_2pi=true)
  F2    = BandlimitedOperators.NUFFT3(fqs, pts, -1; with_2pi=true)

  template1 = [complex(Float64(i+j))/(20*30) for i in 1:20, j in 1:30]
  template2 = [complex(Float64(i+j))/(20*30) for i in 1:30, j in 1:20]

  @test maximum(abs, real(F1*template1  - F2*template1)) < 1e-13
  @test maximum(abs, imag(F1'*template2 - F2'*template2)) < 1e-13

end

