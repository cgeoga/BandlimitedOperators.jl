
using Test, LinearAlgebra, StableRNGs, BandlimitedOperators, StaticArrays, Bessels

@testset "NUFFT" begin
  let scope_dummy = 0
    include("./tests/nufft.jl")
  end
end

@testset "Fast Gauss" begin
  let scope_dummy = 0
    include("./tests/fastgauss.jl")
  end
end

@testset "Fast sinc" begin
  let scope_dummy = 0
    include("./tests/fastsinc.jl")
  end
end

@testset "Fast sinc squared" begin
  let scope_dummy = 0
    include("./tests/fastsincsquared.jl")
  end
end

@testset "Polar coordinates" begin
  let scope_dummy = 0
    include("./tests/polar.jl")
  end
end

