
using Test, LinearAlgebra, StableRNGs, BandlimitedOperators, StaticArrays, Bessels

# TODO (cg 2025/07/16 09:42): 
#
# 1. Write tests for kernels where the FT support is a rectangle and not a square.
# 2. Write more tests for the polar case.
# 3. Develop (and write tests for) ellipsoidal support.

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

