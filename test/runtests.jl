
using Test, LinearAlgebra, StableRNGs, BandlimitedOperators
using BandlimitedOperators.StaticArrays

@testset "NUFFT" begin
  let scope_dummy = 0
    include("./tests/nufft.jl")
  end
end

@testset "Fast sinc" begin
  let scope_dummy = 0
    include("./tests/fastsinc.jl")
  end
end

