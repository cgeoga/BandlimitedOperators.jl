
module BandlimitedOperators
 
  using LinearAlgebra, FastGaussQuadrature, FINUFFT, StaticArraysCore

  export NUFFT3, FastBandlimited

  include("nufft.jl")
  include("operator.jl")

end 

