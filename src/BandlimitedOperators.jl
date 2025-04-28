
module BandlimitedOperators
 
  using LinearAlgebra, FastGaussQuadrature, FINUFFT, StaticArraysCore

  export FastBandlimited

  include("nufft.jl")
  include("operator.jl")

end 

