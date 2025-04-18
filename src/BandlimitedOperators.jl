
module BandlimitedOperators
 
  using LinearAlgebra, FastGaussQuadrature, FINUFFT, StaticArrays

  export FastBandlimited

  include("nufft.jl")
  include("operator.jl")

end 

