
struct FastBandlimited
  ft1::NUFFT3
  ft2::NUFFT3
  herm::Bool
  op::Vector{ComplexF64}
end

function Base.display(fs::FastBandlimited)
  println("FastBandlimited operator with:")
  println("  - size: $(size(fs))")
  println("  - quadrature rule order: $(length(fs.op))")
end

"""
`FastBandlimited(s1, s2, fun, bandlimit;
                quadn_add=default_extra_quad(s1),
                roughpoints=nothing)`

A constructor for a `FastBandlimited` object that represents the action of a matrix
`M` with entries `M[j,k] = g(s1[j] - s2[k])` with `g` satisfying

\$ g(t) = ∫_{[-bandlimit, bandlimit]^{dim}} e^{-2 π i t^T ω} fun(ω) d ω \$

for `t` and `Ω` in R^{dim}. Arguments are:

- `s1::Union{Vector{Float64}, Vector{SVector{D,Float64}}` with `1 <= D <= 3`. 

- `s2::Union{Vector{Float64}, Vector{SVector{D,Float64}}` with `1 <= D <= 3`. 

- `bandlimit::Union{Float64, [iterable]{D,Float64}}`. If you pass in points in
  multiple dimensions and bandlimit is a `Float64`, that will be interpreted to mean
  that the bandlimit in every dimension is the same.

Keyword arguments are:

- `quadn_add=default_extra_quad(s1)`. the number of quadrature nodes beyond
  Nyquist to add in each dimension. So in 1D you can make this 1000 without
  breaking a sweat. But if you make it 1000 in 3D, you're adding 10^9 nodes! 
  So be careful. You hopefully won't have to touch this argument so long as you
  provide roughpoints correctly.

- `roughpoints::Union{Nothing, [iterable]{D,Float64}}=nothing`. This argument 
  is the way that you communicate locations at which `fun` is not smooth. If
  you don't do this, the quadrature rule will be much less accurate. This kwarg
  should just be an iterable of frequencies that match the dimension and type of
  your location. See the example file of the fast sinc squared transform for an
  example of telling `FastBandlimited` that the 2D triangular function is rough 
  at the origin. This is an important arg to get right! And there is no easy way
  for the code to catch it if you forget---you'll just silently get wrong answers.

- `polar::Bool=false`. This arguments indicates whether `fun`, the FT of your
  kernel, is compactly supported on a disk of radius `bandlimit` instead of a
  square with sidelength `rectangle`. **Note**: your function should still
  accept a frequency argument in _cartesian_ coordinates. See the
  `./example/fastjinc.jl` demo for an example. This is only supported for 2D
  kernel matrices and at present supports only disk-supported kernels, not
  general ellipses. But ellipses wouldn't be a huge challenge to add, so if you
  need that functionality please open an issue and we can talk.
"""
function FastBandlimited(s1::Vector, s2::Vector, fn, bandlimit; 
                         quadn_add=default_extra_quad(s1), 
                         roughpoints=nothing, polar=false)
  (_s1, _s2) = internal_shift(s1, s2)
  (no, wt)   = if polar 
    polar_bandlimited_quadrule(_s1, _s2, bandlimit, quadn_add, roughpoints)
  else
    bandlimited_quadrule(_s1, _s2, bandlimit, quadn_add, roughpoints)
  end
  op  = complex(wt.*fn.(no))
  ft1 = NUFFT3(no, _s2.*(2*pi), -1)
  ft2 = NUFFT3(no, _s1.*(2*pi), -1)
  FastBandlimited(ft1, ft2, s1==s2, op)
end

LinearAlgebra.ishermitian(fs::FastBandlimited) = fs.herm
LinearAlgebra.issymmetric(fs::FastBandlimited) = fs.herm
LinearAlgebra.adjoint(fs::FastBandlimited) = Adjoint{Float64, FastBandlimited}(fs)
Base.eltype(fs::FastBandlimited) = Float64
Base.size(fs::FastBandlimited)   = (size(fs.ft2, 2), size(fs.ft1, 2))
Base.size(fs::FastBandlimited, j::Int) = size(fs)[j]

# TODO (cg 2025/06/24 13:36): arguably, this complex converstion that allocates
# is already a violation of the promises of mul!. Making this _completely_
# in-place is clearly still a question.
function LinearAlgebra.mul!(buf, fs::FastBandlimited, v::VecOrMat{Float64})
  cv = complex(v)
  fourier_buf = fs.ft1*cv
  for j in 1:size(fourier_buf, 2)
    buf_colj   = view(fourier_buf, :, j)
    buf_colj .*= fs.op
  end
  tmp  = fs.ft2'*fourier_buf
  buf .= real(tmp)
end

function Base.:*(fs::FastBandlimited, v)
  out = if (v isa Vector{Float64})
    Array{Float64}(undef, size(fs, 1))
  else
    Array{Float64}(undef, (size(fs, 1), size(v,2)))
  end
  mul!(out, fs, v)
end

function LinearAlgebra.mul!(buf, afs::Adjoint{Float64, FastBandlimited}, 
                            v::VecOrMat{Float64}) 
  fs = afs.parent
  cv = complex(v)
  fourier_buf = fs.ft2*cv
  for j in 1:size(fourier_buf, 2)
    buf_colj   = view(fourier_buf, :, j)
    buf_colj .*= fs.op
  end
  tmp = fs.ft1'*fourier_buf
  buf.= real(tmp)
end

function Base.:*(afs::Adjoint{Float64, FastBandlimited}, v)
  out = if (v isa Vector{Float64})
    Array{Float64}(undef, size(afs, 1))
  else
    Array{Float64}(undef, (size(afs, 1), size(v,2)))
  end
  mul!(out, afs, v)
end

