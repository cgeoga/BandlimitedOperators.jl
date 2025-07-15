
struct FastBandlimited
  ft1::NUFFT3
  ft2::NUFFT3
  herm::Bool
  op::Vector{ComplexF64}
  buf_s1::Matrix{ComplexF64}
  buf_no::Matrix{ComplexF64}
  buf_s2::Matrix{ComplexF64}
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
  rule = shifted_bandlimited_quadrule(s1, s2, bandlimit, quadn_add, 
                                      roughpoints; polar=polar)
  op   = complex(rule.wt.*fn.(rule.no))
  ft1  = NUFFT3(rule.no, rule._s2.*(2*pi), -1)
  ft2  = NUFFT3(rule.no, rule._s1.*(2*pi), -1)
  buf_s1 = Matrix{ComplexF64}(undef, length(s1), 1)
  buf_no = Matrix{ComplexF64}(undef, length(op), 1)
  buf_s2 = Matrix{ComplexF64}(undef, length(s2), 1)
  FastBandlimited(ft1, ft2, s1==s2, op, buf_s1, buf_no, buf_s2)
end

LinearAlgebra.ishermitian(fs::FastBandlimited) = fs.herm
LinearAlgebra.issymmetric(fs::FastBandlimited) = fs.herm
LinearAlgebra.adjoint(fs::FastBandlimited) = Adjoint{Float64, FastBandlimited}(fs)
Base.eltype(fs::FastBandlimited) = Float64
Base.size(fs::FastBandlimited)   = (size(fs.ft2, 2), size(fs.ft1, 2))
Base.size(fs::FastBandlimited, j::Int) = size(fs)[j]

function LinearAlgebra.mul!(buf, fs::FastBandlimited, v)
  for j in 1:size(v, 2)
    copyto!(fs.buf_s2, view(v, :, j))
    finufft_exec!(fs.ft1.plan, fs.buf_s2, fs.buf_no)
    fs.buf_no .*= fs.op 
    finufft_exec!(fs.ft2.adjplan, fs.buf_no, fs.buf_s1)
    for k in 1:size(buf, 1)
      buf[k,j] = real(fs.buf_s1[k])
    end
  end
  buf
end

function Base.:*(fs::FastBandlimited, v::AbstractVector{Float64})
  out = Array{Float64}(undef, size(fs, 1))
  mul!(out, fs, v)
end

function Base.:*(fs::FastBandlimited, v::AbstractMatrix{Float64})
  out = Array{Float64}(undef, (size(fs, 1), size(v,2)))
  mul!(out, fs, v)
end

function LinearAlgebra.mul!(buf, afs::Adjoint{Float64, FastBandlimited}, v) 
  fs = afs.parent
  for j in 1:size(v, 2)
    copyto!(fs.buf_s1, view(v, :, j))
    finufft_exec!(fs.ft2.plan, fs.buf_s1, fs.buf_no)
    fs.buf_no .*= fs.op 
    finufft_exec!(fs.ft1.adjplan, fs.buf_no, fs.buf_s2)
    for k in 1:size(buf, 1)
      buf[k,j] = real(fs.buf_s2[k])
    end
  end
  buf
end

function Base.:*(afs::Adjoint{Float64, FastBandlimited}, v::AbstractVector{Float64})
  out = Array{Float64}(undef, size(afs, 1))
  mul!(out, afs, v)
end

function Base.:*(afs::Adjoint{Float64, FastBandlimited}, v::AbstractMatrix{Float64})
  out = Array{Float64}(undef, (size(afs, 1), size(v,2)))
  mul!(out, afs, v)
end

