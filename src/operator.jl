
struct FastBandlimited{T}
  sym::Bool
  sz::NTuple{2,Int64}
  ft1::NUFFT3
  ft2::NUFFT3
  no::Vector{T}
  op::Vector{Float64}
  buf::Vector{ComplexF64}
end

function glquadrule(n::Int64, a::Float64, b::Float64)
  (no, wt) = gausslegendre(n)
  (bmad2, bpad2) = ((b-a)/2, (b+a)/2)
  @inbounds for j in 1:n
    no[j] = no[j]*bmad2 + bpad2
    wt[j] = wt[j]*bmad2
  end
  (no, wt)
end

# panel version of above
function glquadrule(n::Int64, ab_breakpoints::AbstractVector{Float64})
  bma    = ab_breakpoints[end] - ab_breakpoints[1]
  ivs    = collect(zip(ab_breakpoints, ab_breakpoints[2:end]))
  nquadv = map(ivs) do (aj, bj)
    Int(ceil(n*(bj-aj)/bma ))
  end
  no_wt_v = [glquadrule(nquadv[j], ivs[j][1], ivs[j][2]) for j in eachindex(ivs)]
  (reduce(vcat, getindex.(no_wt_v, 1)), reduce(vcat, getindex.(no_wt_v, 2)))
end

function glquadrule(nv::NTuple{N, Int64}, a::NTuple{N,Float64},
                    b::NTuple{N,Float64}) where{N}
  no_wt_v = [glquadrule(nv[j], a[j], b[j]) for j in 1:N]
  nodes   = vec(SVector{N,Float64}.(Iterators.product(getindex.(no_wt_v, 1)...)))
  weights = vec(prod.(Iterators.product(getindex.(no_wt_v, 2)...)))
  (nodes, weights)
end

function bandlimited_quadrule(s1::Vector{Float64}, s2::Vector{Float64}, 
                              bandlimit::Float64, quadn_add=0, roughpoints=nothing)
  maxs  = max(maximum(abs, s1), maximum(abs, s2))
  quadn = Int(ceil(8*bandlimit*maxs + quadn_add))
  isnothing(roughpoints) && return glquadrule(quadn, -bandlimit, bandlimit)
  all(x->-bandlimit < x < bandlimit, roughpoints) || throw(error("Rough points are outside the bandlimit region."))
  regions = vcat(-bandlimit, sort(unique(roughpoints)), bandlimit)
  glquadrule(quadn, -bandlimit, bandlimit)
end

# TODO (cg 2025/04/18 16:00): extend the panel logic to the 2D case
function bandlimited_quadrule(s1::Vector{SVector{D,Float64}}, 
                              s2::Vector{SVector{D,Float64}}, 
                              bandlimit, quadn_add=0, roughpoints=nothing) where{D}
  bandlimits = (bandlimit isa Float64) ? ntuple(_->bandlimit, D) : bandlimits
  quadnv  = ntuple(D) do j
    maxsj = max(maximum(x->abs(x[j]), s1), maximum(x->abs(x[j]), s2))
    Int(ceil(4*bandlimits[j]*maxsj + quadn_add))
  end
  glquadrule(quadnv, .-bandlimits, bandlimits)
end

function FastBandlimited(s1, s2, fn, bandlimit; quadn_add=50, roughpoints=nothing)
  (no, wt)  = bandlimited_quadrule(s1, s2, bandlimit, quadn_add, roughpoints)
  op  = wt.*fn.(no)
  ft1 = NUFFT3(s1.*(2*pi), no, false)
  ft2 = NUFFT3(no, s2.*(2*pi), true)
  buf = Vector{ComplexF64}(undef, length(no))
  FastBandlimited(s1==s2, (length(s1), length(s2)), ft1, ft2, no, op, buf)
end

LinearAlgebra.ishermitian(fs::FastBandlimited{T}) where{T} = fs.sym
Base.eltype(fs::FastBandlimited{T}) where{T} = Float64
Base.size(fs::FastBandlimited)         = fs.sz
Base.size(fs::FastBandlimited, j::Int) = size(fs)[j]

function LinearAlgebra.mul!(buf, fs::FastBandlimited, v)
  mul!(fs.buf, fs.ft1, complex(v))
  fs.buf .*= fs.op
  tmp = Vector{ComplexF64}(undef, length(v))
  mul!(tmp, fs.ft2, fs.buf)
  buf .= real(tmp)
end

function LinearAlgebra.mul!(buf, afs::Adjoint{Float64, FastBandlimited{T}}, v) where{T}
  fs = afs.parent
  mul!(fs.buf, adjoint(fs.ft2), complex(v))
  fs.buf .*= fs.op
  tmp = Vector{ComplexF64}(undef, length(v))
  mul!(tmp, adjoint(fs.ft1), fs.buf)
  buf .= real(tmp)
end

