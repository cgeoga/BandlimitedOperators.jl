
struct FastBandlimited{T}
  sym::Bool
  sz::NTuple{2,Int64}
  ft1::NUFFT3
  ft2::NUFFT3
  no::Vector{T}
  op::Vector{ComplexF64}
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
                              bandlimit::Float64, quadn_add, roughpoints)
  maxs  = max(maximum(abs, s1), maximum(abs, s2))
  quadn = Int(ceil(8*bandlimit*maxs + quadn_add))
  isnothing(roughpoints) && return glquadrule(quadn, -bandlimit, bandlimit)
  regions = vcat(-bandlimit, sort(unique(roughpoints)), bandlimit)
  glquadrule(quadn, regions)
end

function bandlimited_quadrule(s1::Vector{SVector{D,Float64}}, 
                              s2::Vector{SVector{D,Float64}}, 
                              bandlimit, quadn_add, roughpoints) where{D}
  bandlimits = (bandlimit isa Float64) ? ntuple(_->bandlimit, D) : bandlimits
  quadnv = ntuple(D) do j
    maxsj = max(maximum(x->abs(x[j]), s1), maximum(x->abs(x[j]), s2))
    Int(ceil(8*bandlimits[j]*maxsj + quadn_add))
  end
  no_wt_v = ntuple(D) do j
    breakpointsj = if isnothing(roughpoints)
      [-bandlimits[j], bandlimits[j]]
    else
      splitsj = unique(getindex.(roughpoints, j))
      bp      = sort(vcat(-bandlimits[j], splitsj, bandlimits[j]))
      filter(x-> abs(x) <= bandlimits[j], bp)
    end
    glquadrule(quadnv[j], breakpointsj)
  end
  nodes = getindex.(no_wt_v, 1)
  wts   = getindex.(no_wt_v, 2)
  nodes = vec(SVector{D,Float64}.(Iterators.product(nodes...)))
  wts   = vec(prod.(Iterators.product(wts...)))
  (nodes, wts)
end

function FastBandlimited(s1, s2, fn, bandlimit; quadn_add=10, 
                         roughpoints=nothing)
  (no, wt)  = bandlimited_quadrule(s1, s2, bandlimit, quadn_add, roughpoints)
  op  = complex(wt.*fn.(no))
  ft1 = NUFFT3(no, s2.*(2*pi), false, 1e-15)
  ft2 = NUFFT3(no, s1.*(2*pi), false, 1e-15)
  FastBandlimited(s1==s2, (length(s1), length(s2)), ft1, ft2, no, op)
end

LinearAlgebra.ishermitian(fs::FastBandlimited{T}) where{T} = fs.sym
LinearAlgebra.adjoint(fs::FastBandlimited{T}) where{T} = Adjoint{Float64, FastBandlimited{T}}(fs)
Base.eltype(fs::FastBandlimited{T}) where{T} = Float64
Base.size(fs::FastBandlimited)         = fs.sz
Base.size(fs::FastBandlimited, j::Int) = size(fs)[j]


function LinearAlgebra.mul!(buf, fs::FastBandlimited, v)
  cv            = complex(v)
  fourier_buf   = fs.ft1*cv
  fourier_buf .*= fs.op
  tmp           = fs.ft2'*fourier_buf
  buf          .= real(tmp)
end

function Base.:*(fs::FastBandlimited, v)
  out = if (v isa Vector{Float64})
    Array{Float64}(undef, size(fs, 1))
  else
    Array{Float64}(undef, (size(fs, 1), size(v,2)))
  end
  mul!(out, fs, v)
end

function LinearAlgebra.mul!(buf, afs::Adjoint{Float64, FastBandlimited{T}}, v) where{T}
  fs = afs.parent
  cv = complex(v)
  fourier_buf   = fs.ft2*cv
  fourier_buf .*= fs.op
  tmp           = fs.ft1'*fourier_buf
  buf          .= real(tmp)
end

function Base.:*(afs::Adjoint{Float64, FastBandlimited}, v)
  out = if (v isa Vector{Float64})
    Array{Float64}(undef, size(afs, 1))
  else
    Array{Float64}(undef, (size(afs, 1), size(v,2)))
  end
  mul!(out, afs, v)
end

