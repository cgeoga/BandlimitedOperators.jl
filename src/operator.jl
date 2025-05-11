
struct FastBandlimited
  ft1::NUFFT3
  ft2::NUFFT3
  op::Vector{ComplexF64}
end

function Base.display(fs::FastBandlimited)
  println("FastBandlimited operator with:")
  println("  - size: $(size(fs))")
  println("  - dimension: $(length(fs.ft1.s2))")
  println("  - quadrature rule order: $(length(fs.op))")
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

function highest_frequency(s1::Vector{Float64}, s2::Vector{Float64})
  (s1min, s1max) = extrema(s1)
  (s2min, s2max) = extrema(s2)
  max(abs(s2max - s1min), abs(s1max - s2min))
end

function highest_frequency(s1, s2, j::Int)
  (s1min, s1max) = extrema(x->x[j], s1)
  (s2min, s2max) = extrema(x->x[j], s2)
  max(abs(s2max - s1min), abs(s1max - s2min))
end

function bandlimited_quadrule(s1::Vector{Float64}, s2::Vector{Float64}, 
                              bandlimit::Float64, quadn_add, roughpoints)
  fmax  = highest_frequency(s1, s2)
  quadn = Int(ceil(4*bandlimit*fmax + quadn_add))
  isnothing(roughpoints) && return glquadrule(quadn, -bandlimit, bandlimit)
  regions = vcat(-bandlimit, sort(unique(roughpoints)), bandlimit)
  glquadrule(quadn, regions)
end

function bandlimited_quadrule(s1::Vector{SVector{D,Float64}}, 
                              s2::Vector{SVector{D,Float64}}, 
                              bandlimit, quadn_add, roughpoints) where{D}
  bandlimits = (bandlimit isa Float64) ? ntuple(_->bandlimit, D) : bandlimits
  quadnv = ntuple(D) do j
    fmaxj = highest_frequency(s1, s2, j)
    Int(ceil(4*bandlimits[j]*fmaxj + quadn_add))
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

# TODO (cg 2025/04/28 09:37): could choose more thoughtful default values here.
default_extra_quad(s1::Vector{Float64})            = 100 # 100  extra nodes
default_extra_quad(s1::Vector{SVector{2,Float64}}) = 20  # 400  extra nodes
default_extra_quad(s1::Vector{SVector{3,Float64}}) = 10  # 1000 extra nodes (least tested)

function internal_shift(s1::Vector{Float64}, s2::Vector{Float64})
  shifter = (sum(s1)/length(s1) + sum(s2)/length(s2))/2 
  (s1.-shifter, s2.-shifter)
end

function internal_shift(s1::Vector{SVector{D,Float64}}, 
                        s2::Vector{SVector{D,Float64}}) where{D}
  _shifter = ntuple(D) do j
    m1j = sum(x->x[j], s1)/length(s1)
    m2j = sum(x->x[j], s2)/length(s2)
    (m1j + m2j)/2
  end
  shifter = SVector{D,Float64}(_shifter)
  ([SVector{D,Float64}(x.data.-shifter.data) for x in s1], 
   [SVector{D,Float64}(x.data.-shifter.data) for x in s2])
end

function FastBandlimited(s1, s2, fn, bandlimit; 
                         quadn_add=default_extra_quad(s1), 
                         roughpoints=nothing)
  (_s1, _s2) = internal_shift(s1, s2)
  (no, wt)   = bandlimited_quadrule(_s1, _s2, bandlimit, quadn_add, roughpoints)
  op  = complex(wt.*fn.(no))
  ft1 = NUFFT3(no, _s2.*(2*pi), false, 1e-15)
  ft2 = NUFFT3(no, _s1.*(2*pi), false, 1e-15)
  FastBandlimited(ft1, ft2, op)
end

LinearAlgebra.ishermitian(fs::FastBandlimited) = (fs.ft1.s2 == fs.ft2.s2)
LinearAlgebra.issymmetric(fs::FastBandlimited) = (fs.ft1.s2 == fs.ft2.s2)
LinearAlgebra.adjoint(fs::FastBandlimited) = Adjoint{Float64, FastBandlimited}(fs)
Base.eltype(fs::FastBandlimited) = Float64
Base.size(fs::FastBandlimited)   = (size(fs.ft2, 2), size(fs.ft1, 2))
Base.size(fs::FastBandlimited, j::Int) = size(fs)[j]


function LinearAlgebra.mul!(buf, fs::FastBandlimited, v::AbstractVecOrMat{Float64})
  cv = complex(v)
  fourier_buf = fs.ft1*cv
  for j in 1:size(fourier_buf, 2)
    buf_colj   = view(fourier_buf, :, j)
    buf_colj .*= fs.op
  end
  tmp  = fs.ft2'*fourier_buf
  buf .= real(tmp)
end

function Base.:*(fs::FastBandlimited, v::AbstractVecOrMat{Float64})
  out = if (v isa Vector{Float64})
    Array{Float64}(undef, size(fs, 1))
  else
    Array{Float64}(undef, (size(fs, 1), size(v,2)))
  end
  mul!(out, fs, v)
end

function LinearAlgebra.mul!(buf, afs::Adjoint{Float64, FastBandlimited}, 
                            v::AbstractVecOrMat{Float64}) 
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

function Base.:*(afs::Adjoint{Float64, FastBandlimited}, v::AbstractVecOrMat{Float64})
  out = if (v isa Vector{Float64})
    Array{Float64}(undef, size(afs, 1))
  else
    Array{Float64}(undef, (size(afs, 1), size(v,2)))
  end
  mul!(out, afs, v)
end

