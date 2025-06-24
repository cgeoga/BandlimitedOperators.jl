
function trapquadrule(n::Int64, a::Float64, b::Float64; periodic=false)
  nquad = periodic ? n+1 : n
  no    = range(a, b, length=nquad)[1:n]
  wt    = fill(no[2] - no[1], n)
  (no, wt)
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

# TODO (cg 2025/06/24 10:43): note that this includes the extra r term in the
# change-of-variables going from euclidean to polar. I'm not sure if that is
# something this code should do or users should be trusted to handle themselves.
function polar_quadrule(nquadr::Int, nquadth_scale, rmax; min_nquadth::Int=8)
  # Gauss-Legendre for the radial part.
  (radial_no, radial_wt) = glquadrule(nquadr, 0.0, rmax)
  circular_rules = map(eachindex(radial_no)) do j
    (rnoj, rwtj) = (radial_no[j], radial_wt[j])
    # Trapezoidal for each of the angular parts.
    cnquadj = max(min_nquadth, Int(ceil(rnoj*nquadth_scale)))
    (cnoj, cwtj) = trapquadrule(cnquadj, 0.0, 2*pi; periodic=true)
    ([SVector{2,Float64}((rnoj, cj)) for cj in cnoj], cwtj.*(rwtj*rnoj))
  end
  (reduce(vcat, getindex.(circular_rules, 1)), 
   reduce(vcat, getindex.(circular_rules, 2)))
end

function polar_to_cartesian(x::SVector{2,Float64}) 
  SVector{2,Float64}((x[1]*cos(x[2]), x[1]*sin(x[2])))
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

# TODO (cg 2025/06/24 09:33): For now, this is only for kernels whose FT is both
# compactly supported and isotropic. But that restriction could certainly be
# lifted.
function polar_bandlimited_quadrule(s1::Vector{SVector{2,Float64}}, 
                                    s2::Vector{SVector{2,Float64}}, 
                                    bandlimit, quadn_add, roughpoints)
  if !isnothing(roughpoints) 
    throw(error("Panel polar quadrature is not currently supported. Please open an issue."))
  end
  fmax  = maximum(j->highest_frequency(s1, s2, j), 1:2)
  nquad = Int(ceil(2*pi*bandlimit*fmax + quadn_add))
  (no_polar, wt) = polar_quadrule(nquad, 4*pi*nquad, bandlimit)
  (polar_to_cartesian.(no_polar), wt)
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

