
# This is just for debugging and testing.
function nudftmatrix(s1, s2, sgn)
  sgn in (-1.0, 1.0) || throw(error("Sign should be either -1.0 or 1.0! You provided $sgn."))
  [cispi(2*sgn*dot(sj, sk)) for sj in s1, sk in s2]
end

# TODO (cg 2025/04/25 09:10): decide on a scheme for pre-allocating buffers
# here. The mul! routine is column-agnostic, and certainly the NUFFT is
# expensive enough that you really want to re-use the spreading work for many
# columns. So maybe in the end we want to allocate, say, 10 or 100 columns, and
# the mul! we write here can in the worst case work on chunks of that size. But
# I don't think FINUFFT exposes a plan object that saves the spreading work.
struct NUFFT3
  s1::Vector{Vector{Float64}}
  s2::Vector{Vector{Float64}}
  sgn::Bool
  tol::Float64
end

"""
  NUFFT3: a struct representing the action of a nonuniform Fourier matrix using the NUFFT.
  Calling

  ```
    fs = NUFFT3(s1, s2, sgn, tol=1e-15)
  ```

  gives an object whose `mul!` operation represents the action of the matrix 

  ```
  sign = sgn ? 1 : -1
  [cispi(sign*2*dot(sj, sk)) for sj in s1, sk in s2]
  ```

  The `tol` keyword is provided to FINUFFT. Unless you have a specific reason to make it looser, we suggest keeping it at the default `1e-15`.
"""
function NUFFT3(s1::Vector{Float64}, s2::Vector{Float64}, sgn::Bool, tol=1e-15)
  NUFFT3([s1], [s2], sgn, tol)
end

function NUFFT3(s1::Matrix{Float64}, s2::Matrix{Float64}, sgn::Bool, tol=1e-15)
  size(s1, 2) == size(s2, 2) || throw(error("s1 and s2 aren't of the same dimension."))
  in(size(s1, 2), (1,2,3))   || throw(error("This operator is only implemented in 1, 2, or 3D."))
  NUFFT3(collect.(eachcol(s1)), collect.(eachcol(s2)), sgn, tol)
end

function NUFFT3(s1::Vector{SVector{S,Float64}}, 
                s2::Vector{SVector{S,Float64}}, sgn::Bool, tol=1e-15) where{S}
  in(S, (1,2,3))   || throw(error("This operator is only implemented in 1, 2, or 3D."))
  s1v = [getindex.(s1, j) for j in 1:S]
  s2v = [getindex.(s2, j) for j in 1:S]
  NUFFT3(s1v, s2v, sgn, tol)
end

Base.eltype(nf::NUFFT3)       = ComplexF64
Base.size(nf::NUFFT3)         = (length(nf.s1[1]), length(nf.s2[1]))
Base.size(nf::NUFFT3, j::Int) = size(nf)[j]

LinearAlgebra.adjoint(nf::NUFFT3) = Adjoint{ComplexF64, NUFFT3}(nf)

LinearAlgebra.ishermitian(nf::NUFFT3)  = false
function LinearAlgebra.mul!(buf, nf::NUFFT3, x)
  ifl = nf.sgn ? Int32(1) : Int32(-1)
  dim = length(nf.s1)
  if dim == 1
    nufft1d3!(nf.s2[1], collect(x), ifl, nf.tol, nf.s1[1], buf)
  elseif dim == 2
    s11 = nf.s1[1]
    s12 = nf.s1[2]
    s21 = nf.s2[1]
    s22 = nf.s2[2]
    nufft2d3!(s21, s22, collect(x), ifl, nf.tol, s11, s12, buf)
  elseif dim == 3
    s11 = nf.s1[1]
    s12 = nf.s1[2]
    s13 = nf.s1[3]
    s21 = nf.s2[1]
    s22 = nf.s2[2]
    s23 = nf.s2[3]
    nufft3d3!(s21, s22, s23, collect(x), ifl, nf.tol, s11, s12, s13, buf)
  else
    throw(error("This operator is only defined in dimensions 1, 2, and 3."))
  end
  buf
end

function Base.:*(fs::NUFFT3, x::AbstractVector{S}) where{S}
  buf = Array{ComplexF64}(undef, size(fs, 1))
  mul!(buf, fs, complex(x))
end

function Base.:*(fs::NUFFT3, x::AbstractMatrix{S}) where{S}
  buf = Array{ComplexF64}(undef, (size(fs, 1), size(x,2)))
  mul!(buf, fs, complex(x))
end

function LinearAlgebra.mul!(buf, anf::Adjoint{ComplexF64, NUFFT3}, x)
  nf  = anf.parent
  ifl = nf.sgn ? Int32(-1) : Int32(1)
  dim = length(nf.s1)
  if dim == 1
    nufft1d3!(nf.s1[1], collect(x), ifl, nf.tol, nf.s2[1], buf)
  elseif dim == 2
    s11 = nf.s1[1]
    s12 = nf.s1[2]
    s21 = nf.s2[1]
    s22 = nf.s2[2]
    nufft2d3!(s11, s12, collect(x), ifl, nf.tol, s21, s22, buf)
  elseif dim == 3
    s11 = nf.s1[1]
    s12 = nf.s1[2]
    s13 = nf.s1[3]
    s21 = nf.s2[1]
    s22 = nf.s2[2]
    s23 = nf.s2[3]
    nufft3d3!(s11, s12, s13, collect(x), ifl, nf.tol, s21, s22, s23, buf)
  else
    throw(error("This operator is only defined in dimensions 1, 2, and 3."))
  end
  buf
end

function Base.:*(afs::Adjoint{ComplexF64, NUFFT3}, x::AbstractVector{S}) where{S}
  fs  = afs.parent
  buf = Array{ComplexF64}(undef, (size(fs, 2)))
  mul!(buf, afs, complex(x))
end

# TODO (cg 2025/04/25 10:06): Splitting this into a separate method because of
# some weird dispatch ambiguity errors that the compiler throws. Not sure if
# that is me doing something dumb or a compiler error.
function Base.:*(afs::Adjoint{ComplexF64, NUFFT3}, x::AbstractMatrix{S}) where{S}
  fs  = afs.parent
  buf = Array{ComplexF64}(undef, (size(afs, 2), size(x,2)))
  mul!(buf, afs, complex(x))
end

