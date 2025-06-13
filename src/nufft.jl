
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

function Base.:*(fs::NUFFT3, x::Vector{S}) where{S}
  buf = Array{ComplexF64}(undef, size(fs, 1))
  mul!(buf, fs, complex(x))
end

function Base.:*(fs::NUFFT3, x::Matrix{S}) where{S}
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

function Base.:*(afs::Adjoint{ComplexF64, NUFFT3}, x::Vector{S}) where{S}
  buf = Array{ComplexF64}(undef, (size(afs, 1)))
  mul!(buf, afs, complex(x))
end

# TODO (cg 2025/04/25 10:06): Splitting this into a separate method because of
# some weird dispatch ambiguity errors that the compiler throws. Not sure if
# that is me doing something dumb or a compiler error.
function Base.:*(afs::Adjoint{ComplexF64, NUFFT3}, x::Matrix{S}) where{S}
  buf = Array{ComplexF64}(undef, (size(afs, 1), size(x,2)))
  mul!(buf, afs, complex(x))
end

struct NUFFT3Plan
  plan::finufft_plan
  sz::Tuple{Int64, Int64}
  ncol::Int64
end

# NOTE: this does _not_ do the 2pi scaling for you.
function NUFFT3Plan(s1::Matrix{Float64}, s2::Matrix{Float64}, sgn::Int;
                    tol=1e-15, ncol=1, ncol_warning=true)
  if ncol > 1 && ncol_warning
    @warn "Specifying ncol > 1 means that you can _only_ mul! this operator on things with exactly that number of columns. You can disable this warning with the kwarg ncol_warning=false in the constructor call."
  end
  dim  = size(s1, 2)
  isone(dim) || throw(error("for now"))
  plan = finufft_makeplan(3, dim, Int32(sgn), ncol, tol)
  if dim == 1
    finufft_setpts!(plan, s2[:,1], Float64[], Float64[],
                          s1[:,1], Float64[], Float64[])
  elseif dim == 2
    finufft_setpts!(plan, s2[:,1], s2[:,2], Float64[],
                          s1[:,1], s1[:,2], Float64[])
  elseif dim == 3
    finufft_setpts!(plan, s2[:,1], s2[:,2], s2[:,3],
                          s1[:,1], s1[:,2], s1[:,3])
  else
    throw(error("This transform is only implemented in dimensions {1,2,3}."))
  end
  NUFFT3Plan(plan, (size(s1, 1), size(s2, 1)), ncol)
end

function NUFFT3Plan(s1::Vector{Float64}, s2::Vector{Float64}, sgn::Int;
                    tol=1e-15, ncol=1, ncol_warning=true)
  NUFFT3Plan(hcat(s1), hcat(s2), sgn; tol=tol, ncol=ncol, ncol_warning=ncol_warning)
end

Base.eltype(nf::NUFFT3Plan)       = ComplexF64
Base.size(nf::NUFFT3Plan)         = nf.sz
Base.size(nf::NUFFT3Plan, j::Int) = size(nf)[j]

# TODO (cg 2025/06/13 10:26): if nf.ncol != size(x,2), this throws an error that
# is not easy to read. Should this method potentially add columns?
function LinearAlgebra.mul!(buf, nf::NUFFT3Plan, x)
  if nf.ncol == size(x, 2) == size(buf, 2)
    finufft_exec!(nf.plan, x, buf)
  elseif isone(nf.ncol) && (size(x, 2) > 1) && size(buf, 2) == size(x, 2)
    for (bufj, xj) in zip(eachcol(buf), eachcol(x))
      finufft_exec!(nf.plan, xj, bufj)
    end
  else
    throw(error("Either your input and output sizes don't agree, or you specified your NUFFT3Plan operator to act on more than one column at once and the number of provided columns don't match that."))
  end
  buf
end

function Base.:*(fs::NUFFT3Plan, x::Vector)
  buf = Array{ComplexF64}(undef, size(fs, 1))
  mul!(buf, fs, complex(x))
end

function Base.:*(fs::NUFFT3Plan, x::Matrix)
  buf = Array{ComplexF64}(undef, size(fs, 1), size(x, 2))
  mul!(buf, fs, complex(x))
end

