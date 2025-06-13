
# This is just for debugging and testing.
function nudftmatrix(s1, s2, sgn)
  sgn in (-1.0, 1.0) || throw(error("Sign should be either -1.0 or 1.0! You provided $sgn."))
  [cispi(2*sgn*dot(sj, sk)) for sj in s1, sk in s2]
end

"""
  NUFFT3: a struct representing the action of a nonuniform Fourier matrix using the NUFFT.
  Calling

  ```
    fs = NUFFT3(s1, s2, sgn::Int)
  ```

  gives an object whose `mul!` operation represents the action of the matrix 

  ```
  sign = sgn < 0 ? -1 : 1
  [cispi(sign*2*dot(sj, sk)) for sj in s1, sk in s2]
  ```

  Keyword arguments are:

  -- `tol=1e-15`: The accuracy of the NUFFT. Unless you have a specific reason
     to make it looser, we suggest keeping it at the default `1e-15`.

  -- `ncol=1`: the number of columns of a matrix that the object can act on at once. If you leave `ncol=1`, this package has methods to automatically apply to several columns in a loop. But if you set `ncol` to be >1, then it will be your responsibility to make sure that you pass in your `mul!` arguments with the appropriate number of columns (or else you get an error).

  -- `ncol_warning=true`: a kwarg giving you the option to disable the warning about `ncol` above.

  -- `make_adjoint=true`: whether or not to also build the adjoint operator. If you want to do `fs'*v`, then you need to do this. But if you know you won't need to do this, you can save the (slight) memory and time with `make_adjoint=false`.
"""
struct NUFFT3
  plan::finufft_plan
  adjplan::Union{finufft_plan, Nothing}
  sz::Tuple{Int64, Int64}
  ncol::Int64
end

function nufft3plan(s1::Matrix{Float64}, s2::Matrix{Float64}, sgn::Int, tol, ncol)
  dim  = size(s1, 2)
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
  plan
end

# NOTE: this does _not_ do the 2pi scaling for you.
function NUFFT3(s1::Matrix{Float64}, s2::Matrix{Float64}, sgn::Int;
                tol=1e-15, ncol=1, ncol_warning=true, make_adjoint=true)
  iszero(sgn) && throw(error("Please provide sign as < 0 or > 0."))
  if ncol > 1 && ncol_warning
    @warn "Specifying ncol > 1 means that you can _only_ mul! this operator on things with exactly that number of columns. You can disable this warning with the kwarg ncol_warning=false in the constructor call." maxlog=1
  end
  plan = nufft3plan(s1, s2, sgn, tol, ncol)
  adjplan = make_adjoint ? nufft3plan(s2, s1, -sgn, tol, ncol) : nothing
  NUFFT3(plan, adjplan, (size(s1, 1), size(s2, 1)), ncol)
end

function NUFFT3(s1::Vector{SVector{S,Float64}}, 
                s2::Vector{SVector{S,Float64}}, sgn::Int; kwargs...) where{S}
  s1m = permutedims(reduce(hcat, s1))
  s2m = permutedims(reduce(hcat, s2))
  NUFFT3(s1m, s2m, sgn; kwargs...)
end

function NUFFT3(s1::Vector{Float64}, s2::Vector{Float64}, sgn::Int; kwargs...)
  NUFFT3(hcat(s1), hcat(s2), sgn; kwargs...)
end

Base.eltype(nf::NUFFT3)       = ComplexF64
Base.size(nf::NUFFT3)         = nf.sz
Base.size(nf::NUFFT3, j::Int) = size(nf)[j]

LinearAlgebra.adjoint(nf::NUFFT3) = Adjoint(nf)

# TODO (cg 2025/06/13 10:26): if nf.ncol != size(x,2), this throws an error that
# is not easy to read. Should this method potentially add columns?
function LinearAlgebra.mul!(buf, nf::NUFFT3, x)
  if nf.ncol == size(x, 2) == size(buf, 2)
    finufft_exec!(nf.plan, x, buf)
  elseif isone(nf.ncol) && (size(x, 2) > 1) && size(buf, 2) == size(x, 2)
    for (bufj, xj) in zip(eachcol(buf), eachcol(x))
      finufft_exec!(nf.plan, xj, bufj)
    end
  else
    throw(error("Either your input and output sizes don't agree, or you specified your NUFFT3 operator to act on more than one column at once and the number of provided columns don't match that."))
  end
  buf
end

function LinearAlgebra.mul!(buf, anf::Adjoint{ComplexF64, NUFFT3}, x)
  nf = anf.parent
  if nf.ncol == size(x, 2) == size(buf, 2)
    finufft_exec!(nf.adjplan, x, buf)
  elseif isone(nf.ncol) && (size(x, 2) > 1) && size(buf, 2) == size(x, 2)
    for (bufj, xj) in zip(eachcol(buf), eachcol(x))
      finufft_exec!(nf.adjplan, xj, bufj)
    end
  else
    throw(error("Either your input and output sizes don't agree, or you specified your NUFFT3 operator to act on more than one column at once and the number of provided columns don't match that."))
  end
  buf
end

function Base.:*(fs::NUFFT3, x::Vector)
  buf = Array{ComplexF64}(undef, size(fs, 1))
  mul!(buf, fs, complex(x))
end

function Base.:*(fs::Adjoint{ComplexF64, NUFFT3}, x::Vector)
  buf = Array{ComplexF64}(undef, size(fs, 1))
  mul!(buf, fs, complex(x))
end

function Base.:*(fs::NUFFT3, x::Matrix)
  buf = Array{ComplexF64}(undef, size(fs, 1), size(x, 2))
  mul!(buf, fs, complex(x))
end

function Base.:*(fs::Adjoint{ComplexF64, NUFFT3}, x::Matrix)
  buf = Array{ComplexF64}(undef, size(fs, 1), size(x, 2))
  mul!(buf, fs, complex(x))
end
