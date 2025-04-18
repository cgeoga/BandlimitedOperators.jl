
# BandlimitedOperators.jl

A simple package providing objects that implicitly represent the action of a
bandlimited kernel matrix on a vector. Using the NUFFT, this action is computed
at the cost of two NUFFTs, which can often turn O(n^2) work into O(n \log n).
The implementation here is a simple extension of the 
[fast sinc transform](https://msp.org/camcos/2006/1-1/camcos-v1-n1-p06-p.pdf).

Here is a brief demonstration:
```julia
pts1 = sort(rand(1000)).*3
pts2 = sort(rand(1000)).*3

# The matrix we want to approximate, which corresponds to a kernel matrix whose
# kernel function's Fourier transform is supported on [-18.1, 18.1].
M = [sinc(2*18.1*(xj-xk)) for xj in pts2, xk in pts1]

# We approximate it with a FastBandlimited object. Note the Fourier transform of
# this kernel function is precisely ω ↦ 2*χ(|ω| ≤ 18.1)/18.1, so that is the
# third argument here.
fastM = FastBandlimited(pts1, pts2, x->inv(2*18.1), 18.1; quadn_add=10)

# apply it to a vector with mul!, as usual.
buf = zeros(length(x))
mul!(buf, fastM, x)

@show maximum(abs, M'*x - buf) # ~1e-13
```

**Notes**:

-- This package uses [`FINUFFT.jl`](https://github.com/ludvigak/FINUFFT.jl),
which offers transformations in 1D, 2D, and 3D. For 2D and 3D transformations,
please pass points in as `SVector`s.

-- In 1D there is an interface for kernels whose compactly supported Fourier
transform has points of non-smoothness. Since the accuracy of this package
depends on the accuracy of a quadrature rule for resolving Fourier integrals of
this function, it is crucial to communicate these rough points so that panel
quadrature can be used. Please pass them in as some kind of iterable collection
using the `roughpoints=(...)` kwarg. If you want this functionality in 2D,
please open an issue or a PR.

