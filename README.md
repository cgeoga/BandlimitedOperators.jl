
# BandlimitedOperators.jl

A simple package providing objects that implicitly represent the action of a
bandlimited kernel matrix on a vector. This action is computed at the cost of
two NUFFTs, which can often turn O(n^2) work into O(n \log n).  The
implementation here is based on the 
[fast sinc transform](https://msp.org/camcos/2006/1-1/camcos-v1-n1-p06-p.pdf). 
It uses [FINUFFT](https://github.com/ludvigak/FINUFFT.jl) internally, and so
transforms are available in one, two, and three dimensions.

Here is a brief demonstration:
```julia
using BandlimitedOperators

pts1 = sort(rand(1000)).*3
pts2 = sort(rand(1000)).*3

# The matrix we want to approximate, which corresponds to a kernel matrix whose
# kernel function's Fourier transform is supported on [-18.1, 18.1].
M = [sinc(2*18.1*(xj-xk)) for xj in pts1, xk in pts2]

# We approximate it with a FastBandlimited object. Note the Fourier transform of
# this kernel function is precisely ω ↦ 2*χ(|ω| ≤ 18.1)/18.1, so that is the
# third argument here.
fastM = FastBandlimited(pts1, pts2, x->inv(2*18.1), 18.1)

x = randn(1000)
@show maximum(abs, M*x - fastM*x) # ~1e-13
```

If you have a kernel function whose Fourier transform has non-smooth points,
please pass those in with the `roughpoints` API. For example, here is how you
would do a fast sinc squared transform:
```julia
pts1 = sort(rand(1000)).*3
pts2 = sort(rand(1000)).*3

# Note that the Fourier transform of sinc(x)^2 is a triangle function. That is
# still an easy numerical integral...if you use a _panel_ quadrature rule
# because the function is not smooth at the origin. So we provide that location
# for splitting panels with roughpoints=(0.0,) (that arg should be an iterable):
triangle(x, bw) = max(0.0, 1-abs(x)/bw)
fastM = FastBandlimited(pts1, pts2, x->triangle(x, 18.1)/18.1, 18.1; 
                        roughpoints=(0.0,))

x     = randn(length(pts1))

# You can also use the in-place mul! of course:
buf   = zeros(length(x))
mul!(buf, fastM, x)

M = [sinc(18.1*(xj-xk))^2 for xj in pts1, xk in pts2]
@show maximum(abs, M*x - buf) # ~1e-13
```

**Please see the files in `./example` for demonstrations in 2D.**

