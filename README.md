
# BandlimitedOperators.jl

A simple package providing objects that implicitly represent the action of a
bandlimited kernel matrix on a vector. This action is computed at the cost of
two NUFFTs, which can often turn O(n^2) work into O(n \log n).  The
implementation here is based on the approach of the
[fast sinc transform](https://msp.org/camcos/2006/1-1/camcos-v1-n1-p06-p.pdf). 
It uses [FINUFFT](https://github.com/ludvigak/FINUFFT.jl) internally, and so
transforms are available in one, two, and three dimensions.

## A quick demo:

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

## Non-smooth points in your kernel's Fourier transform

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

**NOTE:** Even if the Fourier transform of your kernel is smooth at the origin,
if it has anything resembling a sharp peak there (or at any other location), it
is likely advantageous for you to provide the locations for those sharp peaks as
`roughpoints`. The default `roughpoints` argument is now (as of v0.1.7) the
origin (except in the case of `polar=true`, see below).

## Applying to multiple vectors at a time

By default, `FastBandlimited` uses planned "guru-mode" NUFFTs, which can provide
a big speedup. But guru mode plans require committing to a certain number of
vectors to apply to at once, so applying to a different number means looping
over columns or other fallback strategies that are less efficient. If you plan
to apply your operator to multiple vectors at a time, it can sometimes be faster to
instead use the non-guru mode and exploit the multiple-column speedup that
FINUFFT offers. The downside is simply that the `mul!` command will now make
allocations. To enable this, all you need to do is provide the
`allocating_mul=trud` kwarg:
```julia
fastM = FastBandlimited(pts1, pts2, fn, bandlimit; 
                        allocating_mul=true, kwargs...)
```

## Experimental: Bandlimited operators with support on a disk

Some functions have a Fourier transform that is supported on regions other than
rectangles. This package now offers the option to specify the support of your
kernel FT as being a disk of radius `bandwidth` **in two dimensions only** like so:
```julia
pts1  = [...] # ::Vector{SVector{2,Float64}}
pts2  = [...] # ::Vector{SVector{2,Float64}}

# the kernel ft `fn` should still be in __cartesian__ coordinates!
fastM = FastBandlimited(pts1, pts2, fn, bandwidth; polar=true)
```
**Note:** As of now, the quadrature rule being used in the Fourier domain is a
hybrid trapezoidal and Gauss-Legendre. Panel support is not currently
implemented, so your kernel FT should be smooth. Secondly, the size of the
quadrature rule is not very optimally designed and probably has more
oversampling than is necessary.

**Please see the files in `./example` for various other demonstrations.**

## Roadmap (PRs welcome!)

- `polar=true` in 3D
- More rigorous quadrature rule design for `polar=true` option
- Supporting ellipsoidal support regions for kernel FTs in 2 and 3 dimensions


