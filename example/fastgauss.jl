
using BandlimitedOperators

# (No need to sort, it just makes a prettier matrix plot of M when you are debugging)
pts1 = sort(rand(1000)).*10
pts2 = sort(rand(1200)).*10

const ALPHA = 1000.0 # shape parameter for Gaussian function
gauss_kernel(t) = exp(-ALPHA*abs2(t))
M = [gauss_kernel(xj-xk) for xj in pts1, xk in pts2]

# The Gaussian function isn't actually bandlimited, but its FT is also a
# Gaussian function and so numerically speaking it may as well be. The BANDWIDTH
# value here is the cutoff at which the FT goes below 1e-18.
const BANDWIDTH = sqrt(-(ALPHA/(pi^2))*log(sqrt(ALPHA/pi)*1e-18))
gauss_ft(omega) = sqrt(pi/ALPHA)*exp(-abs2(pi*omega)/ALPHA)

fastM = FastBandlimited(pts1, pts2, gauss_ft, BANDWIDTH)

x      = randn(length(pts2))
Mx     = M*x
fastMx = fastM*x
@show maximum(abs, Mx - fastMx)

# adjoint apply:
x2       = randn(length(pts1))
Mtx2     = M'*x2
fastMtx2 = fastM'*x2
@show maximum(abs, Mtx2 - fastMtx2)

