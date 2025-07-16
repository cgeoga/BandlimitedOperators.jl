
using BandlimitedOperators, LowRankApprox

pts1  = sort(rand(1000)).*3
pts2  = sort(rand(1000)).*3
f(w)  = inv(2*18.1)
fastM = FastBandlimited(pts1, pts2, f, 18.1; allocating_mul=true)

(U, s, V) = psvd(fastM; rtol=1e-13)

M = [sinc(2*18.1*(xj-xk)) for xj in pts1, xk in pts2]
@show (rank(M), length(s))
@show opnorm(U*Diagonal(s)*V' - M)

