
#
# Testset 1: the mixed GL+trapezoidal polar coordinate quadrature.
#
(no_polar, wt) = BandlimitedOperators.polar_quadrule(100, 20, 1.0)
no_cartesian   = BandlimitedOperators.polar_to_cartesian.(no_polar)
@test sum(wt) ≈ pi

# integrating f(x)=e^{-||x||^2} on the unit disk:
gauss_polar(r_theta) = exp(-r_theta[1]^2) # f(x) -> f((r, θ))
gauss_cart(xy)       = exp(-norm(xy)^2)

@test dot(wt, gauss_polar.(no_polar))    ≈ pi*(1-exp(-1))  
@test dot(wt, gauss_cart.(no_cartesian)) ≈ pi*(1-exp(-1)) 

#
# Testset 2: the fast jinc transform. This is just ripped from the example file,
# at least for now.
# 

pts1 = [x + SA[10.0, 10.0] for x in rand(SVector{2,Float64}, 1000).*5.0]
pts2 = [x + SA[10.0, 10.0] for x in rand(SVector{2,Float64}, 1100).*0.1]

function jinc(x::SVector{2,Float64}, y::SVector{2,Float64}, bw)
  nx = 2*pi*bw*norm(x-y)
  iszero(nx) && return 0.5
  Bessels.besselj1(nx)/nx
end

trueM = [jinc(x, y, 1.0) for x in pts1, y in pts2]
fastM = FastBandlimited(pts1, pts2, x->inv(2*pi), 1.0; polar=true)

# forward apply:
x   = randn(length(pts2))
buf = zeros(length(pts1))
mul!(buf, fastM, x)
@test maximum(abs, trueM*x - buf) < 1e-11

# adjoint apply:
x   = randn(length(pts1))
buf = zeros(length(pts2))
mul!(buf, Adjoint(fastM), x)
@test maximum(abs, trueM'*x - buf) < 1e-11

