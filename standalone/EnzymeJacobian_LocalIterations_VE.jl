
using Enzyme, Plots, LinearAlgebra

function Stress( ε̇, ηv, G, Δt )
    η_eff = inv( 1/ηv + 1/(G*Δt) )
    τ     = [2*η_eff*ε̇[1], 2*η_eff*ε̇[2], 2*η_eff*ε̇[3]]
    return τ
end

function Stress_pwl( ε̇, η0, n, G, Δt )
    η_eff = inv( 1/ηv + 1/(G*Δt) )
    τ     = [2*η_eff*ε̇[1], 2*η_eff*ε̇[2], 2*η_eff*ε̇[3]]
    return τ
end

# function Residual(τ_trial, ε̇, τ0, G, Δt, B, n)
#     τII_trial = sqrt(0.5*(τ_trial[1]^2 + τ_trial[2]^2) + τ_trial[3]^2)    #sqrt(([1/2; 1/2; 1].*τ_trial)'*τ_trial)
#     ε̇_el      = (τ_trial .- τ0) ./(2*G.*Δt)
#     ε̇_vis     = B.*τII_trial.^n .* (τ_trial./2/τII_trial)
#     ε̇_trial   = ε̇_el + ε̇_vis
#     r         = ε̇ - ε̇_trial
#     return r
# end

function StrainRateTrial(τ_trial, ε̇, τ0, G, Δt, B, n)
    τII_trial = sqrt(0.5*(τ_trial[1]^2 + τ_trial[2]^2) + τ_trial[3]^2)    #sqrt(([1/2; 1/2; 1].*τ_trial)'*τ_trial)
    ε̇_el      = (τ_trial .- τ0) ./(2*G.*Δt)
    ε̇II_vis   = B.*τII_trial.^n 
    ε̇_vis     = ε̇II_vis .* (τ_trial./τII_trial)
    ε̇_trial   = ε̇_el + ε̇_vis
    return ε̇_trial
end

function StressTrial(ε̇, τ_trial, τ0, G, Δt, η0, n)
    ε̇_el    = (τ_trial .- τ0) ./(2*G.*Δt)
    ε̇vis    = ε̇ - ε̇_el
    ε̇II_vis = sqrt(0.5*(ε̇vis[1]^2 + ε̇vis[2]^2) + ε̇vis[3]^2) 
    ηv      = η0^(-1/n) * ε̇II_vis.^(1/n-1)
    η_eff   = inv( 1/ηv + 1/(G*Δt) )
    τ = 2*η_eff*(ε̇ .+ τ0 ./(2*G.*Δt))
    return τ
end

# let 
#     η   = 1.0
#     G   = 1.0
#     Δt  = 0.5
#     D_BC = [-1  0;
#              0  1]
#     ε̇  = [D_BC[1,1]-1/3*(D_BC[1,1]+D_BC[2,2]), D_BC[2,2]-1/3*(D_BC[1,1]+D_BC[2,2]), 1/2*(D_BC[1,2] + D_BC[2,1])]
#     τ  = [0., 0., 0.]
#     τ0 = zero(τ)
#     ϵ̇  = zero(ε̇)
#     nt    = 30
#     τII_t = zeros(nt)
#     for it in 1:nt 
#         τ0 .= τ
#         ϵ̇  .= ε̇ .+ τ0 ./(2*G.*Δt)
#         jac = Enzyme.jacobian(Enzyme.ForwardWithPrimal, Stress, ϵ̇, η, G, Δt)
#         D   = jac.derivs[1]
#         τ   = jac.val
#         # Log stress invariant
#         τII_t[it] = sqrt(([1/2; 1/2; 1].*τ)'*τ)
#     end
#     plot((1:nt).*Δt, τII_t)
# end

let 
    # Materials
    η0  = 1.0
    n   = 2.0
    B   = (2*η0)^(-n)
    G   = 1.0

    # Kinematics
    D_BC = [-1  0;
    0  1]

    # Numerics
    nt    = 1
    niter = 15
    Δt    = 0.5

    #  Initialisation
    ε̇     = [D_BC[1,1]-1/3*(D_BC[1,1]+D_BC[2,2]), D_BC[2,2]-1/3*(D_BC[1,1]+D_BC[2,2]), 1/2*(D_BC[1,2] + D_BC[2,1])]
    τ     = [0., 0., 0.]
    τ0    = zero(τ)
    ϵ̇     = zero(ε̇)

    # Logs
    τII_t  = zeros(nt)
    r_iter = zeros(niter)

    for it in 1:nt 

        τ0     .= τ
        ε̇II     = sqrt(([1/2; 1/2; 1].*ε̇)'*ε̇)
        ηv      = η0^(-1/n) * ε̇II^(1/n-1)
        η_eff   = inv( 1/ηv + 1/(G*Δt) )
        τ_trial = 2*η_eff*( ε̇ .+ τ0 ./(2*G.*Δt) )
        
        for iter=1:niter
            # r = Residual(τ_trial, ε̇, τ0, G, Δt, B, n) 
            # J = Enzyme.jacobian(Enzyme.Forward, Residual, τ_trial, ε̇, τ0, G, Δt, B, n)
            # τ_trial .-= J[1]\r
            # r_iter[iter] = norm(r)
            # @show iter, norm(r)

            # Local residual
            r         = ε̇ - StrainRateTrial(τ_trial, ε̇, τ0, G, Δt, B, n) 
            J         = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial, τ_trial, ε̇, τ0, G, Δt, B, n)
            J⁻¹       = inv(J[1])
            τ_trial .+= J[1]\r
            r_iter[iter] = norm(r)
            @show iter, norm(r)

            # If we store the inverse, then we get the consistent tangent for global iterations
            D_ctl = inv(J[1])
            display( D_ctl )

            # τ1 = StressTrial(ε̇, τ_trial, τ0, G, Δt, η0, n)
            # D = Enzyme.jacobian(Enzyme.Forward, StressTrial, ε̇, τ_trial, τ0, G, Δt, B, n)
            # display( D[1] )
            # @show τ1 , τ_trial
        end

        # Log stress invariant
        τII_t[it] = sqrt(([1/2; 1/2; 1].*τ)'*τ)
        display( plot((1:niter), log10.(r_iter)) )
    end
    # plot((1:nt).*Δt, τII_t)
end