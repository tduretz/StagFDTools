
using Enzyme, Plots, LinearAlgebra, StaticArrays

# function Stress( ε̇, ηv, G, Δt )
#     η_eff = inv( 1/ηv + 1/(G*Δt) )
#     τ     = [2*η_eff*ε̇[1], 2*η_eff*ε̇[2], 2*η_eff*ε̇[3]]
#     return τ
# end

# function Stress_pwl( ε̇, η0, n, G, Δt )
#     η_eff = inv( 1/ηv + 1/(G*Δt) )
#     τ     = [2*η_eff*ε̇[1], 2*η_eff*ε̇[2], 2*η_eff*ε̇[3]]
#     return τ
# end

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

function StrainRateTrial_Invariant(τII, G, Δt, B, n)
    ε̇II_vis   = B.*τII.^n 
    ε̇II_trial = ε̇II_vis + τII/(2*G*Δt)
    return ε̇II_trial
end

function EffectiveViscosity(τII, G, Δt, B, n)
    eta_vis =  1/2*B^-1*τII.^(1-n)
    return inv(1/eta_vis + 1/(G*Δt))
end

# function StressTrial(ε̇, τ_trial, τ0, G, Δt, η0, n)
#     ε̇_el    = (τ_trial .- τ0) ./(2*G.*Δt)
#     ε̇vis    = ε̇ - ε̇_el
#     ε̇II_vis = sqrt(0.5*(ε̇vis[1]^2 + ε̇vis[2]^2) + ε̇vis[3]^2) 
#     ηv      = η0^(-1/n) * ε̇II_vis.^(1/n-1)
#     η_eff   = inv( 1/ηv + 1/(G*Δt) )
#     τ = 2*η_eff*(ε̇ .+ τ0 ./(2*G.*Δt))
#     return τ
# end

let 
    # Materials
    η0  = 1.0
    n   = 2.0
    B   = (2*η0)^(-n)
    G   = 1.0

    # Kinematics
    D_BC = [-4  0.0;
             0  4]
    
    # D_BC = [-0  4.0;
    # 0  0]

    D_BC = [-1  2.4;
    0  1]

    # Numerics
    nt    = 3
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
        τ       = 2*η_eff*( ε̇ .+ τ0 ./(2*G.*Δt) )

        # For invariant form
        ε̇_eff   = ε̇ + τ0 / (2*G*Δt)
        ε̇II_eff = sqrt(([1/2; 1/2; 1].*ε̇_eff)'*ε̇_eff)
        τII     = sqrt(([1/2; 1/2; 1].*τ)'*τ)

        # Storage for CTL
        D_ctl    = @MMatrix zeros(3,3) 
        ∂τII∂ε̇II = 0. 
        ∂η∂τII   = 0.
        
        for iter=1:niter

            #-----------------------------------------#

            # Local iterations using al stress components
            r         = ε̇ - StrainRateTrial(τ, ε̇, τ0, G, Δt, B, n) 
            J         = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial, τ, ε̇, τ0, G, Δt, B, n)
            J⁻¹       = inv(J[1])
            τ       .+= J[1]\r
            r_iter[iter] = norm(r)
            @show iter, norm(r)

            # If we store the inverse, then we get the consistent tangent for global iterations
            D_ctl .= inv(J[1])

            #-----------------------------------------#

            # Local iterations using al stress invariant
            r      = ε̇II_eff - StrainRateTrial_Invariant(τII, G, Δt, B, n)
            ∂ε̇II∂τII = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial_Invariant, τII, G, Δt, B, n)
            ∂τII∂ε̇II = inv(∂ε̇II∂τII[1])
            τII     += ∂τII∂ε̇II*r

            # Kind of need that guy to reconstruct the CTL
            ∂η∂τII = Enzyme.jacobian(Enzyme.Forward, EffectiveViscosity, τII, G, Δt, B, n)

            #-----------------------------------------#
            r_iter[iter] = norm(r)
            norm(r)<1e-11 && break

        end

        @info "Check stress invariants"
        @show τII - sqrt(([1/2; 1/2; 1].*τ)'*τ)

        @info "Check strain rate invariants"
        ε̇II_vis = B.*τII.^n 
        @show ε̇II_eff - ε̇II_vis - τII/(2*G*Δt)

        @info "Check effective viscosity"
        η_eff1  = τII/2/ε̇II_eff
        eta_vis =  1/2*B^-1*τII.^(1-n)
        η_eff2  = inv(1/eta_vis + 1/(G*Δt))
        @show η_eff1 - η_eff2

        @info "Check viscosity derivative"
        ∂η∂τII_1  = −2*B*η_eff2^2*τII^(n−2)*(n−1)
        @show ∂η∂τII[1] - ∂η∂τII_1

        @info "Check consistent tangent"
        ∂ε̇II∂ε̇  = [1/2 1/2 1] .* ε̇_eff/ε̇II_eff
        ∂τ∂τII  = ∂η∂τII[1]*ε̇_eff

        @show ∂τII∂ε̇II*∂ε̇II∂ε̇[3]*∂τ∂τII[3]
        
        D_ctl_reconstructed = [ 
              ∂τII∂ε̇II-2(∂τII∂ε̇II*∂ε̇II∂ε̇[1]*∂τ∂τII[1]+2*∂τII∂ε̇II*∂ε̇II∂ε̇[3]*∂τ∂τII[3]) -2*∂τII∂ε̇II*∂ε̇II∂ε̇[1]*∂τ∂τII[1]                                             4*∂τII∂ε̇II*∂ε̇II∂ε̇[3]*∂τ∂τII[3]*τ[1]/τ[3];
           -2*∂τII∂ε̇II*∂ε̇II∂ε̇[2]*∂τ∂τII[2]                                               ∂τII∂ε̇II-2*(∂τII∂ε̇II*∂ε̇II∂ε̇[2]*∂τ∂τII[2]+2*∂τII∂ε̇II*∂ε̇II∂ε̇[3]*∂τ∂τII[3]) 4*∂τII∂ε̇II*∂ε̇II∂ε̇[3]*∂τ∂τII[3]*τ[2]/τ[3];
            2*∂τII∂ε̇II*∂ε̇II∂ε̇[3]*∂τ∂τII[3]*τ[1]/τ[3]                                   2*∂τII∂ε̇II*∂ε̇II∂ε̇[3]*∂τ∂τII[3]*τ[2]/τ[3]                                     ∂τII∂ε̇II-2*(∂τII∂ε̇II*∂ε̇II∂ε̇[2]*∂τ∂τII[2]+∂τII∂ε̇II*∂ε̇II∂ε̇[1]*∂τ∂τII[1])                  
        ] 
        display( D_ctl_reconstructed .- D_ctl)

        @info "Consistent tangent operator"
        display( D_ctl )
        # display( D_ctl_reconstructed )

        # Log stress invariant
        τII_t[it] = sqrt(([1/2; 1/2; 1].*τ)'*τ)
        p1 = scatter((1:niter), log10.(r_iter), title="convergence", label=:none)
        p2 = plot((1:nt).*Δt, τII_t, title="stress-time", label=:none)
        display( plot(p1, p2) )

        # @show 2*∂τII∂ε̇II*∂ε̇II∂ε̇[3]*∂τ∂τII[3] * τ[1]/τ[3]*2

        # @show τ[1]/τ[3]*2

        # @show (-τ[3])/τII*2
        # @show (-ε̇_eff[3])/ε̇II_eff*2
        # @show ∂τII∂ε̇II


        # @show D_ctl[1,3] 
        # @show 1*∂τII∂ε̇II*∂ε̇II∂ε̇[1]*∂τ∂τII[1]
        # @show 1*∂τII∂ε̇II*∂ε̇II∂ε̇[2]*∂τ∂τII[2]
        # @show 1*∂τII∂ε̇II*∂ε̇II∂ε̇[3]*∂τ∂τII[3]
        # @show (∂τII∂ε̇II* ε̇_eff[3]/ε̇II_eff + τII*ε̇_eff[3]/ε̇II_eff^2)*(ε̇_eff[3]/ε̇II_eff/1)
        # @show (∂τII∂ε̇II* ε̇_eff[1]/ε̇II_eff + τII*ε̇_eff[1]/ε̇II_eff^2)*(ε̇_eff[3]/ε̇II_eff/1)

        # J = 2*[
        #     1+∂τII∂ε̇II*τ[1]/ε̇_eff[1] ∂τII∂ε̇II*τ[1]/ε̇_eff[2] ∂τII∂ε̇II*τ[1]/ε̇_eff[3];
        #     ∂τII∂ε̇II*τ[2]/ε̇_eff[1] 1+∂τII∂ε̇II*τ[2]/ε̇_eff[2] ∂τII∂ε̇II*τ[2]/ε̇_eff[3];
        #     ∂τII∂ε̇II*τ[3]/ε̇_eff[1] ∂τII∂ε̇II*τ[3]/ε̇_eff[2] 1+∂τII∂ε̇II*τ[3]/ε̇_eff[3];
        # ]
        # display(inv(J))
    end
end