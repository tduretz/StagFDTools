function line(p, K, dt, η_ve, ψ, p1, t1)
    p2 = p1 + K*dt*sind(ψ)  # introduce sinϕ ?
    t2 = t1 - η_ve  
    a  = (t2-t1)/(p2-p1)
    b  = t2 - a*p2
    return a*p + b
end

function Kiss2023(τ, P, η_ve, comp, β, Δt, C, φ, ψ, ηvp, σ_T, δσ_T, pc1, τc1, pc2, τc2)

    K         = 1/β
    λ̇         = 0.
    domain_pl = 0.0
    Pc        = P
    τc        = τ

    l1    = line(P, K, Δt, η_ve, 90.0, pc1, τc1)
    l2    = line(P, K, Δt, η_ve, 90.0, pc2, τc2)
    l3    = line(P, K, Δt, η_ve,    ψ, pc2, τc2)

    if max(τ - P*sind(φ) - C*cosd(φ) , τ - P - σ_T , - P - (σ_T - δσ_T) ) > 0.0                                                         # check if F_tr > 0
        if τ <= τc1 
            # pressure limiter 
            dqdp = -1.0
            f    = - P - (σ_T - δσ_T) 
            λ̇    = f / (K*Δt)                                                                                                                          # tensile pressure cutoff
            τc   = τ 
            Pc   = P - K*Δt*λ̇*dqdp
            f    = - Pc - (σ_T - δσ_T) 
            domain_pl = 1.0
        elseif τc1 < τ <= l1    
            # corner 1 
            τc = τ - η_ve*(τ - τc1)/(η_ve + ηvp)
            Pc = P - K*Δt*(P - pc1)/(K*Δt + ηvp)
            domain_pl = 2.0
        elseif l1 < τ <= l2            # mode-1
            # tension
            dqdp = -1.0
            dqdτ =  1.0
            f    = τ - P - σ_T 
            λ̇    = f / (K*Δt + η_ve + ηvp) 
            τc   = τ - η_ve*λ̇*dqdτ
            Pc   = P - K*Δt*λ̇*dqdp
            domain_pl = 3.0 
        elseif l2< τ <= l3 # 2nd corner
            # corner 2
            τc = τ - η_ve*(τ - τc2)/(η_ve + ηvp)
            Pc = P - K*Δt*(P - pc2)/(K*Δt + ηvp)
            domain_pl = 4.0
        elseif l3 < τ  
            # Drucker-Prager                                                              # Drucker Prager
            dqdp = -sind(ψ)
            dqdτ =  1.0
            f    = τ - P*sind(φ) - C*cosd(φ) 
            λ̇    = f / (K*Δt*sind(φ)*sind(ψ) + η_ve + ηvp) 
            τc   = τ - η_ve*λ̇*dqdτ
            Pc   = P - K*Δt*λ̇*dqdp
            domain_pl = 5.0 
        end
    end

    return τc, Pc, λ̇
end

function DruckerPrager(τII, P, ηve, comp, β, Δt, C, cosϕ, sinϕ, sinψ, ηvp)
    λ̇    = 0.0
    F    = τII - C*cosϕ - P*sinϕ - λ̇*ηvp
    if F > 1e-10
        λ̇    = F / (ηve + ηvp + comp*Δt/β*sinϕ*sinψ) 
        τII -= λ̇ * ηve
        P   += comp * λ̇*sinψ*Δt/β
        F    = τII - C*cosϕ - P*sinϕ - λ̇*ηvp
        (F>1e-10) && error("Failed return mapping")
        # (τII<0.0) && error("Plasticity without condom")
    end
    return τII, P, λ̇
end

function Tensile(τII, P, ηve, comp, β, Δt, σ_T, ηvp)
    λ̇    = 0.0
    F    = τII - σ_T - P - λ̇*ηvp
    if F > 1e-10
        λ̇    = F / (ηve + ηvp + comp*Δt/β) 
        τII -= λ̇ * ηve
        P   += comp * λ̇*Δt/β
        F    = τII - σ_T - P - λ̇*ηvp
        (F>1e-10) && error("Failed return mapping")
        (τII<0.0) && error("Plasticity without condom")
    end
    return τII, P, λ̇
end

function StrainRateTrial(τII, G, Δt, B, n)
    ε̇II_vis   = B.*τII.^n 
    ε̇II_trial = ε̇II_vis + τII/(2*G*Δt)
    return ε̇II_trial
end

function LocalRheology(ε̇, materials, phases, Δ)

    eps0 = 0.0*1e-17

    # Effective strain rate & pressure
    ε̇II  = sqrt.( (ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1]-ε̇[2])^2)/2 + ε̇[3]^2 ) + eps0
    P    = ε̇[4]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[phases]
    η0   = materials.η0[phases]
    B    = materials.B[phases]
    G    = materials.G[phases]
    C    = materials.C[phases]

    ϕ    = materials.ϕ[phases]
    ψ    = materials.ψ[phases]

    ηvp  = materials.ηvp[phases]
    sinψ = materials.sinψ[phases]    
    sinϕ = materials.sinϕ[phases] 
    cosϕ = materials.cosϕ[phases]    

    β    = materials.β[phases]
    comp = materials.compressible

    # Initial guess
    η    = (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
    ηvep = inv(1/η + 1/(G*Δ.t))
    τII  = 2*ηvep*ε̇II

    # Visco-elastic powerlaw
    for it=1:20
        r      = ε̇II - StrainRateTrial(τII, G, Δ.t, B, n)
        # @show abs(r)
        (abs(r)<ϵ) && break
        ∂ε̇II∂τII = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial, τII, G, Δ.t, B, n)
        ∂τII∂ε̇II = inv(∂ε̇II∂τII[1])
        τII     += ∂τII∂ε̇II*r
    end
    isnan(τII) && error()
 
    # # Viscoplastic return mapping
    λ̇ = 0.
    if materials.plasticity === :DruckerPrager
        τII, P, λ̇ = DruckerPrager(τII, P, ηvep, comp, β, Δ.t, C, cosϕ, sinϕ, sinψ, ηvp)
    elseif materials.plasticity === :tensile
        τII, P, λ̇ = Tensile(τII, P, ηvep, comp, β, Δ.t, materials.σT[phases], ηvp)
    elseif materials.plasticity === :Kiss2023
        τII, P, λ̇ = Kiss2023(τII, P, ηvep, comp, β, Δ.t, C, ϕ, ψ, ηvp, materials.σT[phases], materials.δσT[phases], materials.P1[phases], materials.τ1[phases], materials.P2[phases], materials.τ2[phases])
    end

    # Effective viscosity
    ηvep = τII/(2*ε̇II)

    return ηvep, λ̇, P
end

function LocalRheology_phase_ratios(ε̇, materials, phase_ratios, Δ)

    nphases = length(materials.n)

    eps0 = 1e-17

    # Effective strain rate & pressure
    ε̇II  = sqrt.( (ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1]-ε̇[2])^2)/2 + ε̇[3]^2 ) #+ eps0
    P    = ε̇[4]

    η_average, λ̇_average, P_average = 0.0, 0.0, 0.0

    for phases = 1:nphases

        # Parameters
        ϵ    = 1e-10 # tolerance
        n    = materials.n[phases]
        η0   = materials.η0[phases]
        B    = materials.B[phases]
        G    = materials.G[phases]
        C    = materials.C[phases]

        ϕ    = materials.ϕ[phases]
        ψ    = materials.ψ[phases]

        ηvp  = materials.ηvp[phases]
        sinψ = materials.sinψ[phases]    
        sinϕ = materials.sinϕ[phases] 
        cosϕ = materials.cosϕ[phases]    

        β    = materials.β[phases]
        comp = materials.compressible

        # Initial guess
        η    = (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
        ηvep = inv(1/η + 1/(G*Δ.t))
        τII  = 2*ηvep*ε̇II

        # Visco-elastic powerlaw
        for it=1:20
            r      = ε̇II - StrainRateTrial(τII, G, Δ.t, B, n)
            # @show abs(r)
            (abs(r)<ϵ) && break
            ∂ε̇II∂τII = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial, τII, G, Δ.t, B, n)
            ∂τII∂ε̇II = inv(∂ε̇II∂τII[1])
            τII     += ∂τII∂ε̇II*r
        end
        isnan(τII) && error()
    
        # # Viscoplastic return mapping
        λ̇ = 0.
        if materials.plasticity === :DruckerPrager
            τII, P, λ̇ = DruckerPrager(τII, P, ηvep, comp, β, Δ.t, C, cosϕ, sinϕ, sinψ, ηvp)
        elseif materials.plasticity === :tensile
            τII, P, λ̇ = Tensile(τII, P, ηvep, comp, β, Δ.t, materials.σT[phases], ηvp)
        elseif materials.plasticity === :Kiss2023
            τII, P, λ̇ = Kiss2023(τII, P, ηvep, comp, β, Δ.t, C, ϕ, ψ, ηvp, materials.σT[phases], materials.δσT[phases], materials.P1[phases], materials.τ1[phases], materials.P2[phases], materials.τ2[phases])
        end

        # Effective viscosity
        ηvep = τII/(2*ε̇II)

        # Phase averaging
        η_average += phase_ratios[phases] * ηvep
        P_average += phase_ratios[phases] * P
        λ̇_average += phase_ratios[phases] * λ̇
        
    end

    return η_average, λ̇_average, P_average
end

function StressVector!(ε̇, materials, phases, Δ) 
    η, λ̇, P = LocalRheology(ε̇, materials, phases, Δ)
    τ       = @SVector([2 * η * ε̇[1],
                        2 * η * ε̇[2],
                        2 * η * ε̇[3],
                                  P])
    return τ, η, λ̇
end

function StressVector_phase_ratios!(ε̇, materials, phases, Δ) 
    η, λ̇, P = LocalRheology_phase_ratios(ε̇, materials, phases, Δ)
    τ       = @SVector([2 * η * ε̇[1],
                        2 * η * ε̇[2],
                        2 * η * ε̇[3],
                                  P])
    return τ, η, λ̇
end

