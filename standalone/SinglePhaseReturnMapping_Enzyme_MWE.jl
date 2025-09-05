using Enzyme, LinearAlgebra

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + x[3]^2) 

# Function to be minimized via inner Newton iteration 
function residual_single_phase(x, ε̇II_eff, divV, P0, p)
    G, K, C, ϕ, ψ, ηvp, Δt = p.G, p.K, p.C, p.ϕ, p.ψ, p.ηvp, p.Δt
    eps   = -1e-13
    ηe    = G*Δt
    χe    = K*Δt
    τII, P, λ̇ = x[1], x[2], x[3]
    f      = τII  - C*cosd(ϕ) - P*sind(ϕ)
    return [ 
        ε̇II_eff  -  (τII)/2/ηe - λ̇*(f>=eps),
        divV     + (P - P0)/χe - λ̇*sind(ψ)*(f>=eps),
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps)
    ]
end

# Evaluate the stress vector involving the solution of a non-linear function
function StressVector(ϵ̇, τ0, P0, params)

    ε̇_eff, divV = ϵ̇[1:3],  ϵ̇[4]
    ε̇II_eff, τII = invII(ε̇_eff), invII(τ0)

    # Initialise inner solution array
    x = [τII, P0, 0.0]

    # Newton iteration
    for iter=1:10
        J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, residual_single_phase, x, Const(ε̇II_eff), Const(divV), Const(P0), Const(params))
        x = x .- J.derivs[1]\J.val
        if norm(J.val)<1e-10
            break
        end
    end

    # Recompute tensorial components using invariant scaling
    τII, P, λ̇ = x[1], x[2], x[3]
    τ = ε̇_eff .* x[1]./ε̇II_eff
    return [τ[1], τ[2], τ[3], x[2]], x[3]
end

function single_phase_return_mapping()

    # Kinematics
    ε̇    = [0.1, -0.1, 0]
    divV = -0.05   

    # Initial conditions
    P    = 0.0
    τ    = [0.0, -0.0, 0]

    # Parameters
    params = (
        G     = 1.0,
        K     = 3.0,
        C     = 1.0,
        ϕ     = 35.0,
        ψ     = 35.0*0,
        ηvp   = 10.0*0,
        Δt    = 1.0,
    )  

    # Old guys
    P0 = P
    τ0 = τ
    
    # Invariants
    ε̇_eff = ε̇ + τ0/(2*params.G*params.Δt)
    ϵ̇     = [ε̇_eff[1], ε̇_eff[2], ε̇_eff[3], divV]

    # Consistent tangent operator ≡ ∂σ∂ϵ̇ 
    J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector, ϵ̇, Const(τ0), Const(P0), Const(params))
    display(J.derivs[1])
end

single_phase_return_mapping()