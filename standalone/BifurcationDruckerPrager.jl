using Enzyme, LinearAlgebra, StaticArrays

# Intends to implement constitutive updates as in RheologicalCalculator

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + x[3]^2) 

function residual_single_phase(x, ε̇II_eff, divV, P0, p)
    G, K, C, ϕ, ψ, ηvp, Δt = p.G, p.K, p.C, p.ϕ, p.ψ, p.ηvp, p.Δt
    eps   = -1e-13
    ηe    = G*Δt
    χe    = K*Δt
    τII, P, λ̇ = x[1], x[2], x[3]
    f      = τII  - C*cosd(ϕ) - P*sind(ϕ)
    return @SVector([ 
        ε̇II_eff  -  (τII)/2/ηe - λ̇*(f>=eps),
        divV     + (P - P0)/χe - λ̇*sind(ψ)*(f>=eps),
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps)
    ])
end

function StressVector(σ, τ0, P0, params)

    ε̇_eff = σ[1:3]
    divV  = σ[4]

    ε̇II_eff = invII(ε̇_eff) 
    τII     = invII(τ0)

    # Rheology update
    x = @MVector([τII, P0, 0.0])

    for iter=1:10
        J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, residual_single_phase, x, Const(ε̇II_eff), Const(divV), Const(P0), Const(params))
        x .-= J.derivs[1]\J.val
        @show norm(J.val)
        if norm(J.val)<1e-10
            break
        end
    end

    # Recompute components
    τII, P, λ̇ = x[1], x[2], x[3]
    τ = ε̇_eff .* τII./ε̇II_eff
    return @SVector([τ[1], τ[2], τ[3], P])
end


function single_phase_return_mapping()

    # Parameters
    nt = 400
    params = (
        G     = 1e10,
        K     = 1e11,
        C     = 1e6,
        ϕ     = 35.0,
        ψ     = -0.0,
        ηvp   = 0.0*0,
        Δt    = 1e8,
    )  

    # Kinematics
    ε̇bg  = -1e-14
    ε̇    = @SVector([ε̇bg, -ε̇bg, 0])
    divV = -0.00   

    # Initial conditions
    P      = 0.0
    τ_DP   = (sind(params.ϕ)*P + params.C*cosd(params.ϕ) )  
    τxx_DP = τ_DP*ε̇[1]/abs(ε̇bg)
    τyy_DP = τ_DP*ε̇[2]/abs(ε̇bg)
    τ      = @SVector([τxx_DP, τyy_DP, 0])

    K, G = params.K, params.G
    De     = @SMatrix([K+4/3*G K-2/3*G 0.0; K-2/3*G K+4/3*G 0.0; 0.0 0.0 2*G])

    # Probes
    probes = (
        τ = zeros(nt),
        P = zeros(nt),
        t = zeros(nt),
        λ̇ = zeros(nt),
    )

    θ = LinRange(-90, 90, 180)
    r = zeros(size(θ))

    # Time loop
    for it=1:nt

        @info "Step $(it)"

        # Old guys
        P0 = P
        τ0 = τ
        
        # Invariants
        ε̇_eff = ε̇ + τ0/(2*params.G*params.Δt)
        ϵ̇     = @SVector([ε̇_eff[1], ε̇_eff[2], ε̇_eff[3], divV])
        σ     = StressVector(ϵ̇, τ0, P0, params)
        τ, P  = σ[1:3], σ[4]

        # Consistent tangent
        J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector, ϵ̇, Const(τ0), Const(P0), Const(params))
        Dep = J.derivs[1]

        display(Dep)
        # @show det(Dep)

        # display(De)

        Te  = @SMatrix([2/3 -1/3 0; -1/3 2/3 0; 0 0 1; 1 1 0 ])
        Ts  = @SMatrix([ 1 0 0 -1; 0 1 0 -1; 0 0 1 0])
        𝐃ep = Ts * Dep * Te 

        for i in eachindex(θ)
            n = @SVector([cosd(θ[i]), sind(θ[i])])
            𝐧 = @SVector([n[1], n[2], 2*n[1]*n[2]])
            r[i] = det(𝐧'*𝐃ep*𝐧)
        end

        # Probes
        probes.t[it] = it*params.Δt
        probes.τ[it] = invII(τ)
        probes.P[it] = P

        if minimum(r) < 0
            @info "Bifurcation"
            break
        end
    end

    ii = argmin(r)
    @show (θ[ii])
    @show 180/4 - (params.ϕ + params.ψ)/4
    @show params.ϕ - params.ψ/2
    @show asind( (sind(params.ϕ) + sind(params.ψ))/2 )

    fig = Figure(size=(500,500))
    ax  = Axis(fig[1,1], title=L"$$Det. acoustic tensor", xlabel=L"\theta", ylabel=L"\det{\mathbf{A}}")
    lines!( ax, θ, r )
    scatter!( ax, θ[ii], r[ii] ) 
    
    display(fig)

    # p1 = plot(probes.t, probes.τ, xlabel="t", ylabel="τ")
    # p2 = plot(probes.t, probes.P, xlabel="t", ylabel="P")
    # p3 = plot(probes.t, probes.λ̇, xlabel="t", ylabel="λ̇")
    # plot(p1, p2, p3, layout=(3, 1))

end

single_phase_return_mapping()