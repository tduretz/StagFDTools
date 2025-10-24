using Enzyme, LinearAlgebra, StaticArrays, CairoMakie

# Intends to implement constitutive updates as in RheologicalCalculator

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + x[3]^2) 

function residual_single_phase(x, ε̇II_eff, P_trial, divV, P0, p)
    G, K, C, ϕ, ψ, ηvp, Δt = p.G, p.K, p.C, p.ϕ, p.ψ, p.ηvp, p.Δt
     τII, P, λ̇ = x[1], x[2], x[3]
    ηe  = G*Δt
    χe  = K*Δt
    f   = τII  - C*cosd(ϕ) - P*sind(ϕ)
    return @SVector([ 
        ε̇II_eff  -  (τII)/2/ηe - λ̇/2,
        divV     + (P - P0)/χe - λ̇*sind(ψ),
        (f - ηvp*λ̇)
    ])
end

function residual_single_phase_trial(x, ε̇II_eff, P_trial, divV, P0, p)
    G, K, C, ϕ, ψ, ηvp, Δt = p.G, p.K, p.C, p.ϕ, p.ψ, p.ηvp, p.Δt
     τII, P, λ̇ = x[1], x[2], x[3]
    ηe  = G*Δt
    χe  = K*Δt
    f   = τII  - C*cosd(ϕ) - P*sind(ϕ)
    return @SVector([ 
        ε̇II_eff  -  (τII)/2/ηe - λ̇/2,
        P - (P_trial + λ̇*sind(ψ)*χe),
        (f - ηvp*λ̇)
    ])
end

function StressVector(σ, τ0, P0, p)

    ε̇_eff = σ[1:3]
    divV  = σ[4]

    # Rheology update
    G, K, C, ϕ, ψ, ηvp, Δt = p.G, p.K, p.C, p.ϕ, p.ψ, p.ηvp, p.Δt
    ηe      = G*Δt
    χe      = K*Δt
    ε̇II_eff = invII(ε̇_eff) 

    # 1 - Trial 
    τ   = 2*ηe*ε̇_eff
    P   = P0 - χe*divV
    τII = invII(τ)
    f   = τII  - C*cosd(ϕ) - P*sind(ϕ)

    P_trial = P

    if f>0
        # e - Correction 
        x = @MVector([τII, P, 0.0])
        for iter=1:10
            J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, residual_single_phase, x, Const(ε̇II_eff), Const(P_trial), Const(divV), Const(P0), Const(p))
            x .-= J.derivs[1]\J.val
            # @show norm(J.val)
            if norm(J.val)<1e-10
                break
            end
        end
        # Recompute components
        τII, P, λ̇ = x[1], x[2], x[3]
    end
    τ = ε̇_eff .* τII./ε̇II_eff
    return @SVector([τ[1], τ[2], τ[3], P])
end


function StressVector_trial(σ, divV, τ0, P0, p)

    ε̇_eff = σ[1:3]
    P     = σ[4]

    # Rheology update
    G, K, C, ϕ, ψ, ηvp, Δt = p.G, p.K, p.C, p.ϕ, p.ψ, p.ηvp, p.Δt
    ηe      = G*Δt
    # χe      = K*Δt
    ε̇II_eff = invII(ε̇_eff) 

    # 1 - Trial 
    τ   = 2*ηe*ε̇_eff
    # P   = P0 - χe*divV
    τII = invII(τ)
    f   = τII  - C*cosd(ϕ) - P*sind(ϕ)

    P_trial = P

    if f>0
        # e - Correction 
        x = @MVector([τII, P, 0.0])
        for iter=1:10
            J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, residual_single_phase_trial, x, Const(ε̇II_eff), Const(P_trial), Const(divV), Const(P0), Const(p))
            x .-= J.derivs[1]\J.val
            @show norm(J.val)
            if norm(J.val)<1e-10
                break
            end
        end
        # Recompute components
        τII, P, λ̇ = x[1], x[2], x[3]
    end
    τ = ε̇_eff .* τII./ε̇II_eff
    return @SVector([τ[1], τ[2], τ[3], P])
end


function single_phase_return_mapping()

    sc = (σ = 3e10, L = 1e3, t = 1e10)

    # Parameters
    nt = 200
    params = (
        G     = 1e10/sc.σ,
        K     = 2e10/sc.σ,
        C     = 3e7/sc.σ,
        ϕ     = 30.0*1,
        ψ     = -10.0*1,
        ηvp   = 1e20*0/(sc.σ*sc.t),
        Δt    = 1e10/sc.t,
    )  

    # Kinematics
    ε̇bg  = 5e-15*sc.t
    ε̇    = @SVector([ε̇bg, -ε̇bg, ε̇bg/4])
    divV = -0.00*sc.t   

    # Initial conditions
    P      = 0.0/sc.σ
    τ_DP   = 0*(sind(params.ϕ)*P + params.C*cosd(params.ϕ) )  
    τxx_DP = τ_DP*ε̇[1]/abs(ε̇bg)
    τyy_DP = τ_DP*ε̇[2]/abs(ε̇bg)
    τ      = @SVector([τxx_DP, τyy_DP, 0])

    K, G = params.K, params.G
    De   = @SMatrix([K+4/3*G K-2/3*G 0.0; K-2/3*G K+4/3*G 0.0; 0.0 0.0 2*G])

    # Probes
    probes = (
        τ    = zeros(nt),
        P    = zeros(nt),
        t    = zeros(nt),
        λ̇    = zeros(nt),
        detA = zeros(nt),
        θ    = zeros(nt),
    )

    # Time loop
    for it=1:nt

        @info "Step $(it)"

        # Old guys
        P0 = P
        τ0 = τ

        # ----------------------------
        
        # Invariants
        ε̇_eff = ε̇ + τ0/(2*params.G*params.Δt)
        ϵ̇     = @SVector([ε̇_eff[1], ε̇_eff[2], ε̇_eff[3], divV])
        σ     = StressVector(ϵ̇, τ0, P0, params)
        τ, P  = σ[1:3], σ[4]

        # Consistent tangent
        J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector, ϵ̇, Const(τ0), Const(P0), Const(params))
        Dep = J.derivs[1]

        # Invariants
        P_trial = P0 - params.K*params.Δt*divV
        ε̇_eff = ε̇ + τ0/(2*params.G*params.Δt)
        ϵ̇     = @SVector([ε̇_eff[1], ε̇_eff[2], ε̇_eff[3], P_trial])
        σ     = StressVector_trial(ϵ̇, divV, τ0, P0, params)
        τ, P  = σ[1:3], σ[4]

        # Consistent tangent
        J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector_trial, ϵ̇, Const(divV), Const(τ0), Const(P0), Const(params))

        χe  = params.K*params.Δt
        @show χe
        Cep =  @SMatrix([ 1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -χe])
        Dep1 = J.derivs[1] * Cep

        display( norm(Dep .- Dep1) )

        # ----------------------------
     
        # Bifurcation analysis
        Te = @SMatrix([2/3 -1/3 0; -1/3 2/3 0; 0 0 1; 1 1 0 ])
        Ts = @SMatrix([ 1 0 0 -1; 0 1 0 -1; 0 0 1 0])
        𝐃ep = Ts * Dep * Te 

        θ    = LinRange(-90, 90, 180)
        detA = zeros(size(θ))

        for i in eachindex(θ)
            n = @SVector([cosd(θ[i]), sind(θ[i])])
            𝐧 = @SVector([n[1], n[2], 2*n[1]*n[2]])
            detA[i] = det(𝐧'*𝐃ep*𝐧)
        end

        @show 180/4 - (params.ϕ + params.ψ)/4

        # Probes
        probes.t[it]    = it*params.Δt
        probes.τ[it]    = invII(τ)
        probes.P[it]    = P
        probes.detA[it] = detA[argmin(detA)]
        probes.θ[it]    = abs(θ[argmin(detA)])
    end

    if minimum(probes.detA) <0
        bif_ind = findfirst(probes.detA .< 0)
    else
        bif_ind = 1
    end

    @info probes.θ[bif_ind]

    fig = Figure(size=(500, 500))
    ax  = Axis(fig[1,1], title=L"$$Det. acoustic tensor", xlabel=L"$t$", ylabel=L"$\tau$")
    lines!(  ax, probes.t*sc.t, probes.τ*sc.σ )
    ax  = Axis(fig[2,1], title=L"$$Det. acoustic tensor", xlabel=L"$t$", ylabel=L"$P$")
    lines!(  ax, probes.t*sc.t, probes.P*sc.σ )
    ax  = Axis(fig[3,1], title=L"$$Det. acoustic tensor", xlabel=L"$t$", ylabel=L"$\det{\mathbf{A}}$")
    lines!(  ax, probes.t*sc.t, probes.detA )
    ax  = Axis(fig[4,1], title=L"$\theta$", xlabel=L"$t$", ylabel=L"$\theta$")
    lines!(  ax, probes.t*sc.t, probes.θ )
    display(fig)

end

single_phase_return_mapping()