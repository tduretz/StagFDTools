using GLMakie, Enzyme, LinearAlgebra

# Intends to implement constitutive updates as in RheologicalCalculator

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + x[3]^2) 

function residual_single_phase!(x, ε̇II_eff, divV, P0, p)
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

function single_phase_return_mapping()

    # Kinematics
    ε̇    = [0.1, -0.1, 0]
    divV = -0.05   

    # Initial conditions
    P    = 0.0
    τ    = [0.0, -0.0, 0]

    # Parameters
    nt = 44
    params = (
        G     = 1.0,
        K     = 3.0,
        C     = 1.0,
        ϕ     = 35.0,
        ψ     = 35.0*0,
        ηvp   = 10.0*0,
        Δt    = 1.0,
    )  

    # Solution array return mapping
    x = zeros(3)

    # Probes
    probes = (
        τ = zeros(nt),
        P = zeros(nt),
        t = zeros(nt),
        λ̇ = zeros(nt),
    )

    # Time loop
    for it=1:nt

        @info "Step $(it)"

        # Old guys
        P0 = P
        τ0 = τ
        
        # Invariants
        ε̇_eff   = ε̇ + τ0/(2*params.G*params.Δt)
        ε̇II_eff = invII(ε̇_eff) 
        τII     = invII(τ)

        # Rheology update
        x[1], x[2], x[3] = τII, P, 0.0 
        for iter=1:10
            J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, residual_single_phase!, x, Const(ε̇II_eff), Const(divV), Const(P0), Const(params))
            # display(J.derivs[1])
            x .-= J.derivs[1]\J.val
            @show norm(J.val)
            if norm(J.val)<1e-10
                break
            end
        end

        # Recompute components
        τII, P, λ̇ = x[1], x[2], x[3]
        τ   .= ε̇_eff .* τII./ε̇II_eff
        @show x[1], invII(τ)

        # Probes
        probes.t[it] = it*params.Δt
        probes.τ[it] = τII
        probes.P[it] = P
        probes.λ̇[it] = λ̇ 
    end

    function figure()
        fig = Figure(fontsize = 20, size = (800, 800) )     
        ax1 = Axis(fig[1,1], title="Deviatoric stress",  xlabel=L"$t$ [yr]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20)
        scatter!(ax1, probes.t, probes.τ)
        ax2 = Axis(fig[2,1], title="Pressure",  xlabel=L"$t$ [yr]",  ylabel=L"$P$ [MPa]", xlabelsize=20, ylabelsize=20)
        scatter!(ax2, probes.t, probes.P)
        ax3 = Axis(fig[3,1], title="Plastic multiplier",  xlabel=L"$P$ [MPa]",  ylabel=L"$\dot{\lambda}$ [1/s]", xlabelsize=20, ylabelsize=20)    
        scatter!(ax3, probes.t, probes.λ̇)
        ax4 = Axis(fig[4,1], title="Invariant space",  xlabel=L"$P$ [MPa]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20)                
        P1 = LinRange( extrema(probes.P)..., 100)
        τ1 = LinRange( extrema(probes.τ)..., 100)
        F  =  τ1' .- params.C*cosd(params.ϕ) .- P1*sind(params.ϕ)
        contour!(ax4, P1, τ1,  F, levels =[0.])
        scatter!(ax4, probes.P, probes.τ)
        display(fig)
    end
    with_theme(figure, theme_latexfonts())

end

single_phase_return_mapping()