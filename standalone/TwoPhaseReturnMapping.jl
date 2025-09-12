using GLMakie, Enzyme, LinearAlgebra#, ForwardDiff

# Intends to implement constitutive updates as in RheologicalCalculator

invII(x) = sqrt(1/2*x[1]^2 + 1/2*x[2]^2 + x[3]^2) 

function residual_two_phase(x, ε̇II_eff, divV, divqD, Pt0, Pf0, ϕ, p)
    G, Kϕ, Ks, Kf, C, ϕ, ψ, ηvp, Δt = p.G, p.Kϕ, p.Ks, p.Kf, p.C, p.ϕ, p.ψ, p.ηvp, p.Δt
    eps   = -1e-13
    ηe    = G*Δt
    χe    = Kϕ*Δt
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]
    f      = τII  - C*cosd(ϕ) - (Pt - Pf)*sind(ϕ)
    return [ 
        ε̇II_eff  -  (τII)/2/ηe - λ̇*(f>=eps),
        divV     + (Pt - Pt0)/χe    - λ̇*sind(ψ)*(f>=eps),
        divqD    - (Pf - Pf0)/Kf/Δt + λ̇*sind(ψ)*(f>=eps),
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps)
    ]
end

function StressVector(ϵ̇, τ0, Pt0, Pf0, ϕ, params)

    ε̇_eff = ϵ̇[1:3]
    divV  = ϵ̇[4]
    divqD = ϵ̇[5]

    ε̇II_eff = invII(ε̇_eff) 
    τII     = invII(τ0)

    # Rheology update
    x = [τII, Pt0, Pf0, 0.0]

    for iter=1:10
        J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, residual_two_phase, x, Const(ε̇II_eff), Const(divV), Const(divqD), Const(Pt0), Const(Pf0), Const(ϕ), Const(params))
        # display(J.derivs[1])
        x .-= J.derivs[1]\J.val
        @show norm(J.val)
        if norm(J.val)<1e-10
            break
        end
    end

    # Recompute components
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]
    τ = ε̇_eff .* τII./ε̇II_eff
    return [τ[1], τ[2], τ[3], Pt, Pf], λ̇
end


function single_phase_return_mapping()

    # Kinematics
    ε̇     = [0.1, -0.1, 0]
    divV  = -0.05   
    divqD = 0.04 

    # Initial conditions
    Pt   = 0.0
    Pf   = 0.0  
    τ    = [0.0, -0.0, 0]
    ϕ    = 0.01 

    # Parameters
    nt = 10
    params = (
        G     = 1.0,
        Kϕ    = 3.0,
        Ks    = 3.0,
        Kf    = 3.0,
        C     = 1.0,
        ϕ     = 35.0,
        ψ     = 35.0*0,
        ηvp   = 10.0*0,
        Δt    = 1.0,
    )  

    # Probes
    probes = (
        τ  = zeros(nt),
        Pt = zeros(nt),
        Pf = zeros(nt),
        Pe = zeros(nt),
        t  = zeros(nt),
        λ̇  = zeros(nt),
    )

    # Time loop
    for it=1:nt

        @info "Step $(it)"

        # Old guys
        Pt0 = Pt
        Pf0 = Pf
        τ0  = τ
        
        # Invariants
        ε̇_eff     = ε̇ + τ0/(2*params.G*params.Δt)
        ϵ̇         = [ε̇_eff[1], ε̇_eff[2], ε̇_eff[3], divV, divqD]
        σ, λ̇      = StressVector(ϵ̇, τ0, Pt0, Pf0, ϕ, params)
        τ, Pt, Pf = σ[1:3], σ[4], σ[5]

        # # Consistent tangent
        # J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector1, ϵ̇, Const(τ0), Const(P0), Const(params))
        # display(J.derivs[1])

        # Probes
        probes.t[it]  = it*params.Δt
        probes.τ[it]  = invII(τ)
        probes.Pt[it] = Pt
        probes.Pf[it] = Pf
        probes.Pe[it] = Pt - Pf
        probes.λ̇[it]  = λ̇ 
    end

    function figure()
        fig = Figure(fontsize = 20, size = (800, 800) )     
        ax1 = Axis(fig[1,1], title="Deviatoric stress",  xlabel=L"$t$ [yr]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20)
        scatter!(ax1, probes.t, probes.τ)
        ax2 = Axis(fig[2,1], title="Pressure",  xlabel=L"$t$ [yr]",  ylabel=L"$P$ [MPa]", xlabelsize=20, ylabelsize=20)
        scatter!(ax2, probes.t, probes.Pt)
        scatter!(ax2, probes.t, probes.Pf)
        ax3 = Axis(fig[3,1], title="Plastic multiplier",  xlabel=L"$t$ [yr]",  ylabel=L"$\dot{\lambda}$ [1/s]", xlabelsize=20, ylabelsize=20)    
        scatter!(ax3, probes.t, probes.λ̇)
        ax4 = Axis(fig[4,1], title="Invariant space",  xlabel=L"$P$ [MPa]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20)                
        P1 = LinRange( extrema(probes.Pe)..., 100)
        τ1 = LinRange( extrema(probes.τ)..., 100)
        F  =  τ1' .- params.C*cosd(params.ϕ) .- P1*sind(params.ϕ)
        contour!(ax4, P1, τ1,  F, levels =[0.])
        scatter!(ax4, probes.Pe, probes.τ)
        display(fig)
    end
    with_theme(figure, theme_latexfonts())

end

single_phase_return_mapping()