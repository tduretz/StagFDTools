using CairoMakie, LinearAlgebra

smax(A, C, eps) = (A + C + sqrt((A-C)^2) + eps) / 2
hyperbolic(τII, P, C, cosΨ, sinΨ, Pt)  =  sqrt(τII^2 + (C * cosΨ + Pt*sinΨ)^2) - (P * sinΨ + C * cosΨ)
compaction(τII, P, r, Pc) = (P - Pc)^2 + τII^2 - r^2   

function main()

    C=1e6
    cosϕ=cosd(30)
    sinϕ=sind(30)
    cosψ=cosd(5.0)
    sinψ=sind(5.0)
    η_vp=0.0
    Pt=-5e5
    Pc=15e6
    r =50e6

    P_vec  = -1e6:1e5:170e6
    τ_vec  =   -0:1e5:135e6
    eps    = 1e6
    F1     = @. hyperbolic(τ_vec', P_vec, C, cosϕ, sinϕ, Pt)
    F2     = @. compaction(τ_vec', P_vec, r, Pc)
    P_plus = @. max(P_vec, Pc) #* sqrt(P_vec.^2 + eps^2 ) / 2
    F3     = @. smax(hyperbolic(τ_vec', P_vec, C, cosϕ, sinϕ, Pt), compaction(τ_vec', P_plus, r, Pc), eps)
    Q3     = @. smax(hyperbolic(τ_vec', P_vec, C, cosψ, sinψ, Pt), compaction(τ_vec', P_plus, r, Pc), eps)

    fig = Figure(fontsize = 20, size = (800, 800) )
    ax1  = Axis(fig[1,1:2], title="Yield function",  xlabel=L"$P$ [MPa]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20, aspect=DataAspect())
    # contour!(ax1, P_vec/1e6, τ_vec/1e6, F1, levels = [0.01], color = :black)
    # contour!(ax1, P_vec/1e6, τ_vec/1e6, F2, levels = [0.01], color = :black)
    contour!(ax1, P_vec/1e6, τ_vec/1e6, F3, levels = [0.00], color = :black)
    xlims!(ax1, -1, 70)
    ylims!(ax1, 0, 30)

    ax2  = Axis(fig[2,1], title="F",  xlabel=L"$P$ [MPa]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20, aspect=DataAspect())
    heatmap!(ax2, P_vec/1e6, τ_vec/1e6, F3)


    ax3  = Axis(fig[2,2], title="Q",  xlabel=L"$P$ [MPa]",  ylabel=L"$\tau_{II}$ [MPa]", xlabelsize=20, ylabelsize=20, aspect=DataAspect())
    heatmap!(ax3, P_vec/1e6, τ_vec/1e6, Q3)


    @show P_plus


    display(fig)

end

main()