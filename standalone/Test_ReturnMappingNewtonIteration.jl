function F_DP_hyperbolic_v1(x, C, σT, cosΨ, sinΨ, ηvp)  
    ϵ = -1e-13
    τ = x[1] 
    P = x[2] 
    λ̇ = x[3]
    F = sqrt( (τ- 0*λ̇*ηvp)^2 + (C * cosΨ - σT*sinΨ)^2) - (P * sinΨ + C * cosΨ) 
    return (F - λ̇*ηvp)*(F>=ϵ) + (F<ϵ)*λ̇*ηvp
end

function F_shear(x, τ_trial, ε̇_eff, ηve, C, σT, cosΨ, sinΨ, ηvp)
    τ    = x[1]
    λ̇    = x[3]
    ∂Q∂σ = Enzyme.gradient(Enzyme.Forward, F_DP_hyperbolic_v1, x, Const(C), Const(σT), Const(cosΨ), Const(sinΨ), Const(ηvp))
    # return ε̇_eff -  τ/2/ηve  - λ̇/2*∂Q∂σ[1][1]
    return τ - τ_trial + ηve*λ̇*∂Q∂σ[1][1]
end  

function F_vol(x, P_trial, Dkk, P0, K, Δt, C, σT, cosΨ, sinΨ, ηvp)
    P    = x[2]
    λ̇    = x[3]
    ∂Q∂σ = Enzyme.gradient(Enzyme.Forward, F_DP_hyperbolic_v1, x, Const(C), Const(σT), Const(cosΨ), Const(sinΨ), Const(ηvp))
    # return Dkk + (P - P0)/K/Δt + λ̇*∂Q∂σ[1][2]
    return P - P_trial + K*Δt*λ̇*∂Q∂σ[1][2]
end  

function RheologyResidual(x, τ_trial, ε̇_eff, P_trial, Dkk, P0, ηve, K, Δt, C, σT, cosΨ, sinΨ, cosϕ, sinϕ, ηvp)
    return @SVector([
        F_shear(x, τ_trial, ε̇_eff, ηve, C, σT, cosΨ, sinΨ, ηvp),
        F_vol(x, P_trial, Dkk, P0, K, Δt, C, σT, cosΨ, sinΨ, ηvp),
        F_DP_hyperbolic_v1(x,  C, σT, cosϕ, sinϕ, ηvp),
        # F_DP_v1(x,  C, σT, cosϕ, sinϕ, ηvp),
    ])
end

function bt_line_search(Δx, J, x, r, params; α = 1.0, ρ = 0.5, c = 1.0e-4, α_min = 1.0e-8)
    # Borrowed from RheologicalCalculator
    perturbed_x = @. x + α * Δx
    perturbed_r = RheologyResidual(x, params...)

    J_times_Δx = - J * Δx
    while sqrt(sum(perturbed_r .^ 2)) > sqrt(sum((r + (c * α * (J_times_Δx))) .^ 2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_x = @. x + α * Δx
        perturbed_r = RheologyResidual(x, params...)
    end
    return α
end

function DruckerPragerHyperbolic_v1(τII, P, ε̇_eff, Dkk, P0, ηve, β, Δt, C, cosϕ, sinϕ, cosΨ, sinΨ, σT, ηvp)
    
    tol     = 1e-9
    λ̇       = 0.0
    K       = 1/β
    τ_trial = τII
    P_trial = P
    itermax = 100

    x    = @MVector([τII, P, λ̇])
    αvec = @SVector([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    Fvec = @MVector(zeros(length(αvec)))

    R    = RheologyResidual(x, τ_trial, ε̇_eff, P_trial, Dkk, P0, ηve, K, Δt, C, σT, cosΨ, sinΨ, cosϕ, sinϕ, ηvp)
    nR   = abs(R[3])#norm(R)
    iter, nR0 = 0, nR
    R0 = copy(R)

    # @show R

    while nR>tol && (nR/nR0)>tol && iter<itermax

        iter += 1
        x0    = copy(x)
        J     = Enzyme.jacobian(Enzyme.ForwardWithPrimal, RheologyResidual, x, Const(τ_trial), Const(ε̇_eff), Const(P_trial), Const(Dkk), Const(P0), Const(ηve), Const(K), Const(Δt), Const(C), Const(σT), Const(cosΨ), Const(sinΨ), Const(cosϕ), Const(sinϕ), Const(ηvp))
        δx    = - J.derivs[1] \ J.val
        nR    = abs(J.val[3])

        # params = (τ_trial, ε̇_eff, P_trial, Dkk, P0, ηve, K, Δt, C, σT, cosΨ, sinΨ, cosϕ, sinϕ, ηvp)
        # α = bt_line_search(δx, J.derivs[1], x0, J.val, params)
        # x .= x0 .+  α*δx
        # @show iter, nR,  α

        for ils in eachindex(αvec)
            x .= x0 .+  αvec[ils]δx
            R = RheologyResidual(x, τ_trial, ε̇_eff, P_trial, Dkk, P0, ηve, K, Δt, C, σT, cosΨ, sinΨ, cosϕ, sinϕ, ηvp)           
            Fvec[ils] = norm(R) 
        end
        ibest = argmin(Fvec)
        x .= x0 .+  αvec[ibest]*δx
        @show iter, nR, nR/nR0, αvec[ibest]

        if isnan(norm(δx))
            @show R0
            @show J.val
            @show J.derivs[1]
            @show δx
            @show iter, nR, nR/nR0,  αvec[ibest]
        end
    end

    # if iter == itermax && (nR>tol && (nR/nR0)>tol )
    #     R    = RheologyResidual(x, τ_trial, ε̇_eff, P_trial, Dkk, P0, ηve, K, Δt, C, σT, cosΨ, sinΨ, cosϕ, sinϕ, ηvp)
    #     @show τ_trial, P_trial
    #     @show τII, P, ε̇_eff, Dkk, P0, ηve, β, Δt, C, cosϕ, sinϕ, cosΨ, sinΨ, σT, ηvp
    #     @show R0
    #     @show R
    #     @show x
    #     error("Failed return mapping")
    # end

    if  isnan(x[3])
        @show R, x
        error()
    end

    return x[1], x[2], x[3]
end


let 
    # (τII, P, ε̇_eff, Dkk, P0, ηve, β, Δt, C, cosϕ, sinϕ, cosΨ, sinΨ, σT, ηvp) = (0.00833165063026724, -0.08546335722262118, 4.16582531513362, 8.058144704185978, -0.004888154499330499, 0.001, 0.01, 0.0001, 0.01, 0.8191520442889918, 0.573576436351046, 0.573576436351046, 0.573576436351046, 0.005, 0.0004)
    (τII, P, ε̇_eff, Dkk, P0, ηve, β, Δt, C, cosϕ, sinϕ, cosΨ, sinΨ, σT, ηvp) = (0.00023317824856743376, -0.00031113830502547095, 0.0013990694914395793, 0.00022117508879222406, -0.00016408049817799464, 0.08333333333125, 1.5, 1.0, 0.001, 0.8660254037844386, 0.5, 0.17364817766693036, 0.17364817766693036, 0.00016666666666666666, 0.33333333333333337)
    (τII, P, ε̇_eff, Dkk, P0, ηve, β, Δt, C, cosϕ, sinϕ, cosΨ, sinΨ, σT, ηvp) = (0.01870740409269564, 0.02647058984524955, 9.35370204634782, 1.101209787325194, 0.027047835685611955, 0.001, 0.01, 0.0001, 0.01, 0.9961946980917455, 0.08715574274765818, 0.5, 0.5, 0.005, 0.0001)
    # ηvp = 1*ηvp/10
    
    τII_corr, P_corr, λ̇ = DruckerPragerHyperbolic_v1(τII, P, ε̇_eff, Dkk, P0, ηve, β, Δt, C, cosϕ, sinϕ, cosΨ, sinΨ, σT, ηvp)

    ftsz = 20
    P_ax       = LinRange(-σT,  3/1e3, 100)
    τ_ax_rock = @. sqrt(sinϕ*(P_ax + σT)*(2*C*cosϕ + P_ax*sinϕ - sinϕ*σT))


    P_ax_vp       = LinRange(-σT - ηvp*λ̇/sinϕ,  3/1e3, 100)
    τ_ax_rock_vp = @. sqrt( Complex((P_ax_vp*sinϕ + ηvp*λ̇  + sinϕ*σT)*(2*C*cosϕ + P_ax_vp*sinϕ + ηvp*λ̇ - sinϕ*σT)) )

    τ_ax_rock_vp = real.(τ_ax_rock_vp)

    fig = Figure(size=(1000, 1000)) 
    ax  = Axis(fig[1,1], title=L"$$Stress space", xlabel=L"$P$", ylabel=L"$\tau_{II}$", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz, aspect=DataAspect())
    lines!(ax, P_ax, τ_ax_rock, color=:black, linestyle=:dashdot)
    lines!(ax, P_ax_vp, τ_ax_rock_vp, color=:black)
    scatter!(ax, P, τII)
    scatter!(ax, P_corr, τII_corr)
    display(fig)
end