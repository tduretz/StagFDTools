using Enzyme, LinearAlgebra, Plots, SpecialFunctions

function F_nonlinear_corrected(λ̇, τ, P, φ, ψ, C, Ptr, σtr) 

    jac  = Enzyme.gradient(Enzyme.Forward, Q_nonlinear, τ, Const(P), Const(ψ), Const(C), Const(Ptr), Const(σtr))
    dQdτ = jac[1]

    jac  = Enzyme.gradient(Enzyme.Forward, Q_nonlinear,  Const(τ), P, Const(ψ), Const(C), Const(Ptr), Const(σtr))
    dQdP = jac[2]

    φeff    = (1 - (1-sind(φ)) /(1 + exp(-σtr.TS*(P - Ptr.TS))) - (1+sind(φ)) /(1 + exp(-σtr.SC*(P - Ptr.SC)))       )
    dFdτ =  ( 1 /(1 + exp(-σtr.TS*(P - Ptr.TS))) - 1.0 /(1 + exp(-σtr.SC*(P - Ptr.SC)))       )
 

    dQdP = -φeff / sqrt(φeff^2 + dFdτ^2) 
    dQdτ = dFdτ  / sqrt(φeff^2 + dFdτ^2)

    τ = τ - λ̇*dQdτ 
    P = P - λ̇*dQdP
    f = F_nonlinear(τ, P, φ, C, Ptr, σtr)

    return f, τ, P
end

function F_nonlinear(τ, P, φ, C, Ptr, σtr) 
    
    φeff    = (1 - (1-sind(φ)) /(1 + exp(-σtr.TS*(P - Ptr.TS))) - (1+sind(φ)) /(1 + exp(-σtr.SC*(P - Ptr.SC)))       )
    # ψeff    = -(1 - 0.9 /(1 + exp(-σtr.TS*(P - Ptr.TS))) - 1.1 /(1 + exp(-σtr.SC*(P - Ptr.SC)))       )

    dFdτ =  ( 1 /(1 + exp(-σtr.TS*(P - Ptr.TS))) - 1.0 /(1 + exp(-σtr.SC*(P - Ptr.SC)))       )
 
    dFdP = -sin(φeff)
    n = sqrt(dFdτ^2 + dFdP^2)

    dFdτ = dFdτ / n
    dFdP = dFdP / n

    f = τ*dFdτ + P*dFdP  - C*cos(φeff)


    return f
end

function Q_nonlinear(τ, P, φ, C, Ptr, σtr) 
    
    φeff    = (1 - (1-sind(φ)) /(1 + exp(-σtr.TS*(P - Ptr.TS))) - (1+sind(φ)) /(1 + exp(-σtr.SC*(P - Ptr.SC)))       )
    # ψeff    = -(1 - 0.9 /(1 + exp(-σtr.TS*(P - Ptr.TS))) - 1.1 /(1 + exp(-σtr.SC*(P - Ptr.SC)))       )

    dFdτ =  ( 1 /(1 + exp(-σtr.TS*(P - Ptr.TS))) - 1.0 /(1 + exp(-σtr.SC*(P - Ptr.SC)))       )
 

    f = τ*dFdτ - P*sin(φeff)  - C*cos(φeff)


    return f
end

function main()

    sc = (σ=1,)

    P_end = 1000e6
    C     = 20e6
    φ     = 35.0
    ψ     = 5.0
    Cmin  = 1.0e6 # minimum stress
    Ptr   = ( TS= -20.e6, SC= 700.e6) # merging parameter
    σtr   = ( TS= 6e-8, SC=6e-8)   # merging parameter

    P_ax = LinRange(-P_end/10, P_end, 1000)
    τ_F  = collect(P_ax*sind(φ) .+ C*cosd(φ))
    τ_Q  = collect(P_ax*sind(φ) .+ C*cosd(φ))
    F    = zero(τ_F)
    Q    = zero(τ_F)

    ϕeff   = @. -(1 - 0.5 /(1 + exp(-σtr.TS*(P_ax - Ptr.TS))) - 1.5 /(1 + exp(-σtr.SC*(P_ax - Ptr.SC)))       )
    ψeff   = @. -(1 - 0.9 /(1 + exp(-σtr.TS*(P_ax - Ptr.TS))) - 1.1 /(1 + exp(-σtr.SC*(P_ax - Ptr.SC)))       )

    p3 = plot()
    p3 = plot!(P_ax, -asind.(ϕeff), label="ϕ")
    p3 = plot!(P_ax, -asind.(ψeff), label="ψ")


    dFdτ = @. ( 1 /(1 + exp(-σtr.TS*(P_ax - Ptr.TS))) - 1.0 /(1 + exp(-σtr.SC*(P_ax - Ptr.SC)))       )
    p4 = plot()
    p4 = plot!(P_ax, dFdτ, label="ϕ")


    niter = 10
    r_F   = zeros(niter)
    r_Q   = zeros(niter)

    for iter=1:niter

        for i in  eachindex(τ_F)

            # Yield function
            F[i]    = F_nonlinear(τ_F[i], P_ax[i], φ, C, Ptr, σtr) 
            jac     = Enzyme.gradient(Enzyme.Forward, F_nonlinear, τ_F[i], Const(P_ax[i]), Const(φ), Const(C), Const(Ptr), Const(σtr))
            τ_F[i] -= F[i]/jac[1]
            
            # Potential function
            Q[i]    = F_nonlinear(τ_Q[i], P_ax[i], ψ, C, Ptr, σtr) 
            jac     = Enzyme.gradient(Enzyme.Forward, F_nonlinear, τ_F[i], Const(P_ax[i]), Const(φ), Const(C), Const(Ptr), Const(σtr))
            τ_Q[i] -= Q[i]/jac[1]

        end
        r_F[iter] = norm(F)
        r_Q[iter] = norm(Q)
    end

    # Test return mapping
    τ_trial = [ 5e7 2e8 4e8 3e8   5e7]
    P_trial = [-4e7 2e8 6e8 8.5e8 9e8] 
    τ_corr  = zero(τ_trial)
    P_corr  = zero(P_trial)
    r_map   = 1e-10*ones(niter, length(τ_trial))

    α = LinRange(0.5, 1.0, 10)
    r_test = zero(α)

    # Loop over trial stress states
    for i in eachindex(τ_trial)
        τ, P = τ_trial[i], P_trial[i]
        λ̇    = 0.0
        F, τc, Pc    = F_nonlinear_corrected(λ̇, τ, P, φ, ψ, C, Ptr, σtr) 

        if F>1e-10
            for iter=1:niter
                @show F, τc, Pc  = F_nonlinear_corrected(λ̇, τ, P, φ, ψ, C, Ptr, σtr) 
                jac = Enzyme.gradient(Enzyme.Forward, F_nonlinear_corrected, λ̇, Const(τ), Const(P), Const(φ), Const(ψ), Const(C), Const(Ptr), Const(σtr))
                
                
                
                dλ̇ = -F/jac[1][1]

                imin = length(α)
                for i_ls in 1:length(α)
                    r_test[i_ls], τc, Pc = F_nonlinear_corrected(λ̇ + α[i_ls] * dλ̇, τ, P, φ, ψ, C, Ptr, σtr)
                end
                imin = argmin(r_test)
                @show α[imin]
                
                λ̇  += α[imin]*dλ̇
                τ_corr[i] = τc 
                P_corr[i] = Pc
                r_map[iter,i]  = abs(F)
            end
        end

    end
    @show τ_corr[5], P_corr[5]

    # p1 = plot(title="Yield/Potential", aspect_ratio=1)
    p1 = plot(title="Yield/Potential", aspect_ratio=1, xlim=(-0.2, 1.0), ylim=(-0.01, 0.5))

    p1 = plot!(P_ax.*sc.σ/1e9, τ_F.*sc.σ/1e9, c=:black, label="F")
    p1 = plot!(P_ax.*sc.σ/1e9, τ_Q.*sc.σ/1e9, c=:red, label="Q")
    for i in 1:5 #eachindex(τ_trial)
        p1 = scatter!( [P_trial[i] P_corr[i]].*sc.σ/1e9, [τ_trial[i] τ_corr[i]].*sc.σ/1e9, label=:none, markershape=:xcross)
    end

    p2 = plot(title="Convergence")
    p2 = plot!(1:niter, log10.(r_F), c=:black)
    p2 = plot!(1:niter, log10.(r_Q), c=:red)
    for i in eachindex(τ_trial)
        p2 = plot!(1:niter, log10.(r_map[:,i]), c=:green)
    end

    plot(p1, p2, p3, p4)

end 

main()