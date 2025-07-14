using Enzyme, LinearAlgebra, Plots

function F_combined_corrected(λ̇, τ, P, φ, ψ, C, Cmin, Ptr, σtr) 

    jac  = Enzyme.gradient(Enzyme.Forward, F_combined, τ, Const(P), Const(ψ), Const(C), Const(Cmin), Const(Ptr), Const(σtr))
    dQdτ = jac[1]

    jac  = Enzyme.gradient(Enzyme.Forward, F_combined,  Const(τ), P, Const(ψ), Const(C), Const(Cmin), Const(Ptr), Const(σtr))
    dQdP = jac[2]

    τ = τ - λ̇*dQdτ 
    P = P - λ̇*dQdP
    f = F_combined(τ, P, φ, C, Cmin, Ptr, σtr)

    return f, τ, P
end


function F_combined(τ, P, φ, C, Cmin, Ptr, σtr) 
    
    α_TS = 1 /(1 + exp(-σtr.TS*(P - Ptr.TS))) 
    α_SC = 1 /(1 + exp(-σtr.SC*(P - Ptr.SC))) 
    f_DP = τ - P*sind(φ) - C*cosd(φ)
    f_M  = τ - Cmin
    f    = (1-α_TS) * f_M  + (α_TS) * (1 - α_SC) * f_DP +  (α_SC) * f_M 

    return f
end

function main()

    sc = (σ=1,)

    P_end = 800e6
    C     = 20e6
    φ     = 35.0
    ψ     = 5.0
    Cmin  = 1.0e6 # minimum stress
    Ptr   = ( TS= 0.e6, SC= 700.e6) # merging parameter
    σtr   = ( TS= 1e-6, SC= 5e-8)   # merging parameter

    P_ax = LinRange(-P_end/10, P_end, 1000)
    τ_F  = collect(P_ax*sind(φ) .+ C*cosd(φ))
    τ_Q  = collect(P_ax*sind(φ) .+ C*cosd(φ))
    F    = zero(τ_F)
    Q    = zero(τ_F)



    niter = 10
    r_F   = zeros(niter)
    r_Q   = zeros(niter)

    for iter=1:niter

        for i in  eachindex(τ_F)

            # Yield function
            F[i]    = F_combined(τ_F[i], P_ax[i], φ, C, Cmin, Ptr, σtr) 
            jac     = Enzyme.gradient(Enzyme.Forward, F_combined, τ_F[i], Const(P_ax[i]), Const(φ), Const(C), Const(Cmin), Const(Ptr), Const(σtr))
            τ_F[i] -= F[i]/jac[1]
            
            # Potential function
            Q[i]    = F_combined(τ_Q[i], P_ax[i], ψ, C, Cmin, Ptr, σtr) 
            jac     = Enzyme.gradient(Enzyme.Forward, F_combined, τ_F[i], Const(P_ax[i]), Const(φ), Const(C), Const(Cmin), Const(Ptr), Const(σtr))
            τ_Q[i] -= Q[i]/jac[1]

        end
        r_F[iter] = norm(F)
        r_Q[iter] = norm(Q)
    end

    # Test return mapping
    τ_trial = [ 2e7 2e8 4e8 2e8  ]
    P_trial = [-0e7 2e8 6e8 7.5e8]
    τ_corr  = zero(τ_trial)
    P_corr  = zero(P_trial)
    r_map   = 1e-10*ones(niter, length(τ_trial))

    # Loop over trial stress states
    for i in eachindex(τ_trial)
        τ, P = τ_trial[i], P_trial[i]
        λ̇    = 0.0
        F, τc, Pc    = F_combined_corrected(λ̇, τ, P, φ, ψ, C, Cmin, Ptr, σtr) 

        if F>1e-10
            for iter=1:niter
                @show F, τc, Pc  = F_combined_corrected(λ̇, τ, P, φ, ψ, C, Cmin, Ptr, σtr) 
                jac = Enzyme.gradient(Enzyme.Forward, F_combined_corrected, λ̇, Const(τ), Const(P), Const(φ), Const(ψ), Const(C), Const(Cmin), Const(Ptr), Const(σtr))
                λ̇  -= F/jac[1][1]
                τ_corr[i] = τc 
                P_corr[i] = Pc
                r_map[iter,i]  = abs(F)
            end
        end

    end

    p1 = plot(title="Yield/Potential", aspect_ratio=1)
    p1 = plot!(P_ax.*sc.σ/1e9, τ_F.*sc.σ/1e9, c=:black, label="F")
    p1 = plot!(P_ax.*sc.σ/1e9, τ_Q.*sc.σ/1e9, c=:red, label="Q")
    p1 = scatter!( [P_trial P_corr].*sc.σ/1e9, [τ_trial τ_corr].*sc.σ/1e9, label=:none)

    p2 = plot(title="Convergence")
    p2 = plot!(1:niter, log10.(r_F), c=:black)
    p2 = plot!(1:niter, log10.(r_Q), c=:red)
    for i in eachindex(τ_trial)
        p2 = plot!(1:niter, log10.(r_map[:,i]), c=:green)
    end

    plot(p1, p2)

end 

main()