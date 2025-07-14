using Enzyme, LinearAlgebra, Plots

function F_hyperbolic(τ, P, φ, C, σT)  
    return sqrt( τ^2 + (C*cosd(φ)-σT*sind(φ))^2 ) - (C*cosd(φ)  + P*sind(φ))  
end

function Φ(P, βt, Φt, Φc) 
    @show βc = -βt * Φt / Φc # make it differentiable
    if P <0
        Φ = Φt * (1 - exp(-βt*P))
    else
        Φ = Φc * (1 - exp(βc*P))
    end
    return Φ
end 


function F_ModCamClay(τ, P, P0, α, Φt)  
    return sqrt( α^2*τ^2 + (P - P0)^2 ) - Φt  
end

function F_combined(τ, P, P0, α, Φ, φ, C, Cmin, Ptr, σtr) 
    
    # α_TS = 1 /(1 + exp(-σtr.TS*(P - Ptr.TS))) 
    # α_SC = 1 /(1 + exp(-σtr.SC*(P - Ptr.SC))) 
    # α_CM = 0#1 /(1 + exp(-σtr.CM*(P - Ptr.CM))) 
    f_CC = sqrt( α^2*τ^2 + (P - P0)^2 ) - Φ
    # f_DP = τ - P*sind(φ) - C*cosd(φ)
    # f_M  = τ - Cmin
    # f    = (1 - α_SC) * f_DP +  (α_SC) * f_CC * (1-α_CM) + α_CM*f_M

    f    = f_CC#(α_TS) * f_M  + (1 - α_TS) * (1 - α_SC) * f_DP +  (α_SC) * f_M 

    # f    = 1 / ( (1 - α_SC) * 1/abs(f_DP) +  (α_SC) * 1/abs(f_CC) )

    return f
end

function main()

    sc = (σ=1e9,)

    σT    = 0*50.
    P_end = 500e6
    C     = 20e6
    φ     = 35.0
    ψ     = 5.0

    P_ax = LinRange(-P_end, P_end, 1000)
    τ_F  = collect(P_ax*sind(φ) .+ C*cosd(φ))
    τ_Q  = collect(P_ax*sind(φ) .+ C*cosd(φ))
    F    = zero(τ_F)
    Q    = zero(τ_F)

    P0   = 5e8
    Φt   = 1e6
    α    = 1

    Cmin = 1.0
    Ptr = ( SC= 100., CM= 300.)
    σtr = ( SC= 0.5, CM= 3.)

    # plot(P_ax, α)

    niter = 45
    r_F   = zeros(niter)
    r_Q   = zeros(niter)

    for iter=1:niter

        for i in  eachindex(τ_F)
            # F[i] = F_combined(τ_F[i], P_ax[i], P0, α, Φt, φ, C, Cmin, Ptr, σtr) 
            # jac = Enzyme.gradient(Enzyme.Forward, F_combined, τ_F[i], Const(P_ax[i]), Const(P0), Const(α), Const(Φt), Const(φ), Const(C), Const(Cmin), Const(Ptr), Const(σtr))
            # τ_F[i] -= 1*F[i]/jac[1]
            
            F[i] = F_ModCamClay(τ_F[i], P_ax[i], P0, α, Φt) 
            jac = Enzyme.gradient(Enzyme.Forward, F_ModCamClay, τ_F[i], Const(P_ax[i]), Const(P0), Const(α), Const(Φt))
            # jac = τ_ax[i] ./ sqrt.( τ_ax[i].^2 .+ (C*cosd(φ)-σT*sind(φ)).^2 )
            τ_F[i] -= 0.4*F[i]/jac[1]
# @show F[i]


            # Q[i] = F_ModCamClay(τ_Q[i], P_ax[i], P0, α, βt, Φt)  
            # jac = Enzyme.gradient(Enzyme.Forward, F_ModCamClay, τ_Q[i], Const(P_ax[i]), Const(P0), Const(α), Const(βt), Const(Φt))
            # τ_Q[i] -= Q[i]/jac[1]

        end
        r_F[iter] = norm(F)
        r_Q[iter] = norm(Q)
    end

    @show P_ax


    p1 = plot(P_ax.*sc.σ/1e9, τ_F.*sc.σ/1e9, c=:black, aspect_ratio=1)
    # p1 = plot!(P_ax.*sc.σ/1e9, τ_Q.*sc.σ/1e9, c=:red)

    p2 = plot(1:niter, log10.(r_F), c=:black)
    # p2 = plot!(1:niter, log10.(r_Q), c=:red)

    # tr_SC = @. 1 /(1 + exp(-σtr.SC*(P_ax - Ptr.SC)))
    # p3 = plot(P_ax, tr_SC )

    plot(p1, p2)

end 

main()


        # α = LinRange(0.01, 1.0, 10)
        # r_test = zero(α)

        # imin = length(α)
        # for i_ls in 1:length(α)
        #     r_test[i_ls] = F_hyperbolic(τ_ax[i] - α[i_ls] * dτ, P_ax[i], φ, C, σT)
        # end
        # @show imin = argmin(r_test)
        # τ_ax[i] -=  dτ
