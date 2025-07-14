using Enzyme, LinearAlgebra, Plots

function F_hyperbolic(τ, P, φ, C, σT)  
    return sqrt( τ^2 + (C*cosd(φ)-σT*sind(φ))^2 ) - (C*cosd(φ)  + P*sind(φ))  
end

function main()

    sc = (σ=1.,)

    σT    = -50.
    P_end = 100.
    C     = 20
    φ     = 35.0
    ψ     = 5.0

    P_ax = LinRange(-σT+1e-4, P_end, 100)
    τ_F  = collect(P_ax*sind(φ) .+ C*cosd(φ))
    τ_Q  = collect(P_ax*sind(φ) .+ C*cosd(ψ))
    F    = zero(τ_F)
    Q    = zero(τ_F)

    niter = 15
    r_F   = zeros(niter)
    r_Q   = zeros(niter)

    for iter=1:niter

        for i in eachindex(τ_F)

            F[i] = F_hyperbolic(τ_F[i], P_ax[i], φ, C, σT) 
            jac = Enzyme.gradient(Enzyme.Forward, F_hyperbolic, τ_F[i], Const(P_ax[i]), Const(φ), Const(C), Const(σT))
            # jac = τ_ax[i] ./ sqrt.( τ_ax[i].^2 .+ (C*cosd(φ)-σT*sind(φ)).^2 )
            τ_F[i] -= F[i]/jac[1]

            Q[i] = F_hyperbolic(τ_Q[i], P_ax[i], ψ, C, σT) 
            jac = Enzyme.gradient(Enzyme.Forward, F_hyperbolic, τ_Q[i], Const(P_ax[i]), Const(ψ), Const(C), Const(σT))
            τ_Q[i] -= Q[i]/jac[1]

   


        end


        jac = Enzyme.gradient(Enzyme.Forward, F_hyperbolic, τ_Q[i], Const(P_ax[i]), Const(ψ), Const(C), Const(σT))


        r_F[iter] = norm(F)
        r_Q[iter] = norm(Q)
    end

    p1 = plot(P_ax.*sc.σ/1e9, τ_F.*sc.σ/1e9, c=:black)
    p1 = plot!(P_ax.*sc.σ/1e9, τ_Q.*sc.σ/1e9, c=:red)

    p2 = plot(1:niter, log10.(r_F), c=:black)
    p2 = plot!(1:niter, log10.(r_Q), c=:red)

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
