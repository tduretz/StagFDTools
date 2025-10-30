using Enzyme, GLMakie, DelimitedFiles

function Density(T, P, materials, r_vec, niter)

    V     = materials.V0[1]
    r0    = 1.0
    iter  = 0
    err   = 1.0

    tol   = 1e-13
    iter  = 0
    
    while (iter<niter && err>tol)

        iter += 1
        
        # Evaluate the Jacobian: ∂r∂V
        J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, residual, V, Const(P), Const(T), Const(materials) )
        
        r = J.val[1]
        if iter==1 r0 = r end
        err         = abs(r/r0)
        r_vec[iter] = err

        @show r, J.derivs[1]

        # Newton update
        V -= r/J.derivs[1]

    end
    return V, iter
end

function P_mechanical(V, materials)

    V0  = materials.V0[1]
    K0  = materials.K0[1]
    Kp  = materials.Kp[1]
    Kpp = materials.Kpp[1]

    f = ((V0/V)^(2/3) -1)/2
    # P = 3*K0*f*(1+2*f)^(5/2) * (1 + 3/2*(Kp -4)*f + 3/2*(K0*Kpp + (Kp - 4)*(Kp-3) + 35/9)*f^2 )
    P = 3*K0*f*(1+2*f)^(5/2) * (1 + 3/2*(Kp -4)*f )
    return P
end

function U_Einstein(T, θE)
    R = 8.31415
    return R*θE/(exp(θE/T) - 1)
end

function P_thermal(V, T, materials)
    γ0   = materials.γ0[1]
    θE   = materials.θE[1]
    T0   = materials.T0[1]
    V0   = materials.V0[1]
    q    = materials.q[1]
    Nat  = materials.Nat[1]
    γ    = γ0 * (V/V0)^q
    sca  = 1e-3
    P_th = sca * 3*Nat*γ/(V)*(U_Einstein(T, θE) - U_Einstein(T0, θE))
    return P_th  
end

function residual(V, P, T, materials)
    # @show P_mechanical(V, materials) 
    # @show P_thermal(V, T, materials)
    return P - P_mechanical(V, materials) - P_thermal(V, T, materials)
end

function main()

    materials = (
        # Birch-Murnaghan
        ρr  = [3250],
        V0  = [43.8900],
        K0  = [126.30],
        Kp  = [4.54],
        Kpp = [-0.0374],

        # Einstein's model for Pthermal
        θE  = [471.0],
        γ0  = [1.044],
        T0  = [298.15],
        q   = [1.88],
        Nat = [7.0],
    )

    iter  = 0
    niter = 20
    r_vec = zeros(niter)
    P_vec = LinRange(0, 5e9, 51)
    T_vec = LinRange(300, 1100, 17)
    ρ     = zeros(length(T_vec), length(P_vec))
    
    # for i in eachindex(T_vec), j in eachindex(P_vec)

    #     # P-T values
    #     P  = P_vec[j]/1e9
    #     T  = T_vec[i]

    #     # Initial guess
    #     V, iter = Density(T, P, materials, r_vec, niter)
    #     ρ[i,j] = materials.V0[1]/V*materials.ρr[1]
    # end

    function Visualisation()
        V_Ol_Ross = readdlm("./data/MantleOl_Einstein_trimmed.cal")
        ρ_Ol_Ross = materials.V0[1]./V_Ol_Ross .* materials.ρr[1]

        fig = Figure()

        ax = Axis(fig[1,1], xlabel="T (GPa)", ylabel="P (GPa)")
        hm = heatmap!(ax, T_vec, P_vec./1e9, ρ_Ol_Ross')
        Colorbar(fig[2, 1], hm, label = L"$ρ$ (g/cm³)", height=10, width = 200, labelsize = 15, ticklabelsize = 15, vertical=false, valign=true, flipaxis = true )
        
        ax = Axis(fig[3,1], xlabel="T (GPa)", ylabel="P (GPa)")
        hm = heatmap!(ax, T_vec, P_vec./1e9, ρ)
        Colorbar(fig[4, 1], hm, label = L"$ρ$ (g/cm³)", height=10, width = 200, labelsize = 15, ticklabelsize = 15, vertical=false, valign=true, flipaxis = true )

        ax = Axis(fig[3,2], xlabel="T (GPa)", ylabel="P (GPa)")
        hm = heatmap!(ax, T_vec, P_vec./1e9, log10.(abs.(ρ.-ρ_Ol_Ross')))
        Colorbar(fig[4, 2], hm, label = L"$ρ$ (g/cm³)", height=10, width = 200, labelsize = 15, ticklabelsize = 15, vertical=false, valign=true, flipaxis = true )
       
        ax = Axis(fig[1,2], xlabel="iterations", ylabel="r")
        scatterlines!(ax, 1:iter, log10.(r_vec[1:iter]))
        display(fig)
    end
    with_theme(Visualisation, theme_latexfonts())

    # Single point 
    T     = 1000.0
    P     = 1e9/1e9
    @info "Single point"
    V, iter = Density(T, P, materials, r_vec, niter)
    @show materials.V0[1]/V*materials.ρr[1]
end

main()