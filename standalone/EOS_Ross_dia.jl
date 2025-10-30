using Enzyme, GLMakie, DelimitedFiles

function Density(T, P, materials, r_vec, niter)

    V     = materials.V0[1]
    r0    = 1.0
    iter  = 0
    err   = 1.0

    tol   = 1e-10
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
    γ0    = materials.γ0[1]
    θE    = materials.θE[1]
    T0    = materials.T0[1]
    V0    = materials.V0[1]
    q     = materials.q[1]
    Natom = materials.Nat[1]

# So, what you have to do for the diamond, for now, and for any other q-compromise EoS:

# 1) Set the Einstein temperature and keep it constant always. Use that in the equation you have for the integral of Cv.

# 2) When you calculate the thermal pressure as gamma/V times the integral of the Cv, multiply the integral by gamma0/v0

    γ    = γ0 * (V/V0)^q
    sca  = 1e-3

    R    = 8.31415
    # U0   =  R*θE/(exp(θE/T0) - 1)
    # U    =  R*θE/(exp(θE/T)  - 1)
    # P_th = sca * 3*R*Nat*θE*γ/V * (1/(exp(θE/T)-1)- 1/(exp(θE/T0)-1))
    # P_th = sca * 3*γ0/V * (U - 2*U0)
    P     = sca * 3*Natom*γ/V*(U_Einstein(T, θE) - U_Einstein(T0, θE))

    # P_th = sca * 3*Nat*γ/V * (U_Einstein(T, θE) - U_Einstein(T0, θE))

    # u    = θE/T
    # u0   = θE/T0
    # # u    = θE/T
    # # u0   = θE/T0
    # ξ0   = u0^2*exp(u0) / (exp(u0) - 1)^2
    # P_th =  sca * α0*K0/ξ0 * (1/(exp(u)-1) - 1/(exp(u0)-1))
    return P  
end

function residual(V, P, T, materials)
    @show P_mechanical(V, materials) 
    @show P_thermal(V, T, materials)
    return P - P_mechanical(V, materials) - P_thermal(V, T, materials)
end

function main()

    materials = (
        # Birch-Murnaghan
        ρr  = [3515],
        V0  = [ 3.42],
        K0  = [444.0],
        Kp  = [4.0],
        Kpp = [-0.0374],

        # Einstein's model for Pthermal
        θE  = [1500.0],
        γ0  = [0.9726],
        T0  = [298.15],
        q   = [1.0],
        Nat = [1.0],
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

        # ax = Axis(fig[3,2], xlabel="T (GPa)", ylabel="P (GPa)")
        # hm = heatmap!(ax, T_vec, P_vec./1e9, log10.(abs.(ρ.-ρ_Ol_Ross')))
        # Colorbar(fig[4, 2], hm, label = L"$ρ$ (g/cm³)", height=10, width = 200, labelsize = 15, ticklabelsize = 15, vertical=false, valign=true, flipaxis = true )
       
        ax = Axis(fig[1,2], xlabel="iterations", ylabel="r")
        scatterlines!(ax, 1:iter, log10.(r_vec[1:iter]))
        display(fig)
    end
    with_theme(Visualisation, theme_latexfonts())

    # Single point 
    T     = 1100.0
    P     = 5e9/1e9
    @info "Single point"
    V, iter = Density(T, P, materials, r_vec, niter)
    @show V, materials.V0[1]/V*materials.ρr[1]
end

main()

#    3.42002  3.42061  3.42144  3.42250  3.42377  3.42522  3.42683  3.42857  3.43043  3.43240  3.43446  3.43660  3.43880  3.44107  3.44340  3.44577  3.44819
#    3.41236  3.41294  3.41376  3.41481  3.41607  3.41750  3.41909  3.42081  3.42266  3.42460  3.42664  3.42875  3.43093  3.43317  3.43547  3.43782  3.44021
#    3.40478  3.40536  3.40617  3.40721  3.40845  3.40987  3.41144  3.41315  3.41497  3.41689  3.41890  3.42099  3.42315  3.42537  3.42764  3.42996  3.43233
#    3.39729  3.39786  3.39866  3.39969  3.40092  3.40232  3.40388  3.40556  3.40736  3.40927  3.41126  3.41332  3.41545  3.41765  3.41989  3.42219  3.42453
#    3.38988  3.39044  3.39124  3.39225  3.39347  3.39486  3.39639  3.39806  3.39984  3.40173  3.40369  3.40574  3.40784  3.41001  3.41223  3.41451  3.41682
#    3.38255  3.38311  3.38389  3.38490  3.38610  3.38747  3.38899  3.39064  3.39241  3.39427  3.39621  3.39823  3.40032  3.40246  3.40466  3.40691  3.40919
