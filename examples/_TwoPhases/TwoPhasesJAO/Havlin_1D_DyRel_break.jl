# closed top
yr  = 365.25*24*3600
cmy = 100*yr

bulk(ϕ, ηs, m) = ηs*abs(ϕ)^m

perm(ϕ, a) = a^2 * abs(ϕ)^(2.7) / 58 

function compaction_length(ϕ0, p)
    k0 = perm(ϕ0, p.a)
    χ0 = bulk(ϕ0, p.ηs, p.m)
    return sqrt((k0/p.μl) * (χ0 + 4/3*p.ηs)) 
end

function main_Havlin_DR(nc)

    open_top = true

    @load "havlin_ac.jld2" por_snapshot z

    # Parameters
    p = (
        m     = -1.0,
        ϕ0    = 4e-2,
        μl    = 1.0,
        ηs    = 1e16,
        Gs    = 2e110,
        a     = 5e-3,
        ρs    = 3200.0,
        ρl    = 3000.0,
        gy    = -9.8,
        yfact = 10.0,
        KΦ    = 1e110,
        Ks    = 1e110,
        Kf    = 1e110,
    )

    # Compaction length
    lc = compaction_length(p.ϕ0, p)
    @info "Compaction length: $(lc) m --- Model size: $(p.yfact*lc) m"
    @info "Pore Maxwell time: $(p.ηs/p.KΦ) s"

    # Time domain
    nt = 1 #1000
    Δt = 1e6

    # Space domain
    # y   = (min=-p.yfact*lc, max=0.0)
    y   = (min=-43.68e3, max=0.0)
    Δy  = (y.max - y.min)/nc
    yce = LinRange(y.min-Δy/2, y.max+Δy/2, nc+2)
    yv  = LinRange(y.min, y.max, nc+1)

    # Non-linear solver
    niter = 1000
    tol   = 1e-10
    nr0   = 1.0

    # Arrays
    ϕ    = p.ϕ0*ones(nc+2)
    ϕ0   = p.ϕ0*ones(nc+2)
    dϕdt = p.ϕ0*ones(nc+2)
    Vy   =     zeros(nc+3)
    τyy  =     zeros(nc+2)
    τyy0 =     zeros(nc+2)
    Pt   =     zeros(nc+2)
    Pt0  =     zeros(nc+2)
    Pf   =     zeros(nc+2)
    Pf0  =     zeros(nc+2)

    ∂Vy∂y =     zeros(nc+2)
    divVs =     zeros(nc+2)
    divqD =     zeros(nc+2)
    ε̇yy   =     zeros(nc+2)
    dPfdt =     zeros(nc+2)
    dPtdt =     zeros(nc+2)
    dPsdt =     zeros(nc+2)
    ξ     =     zeros(nc+2)
    k     =     zeros(nc+2)
    ρt    =     zeros(nc+2)
    ρtv   =     zeros(nc+1)
    kv    =     zeros(nc+1)
    qy    =     zeros(nc+1)

    Ry    =     zeros(nc+1)
    RPt   =     zeros(nc+2)
    RPf   =     zeros(nc+2)
    dVydτ =     zeros(nc+1)

    # Initial conditions
    ρt0    = p.ρs*(1-p.ϕ0) + p.ϕ0*p.ρl
    Pf_top = 0*20e6
    Pt_top = 0*20e6
    Pt_bot = -ρt0*p.gy*(y.max-y.min)
    Pf_bot = -p.ρl*p.gy*(y.max-y.min)
    @show Pf_bot
    @show Pt_bot
    for j in (nc+2-1):-1:2
        Pf[j] = Pf[j+1] - p.ρl * p.gy .* Δy
        Pt[j] = Pt[j+1] - ρt0  * p.gy .* Δy
    end

    ρlg = p.gy * p.ρl
    
    if open_top
        iVy = 2:length(Ry)-0
        iPt = 2:length(Pt)-0
    else
        iVy = 2:length(Ry)-1
        iPt = 2:length(Pt)-1
    end

    @show Pt[end-1:end]
    @show diff(Pt[end-1:end])/Δy
    
    for iter=1:5000000

        # BC
        Pf[end]   = 2*Pf_top - Pf[end-1]
        Pf[1]     = 2*Pf_bot - Pf[2]

        Pt[end]   = 2*Pt_top - Pt[end-1]
        Pt[1]     = 2*Pt_bot - Pt[2]

        Vy[1]    = Vy[2]
        Vy[end]  = Vy[end-1]

        # Fluxes
        ∂Vy∂y          .= diff(Vy)/Δy
        divVs          .= ∂Vy∂y     
        ε̇yy            .= ∂Vy∂y .- 1/3*divVs
        τyy            .= 2 .* p.ηs .*  ε̇yy

        ξ              .= bulk.(ϕ, p.ηs, p.m) 
        dϕdt           .= @. (Pf - Pt)/ξ
        ϕ              .= @. ϕ0 .+ dϕdt * Δt
        ρt             .= @. ϕ*p.ρl + (1-ϕ)*p.ρs
        ρtv            .= @. (ρt[2:end] + ρt[1:end-1]) / 2 

        # dPfdt          .= @. (Pf - Pf0) / Δt
        # dPtdt          .= @. (Pt - Pt0) / Δt
        # dPsdt          .= @. dϕdt*(Pt - Pf*ϕ)/(1-ϕ)^2 + (dPtdt - ϕ*dPfdt - Pf*dϕdt) / (1 - ϕ)
       
        k              .= perm.(ϕ, p.a)
        kv             .= @. (k[2:end] + k[1:end-1]) / 2 / p.μl
        qy             .= -kv .* (diff(Pf)/ Δy .- ρlg) 

        divqD[2:end-1] .= diff(qy)/ Δy 

        Ry[iVy]        .= diff(τyy[iPt]) / Δy .- diff(Pt[iPt]) / Δy +  ρtv[iVy]*p.gy
        RPt[2:end-1]   .= @. - dϕdt[2:end-1]/(1-ϕ[2:end-1]) + divVs[2:end-1]
        RPf[2:end-1]   .= @.   dϕdt[2:end-1] + ϕ[2:end-1]*divVs[2:end-1] + divqD[2:end-1]
        
        # RPt[end-1] = (Pt[end-1] - Pf[end-1]) / 1e11

        if iter==1 || mod(iter, 10000)==0
            @info iter
            println( norm(Ry)  )
            println( norm(RPt) )
            println( norm(RPf) )
        end

        dVydτ .= Ry + (0.9999)*dVydτ

        Vy[2:end-1] .+= dVydτ / 5e10 * 2
        Pt[2:end-1] .-= RPt[2:end-1] / 1e-11 *4
        Pf[2:end-1] .-= RPf[2:end-1] / 1e-11 *4

    end


     fig = Figure()
            
        ax1 = Axis(fig[1,1], xlabel=L"$Pt$, $Pf$ (MPa)", ylabel=L"$y$ (km)")
        lines!(ax1, Pt[2:end]./1e6, yce[2:end]./1e3)
        lines!(ax1, Pf[2:end]./1e6, yce[2:end]./1e3, linestyle=:dash)

        # ax1 = Axis(fig[1,1], xlabel=L"$\tau_{yy}$ (MPa)", ylabel=L"$y$ (km)")
        # lines!(ax1, τyy[2:end-1]./1e6, yce[2:end-1]./1e3)
        
        ax2 = Axis(fig[1,2], xlabel=L"$\Delta P$ (MPa)", ylabel=L"$y$ (km)")
        lines!(ax2, ((Pf .- Pt) ./ (1 .-ϕ))[2:end-1]./1e6, yce[2:end-1]./1e3)
        # lines!(ax2, Pt[2:end-1]./1e6, yce[2:end-1]./1e3)
        # lines!(ax2, Pf[2:end-1]./1e6, yce[2:end-1]./1e3, linestyle=:dash)

        ax3 = Axis(fig[2,1], xlabel=L"$Vy$ (cm/y)", ylabel=L"$y$ (km)")
        lines!(ax3, Vy[2:end-1]*cmy, yv./1e3)

        ax4 = Axis(fig[2,2], xlabel=L"$\phi$", ylabel=L"$y$ (km)")
        # lines!(ax4, por_snapshot[2:end-1], -z[2:end-1]./1e3, color=:green, label=L"$\phi$ Paris")
        step = 1
        lines!(ax4, ϕ[2:step:end-1], yce[2:step:end-1]./1e3, label=L"$\phi$ Frankfurt")
        axislegend(position=:rb)

        display(fig)

        @show Pt[end-3:end]
        @show Pf[end-3:end]

        @save "havlin_DR_debug.jld2" Pt Pf τyy ϕ Vy Pt0 Pf0 τyy0 ϕ0

end

main_Havlin_DR(51)