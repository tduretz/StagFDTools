using CairoMakie, StaticArrays, ExtendableSparse, LinearAlgebra, Printf, JLD2
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian

yr  = 365.25*24*3600
cmy = 100*yr

perm(ϕ, a) = a^2*ϕ^2.7 / 58

bulk(ϕ, ηs, m) = ηs*ϕ^m

function porosity_rate(Pt, Pf, ϕ0, p)
    χ       = bulk(ϕ0, p.ηs, p.m)
    return (Pf - Pt)/χ 
end

function compaction_length(ϕ0, p)
    k0 = perm(ϕ0, p.a)
    χ0 = bulk(ϕ0, p.ηs, p.m)
    @show χ0
    @show k0
    return sqrt((k0/p.μl) * (χ0 + 4/3*p.ηs)) 
end

function Darcy!(qDy, Vyf, ϕ, Pf, Vys, BC, num, p, Δy, Δt)
    for j = 2:length(qDy)-1

        Pfˡ  = MVector{2}(  Pf[jj]   for jj in j-1:j   )
        ϕˡ   = SVector{2}(   ϕ[jj]   for jj in j-1:j   )

        ρlg     = p.ρl * p.gy

        ∂Pf∂y = (Pfˡ[2] - Pfˡ[1]) / Δy
        ϕav   = (ϕˡ[2]  + ϕˡ[1] ) / 2

        k_μ    = perm.(ϕav, p.a) / p.μl
        qDy[j] = -k_μ * (∂Pf∂y - ρlg) 
        Vyf[j] =  (qDy[j] + ϕav*Vys[j]) / ϕav
    end

end

function momemtum_local(Vys, Pt, Pf, ϕ0, tag, p, Δy, Δt)

    # Neumann BC for surface
    if tag[end] == 2
        Vys[end] = Vys[2]
    end

    # Phi on Vys points
    dϕdt    = SVector{2}( porosity_rate(Pt[i], Pf[i], ϕ0[i], p) for i in 1:2 )
    ϕ       = SVector{2}( @. ϕ0 + Δt * dϕdt )
    ϕy      = ((ϕ[2:end] + ϕ[1:end-1]) / 2)[1] 

    # Kinematics
    ∂Vy∂y = SVector{2}( @. (Vys[2:end] - Vys[1:end-1]) / Δy )
    ε̇yy′  = SVector{2}( @. ∂Vy∂y - 1/3*(∂Vy∂y) )

    # Rheology
    τyy   = SVector{2}( @. 2 * p.ηs * ε̇yy′ )   #* (1-ϕy)

    # Rheology
    ∂τyy∂y = ((τyy[2:end] - τyy[1:end-1]) / Δy)[1] 
    ∂Pt∂y  = (( Pt[2:end] -  Pt[1:end-1]) / Δy)[1] 
    
    # Body force
    ρt     = (1 - ϕy) * p.ρs + ϕy * p.ρl

    return - (∂τyy∂y - ∂Pt∂y + ρt*p.gy)
end

function continuity_local(Vys, Pt, Pf, ϕ0, tag, p, Δy, Δt)

    dlnρsdt = @SVector zeros(3)
   
    # Phi 
    dϕdt    = SVector{3}( porosity_rate(Pt[i], Pf[i], ϕ0[i], p) for i in 1:3 )
    ϕ       = SVector{3}( @. ϕ0 + Δt * dϕdt )

    # Solid divergence
    divVs   = (Vys[2] - Vys[1]) / Δy

    return dlnρsdt[2] - dϕdt[2]/(1-ϕ[2]) + divVs
end

function fluid_continuity_local(Vys, Pt, Pf, ϕ0, tag, p, Δy, Δt)

    dlnρfdt = @SVector zeros(3)

    # Phi 
    dϕdt    = SVector{3}( porosity_rate(Pt[i], Pf[i], ϕ0[i], p) for i in 1:3 )
    ϕ       = SVector{3}( @. ϕ0 + Δt * dϕdt )

    # Buoyancy
    ρlg     = p.ρl * p.gy

    # BC
    if tag[end] == 2 # Top: no flux
       Pf[end] = Pf[end-1] + ρlg * Δy
    end
    if tag[1] == 1 # Bottom: try to set Pf = Pt such that ϕ = ϕ0 
        ϕS     = (ϕ[1] + ϕ[2])/2
        ρtg    = ((1-ϕS)*p.ρs + ϕS*p.ρl) * p.gy
        lc     = compaction_length(p.ϕ0, p)
        y_base = -p.yfact*lc
        Pt_bot = (y_base-3*Δy/2)*ρtg
        # Pt[1]  = 2*Pt_bot - Pt[2]
        # Pf[2]  =  (Pt[1]+Pt[2])/2 / 2
        Pf[1]  = 2*Pt_bot - Pf[2]
    end

    # Darcy
    k       = SVector{3}( perm.(ϕ, p.a) )
    k_μ     = SVector{2}( @. (k[2:end] + k[1:end-1]) / 2 / p.μl) 
    qy      = SVector{2}( @. -k_μ .* ((Pf[2:end] - Pf[1:end-1])/ Δy - ρlg) )

    # Solid divergence
    divVs   = (Vys[2] - Vys[1]) / Δy

    # Darcy flux divergence
    divqD   = (qy[2] - qy[1]) / Δy

    return ϕ[2]*dlnρfdt[2] + dϕdt[2] + ϕ[2]*divVs + divqD
end

function momentum!(M, r, Vys, Pt, Pf, ϕ0, BC, num, p, Δy, Δt)

    ∂R∂Vy   = @MVector zeros(3)
    ∂R∂Pt   = @MVector zeros(2)
    ∂R∂Pf   = @MVector zeros(2)

    for j = 2:length(Vys)-1

        # Local stencil
        Vyˡ  = MVector{3}( Vys[jj]   for jj in j-1:j+1 )
        Ptˡ  = MVector{2}(  Pt[jj]   for jj in j-1:j   )
        Pfˡ  = MVector{2}(  Pf[jj]   for jj in j-1:j   )
        ϕ0ˡ  = SVector{2}(  ϕ0[jj]   for jj in j-1:j   )
        tagˡ = SVector{3}( BC.Vys[jj] for jj in j-1:j+1 )

        # Residual
        if num.Vys[j]>0
            r[num.Vys[j]] = momemtum_local(Vyˡ, Ptˡ, Pfˡ, ϕ0ˡ, tagˡ, p, Δy, Δt)
        end

        # Jacobian
        fill!(∂R∂Vy, 0.0)
        fill!(∂R∂Pt, 0.0)
        fill!(∂R∂Pf, 0.0)
        forwarddiff_gradients!(momemtum_local, Duplicated(Vyˡ, ∂R∂Vy), Duplicated(Ptˡ, ∂R∂Pt), Duplicated(Pfˡ, ∂R∂Pf), Const(ϕ0ˡ), Const(tagˡ), Const(p), Const(Δy), Const(Δt))

        # Vys --- Vys
        connect = SVector{3}( num.Vys[jj]   for jj in j-1:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Vys[j]>0
                M[num.Vys[j], connect[jj]] = ∂R∂Vy[jj] 
            end
        end

        # Vys --- Pt
        connect = SVector{2}( num.Pt[jj]   for jj in j-1:j )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Vys[j]>0
                M[num.Vys[j], connect[jj]] = ∂R∂Pt[jj] 
            end
        end

        # Vys --- Pf
        connect = SVector{2}( num.Pf[jj]   for jj in j-1:j )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Vys[j]>0
                M[num.Vys[j], connect[jj]] = ∂R∂Pf[jj] 
            end
        end
        
    end
end

function continuity!(M, r, Vys, Pt, Pf, ϕ0, BC, num, p, Δy, Δt)

    ∂R∂Vy   = @MVector zeros(2)
    ∂R∂Pt   = @MVector zeros(3)
    ∂R∂Pf   = @MVector zeros(3)

    for j = 2:length(Pt)-1

        # Local stencil
        Vyˡ  = MVector{2}( Vys[jj]   for jj in j:j+1   )
        Ptˡ  = MVector{3}(  Pt[jj]   for jj in j-1:j+1 )
        Pfˡ  = MVector{3}(  Pf[jj]   for jj in j-1:j+1 )
        ϕ0ˡ  = SVector{3}(  ϕ0[jj]   for jj in j-1:j+1 )
        tagˡ = SVector{3}( BC.Pf[jj] for jj in j-1:j+1 )

        # Residuals
        r[num.Pt[j]] = continuity_local(Vyˡ, Ptˡ, Pfˡ, ϕ0ˡ, tagˡ, p, Δy, Δt)

        # Jacobian
        fill!(∂R∂Vy, 0.0)
        fill!(∂R∂Pt, 0.0)
        fill!(∂R∂Pf, 0.0)
        forwarddiff_gradients!(continuity_local, Duplicated(Vyˡ, ∂R∂Vy), Duplicated(Ptˡ, ∂R∂Pt), Duplicated(Pfˡ, ∂R∂Pf), Const(ϕ0ˡ), Const(tagˡ), Const(p), Const(Δy), Const(Δt))

        # Pt --- Vys
        connect = SVector{2}( num.Vys[jj]   for jj in j:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pt[j]>0
                M[num.Pt[j], connect[jj]] = ∂R∂Vy[jj] 
            end
        end

        # Vys --- Pt
        connect = SVector{3}( num.Pt[jj]   for jj in j-1:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pt[j]>0
                M[num.Pt[j], connect[jj]] = ∂R∂Pt[jj] 
            end
        end

        # Vys --- Pf
        connect = SVector{3}( num.Pf[jj]   for jj in j-1:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pt[j]>0
                M[num.Pt[j], connect[jj]] = ∂R∂Pf[jj] 
            end
        end

        # Residuals
        r[num.Pf[j]] = fluid_continuity_local(Vyˡ, Ptˡ, Pfˡ, ϕ0ˡ, tagˡ, p, Δy, Δt)
        
        # Jacobian
        fill!(∂R∂Vy, 0.0)
        fill!(∂R∂Pt, 0.0)
        fill!(∂R∂Pf, 0.0)
        forwarddiff_gradients!(fluid_continuity_local, Duplicated(Vyˡ, ∂R∂Vy), Duplicated(Ptˡ, ∂R∂Pt), Duplicated(Pfˡ, ∂R∂Pf), Const(ϕ0ˡ), Const(tagˡ), Const(p), Const(Δy), Const(Δt))

        # Pt --- Vys
        connect = SVector{2}( num.Vys[jj]   for jj in j:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pf[j]>0
                M[num.Pf[j], connect[jj]] = ∂R∂Vy[jj] 
            end
        end

        # Vys --- Pt
        connect = SVector{3}( num.Pt[jj]   for jj in j-1:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pf[j]>0
                M[num.Pf[j], connect[jj]] = ∂R∂Pt[jj] 
            end
        end

        # Vys --- Pf
        connect = SVector{3}( num.Pf[jj]   for jj in j-1:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pf[j]>0
                M[num.Pf[j], connect[jj]] = ∂R∂Pf[jj] 
            end
        end
    end
end

function main_Havlin(nc)

    @load "havlin_ac.jld2" por_snapshot z

    # Paramaters
    p = (
        m  = -1.0,
        ϕ0 = 4e-2,
        μl = 1.0,
        ηs = 1e16,
        a  = 5e-3,
        ρs = 3200.0,
        ρl = 3000.0,
        gy = -9.8,
        yfact = 10, # model size relative to compaction length
    )

    # Compaction length
    lc = compaction_length(p.ϕ0, p)
    @info "Compaction length: $(lc) m --- Model size: $(p.yfact*lc) m"

    # Time domain
    nt = 1000
    Δt = 1e6

    # Space domain
    y   = (min=-p.yfact*lc, max=0.0)
    Δy  = (y.max - y.min)/nc
    yce = LinRange(y.min-Δy/2, y.max+Δy/2, nc+2)
    yv  = LinRange(y.min, y.max, nc+1)

    # Non-linear solver
    niter = 50
    tol   = 1e-8
    nr0   = 1.0

    # Arrays
    ϕ    = p.ϕ0*ones(nc+2)
    ϕ0   = p.ϕ0*ones(nc+2)
    dϕdt =     zeros(nc+2)
    Vys  =     zeros(nc+3)
    Vyf  =     zeros(nc+3)
    qDy  =     zeros(nc+3)
    Pt   =     zeros(nc+2)
    Pf   =     zeros(nc+2)

    # Initial conditions
    Pt .= -reverse(cumsum(((1 .- ϕ0).*p.ρs .+ ϕ0.*p.ρl)  * p.gy  )*Δy)
    Pf .= Pt
    Vys[3] = 1e-6

    display(lines(Pt[:], yce./1e3))

    # Boundary conditions
    BC  = ( Vys = zeros(Int64, nc+3), Pf = zeros(Int64, nc+2))  
    BC.Vys[[end]] .= 2 # set Neumann
    BC.Vys[[1]]   .= 1 # set Dirichlet
    BC.Pf[[end]] .= 2 # set Neumann
    BC.Pf[[1]]   .= 1 # set weird lower BC

    # Numbering
    num = (Vys = zeros(Int64, nc+3), Pt = zeros(Int64, nc+2), Pf = zeros(Int64, nc+2))
    num.Vys[3:end-1] .= 1:nc # assumes the lower BC is conforming Dirichlet, so it's not a dof
    num.Pt[2:end-1] .= maximum(num.Vys)+1:maximum(num.Vys)+nc 
    num.Pf[2:end-1] .= maximum(num.Pt)+1:maximum(num.Pt)+nc
    ndof = (Vys=sum(num.Vys.!=0), Pt=sum(num.Pt.!=0), Pf=sum(num.Pf.!=0), tot=maximum(num.Pf))

    # Sparse matrices
    r = zeros(ndof.tot)
    x = zeros(ndof.tot)
    M = ExtendableSparseMatrix(ndof.tot, ndof.tot)
    
    # Initial guess
    x[num.Vys[num.Vys.>0]] .= Vys[num.Vys.>0]
    x[num.Pt[num.Pt.>0]] .= Pt[num.Pt.>0]
    x[num.Pf[num.Pf.>0]] .= Pf[num.Pf.>0]
    
    # Time loop
    for it=1:nt
    
        @printf("Time step %04d --- time %1.3f y --- Pt = %1.2f MPa --- Pf = %1.2f MPa --- ϕ = %1.2e\n", it, it*Δt/yr, Pt[2]/1e6, Pf[2]/1e6, ϕ[2]) 
        ϕ0 .= ϕ
        
        # Newton iterations
        for iter = 1:niter

            momentum!(M, r, Vys, Pt, Pf, ϕ0, BC, num, p, Δy, Δt)
            continuity!(M, r, Vys, Pt, Pf, ϕ0, BC, num, p, Δy, Δt)
           
            if iter==1 nr0 = norm(r) end
            @printf("Iteration: %3d - abs. res. = %1.4e - rel. res. = %1.4e\n", iter, norm(r)/sqrt(length(r)), norm(r)/nr0 )
            min(norm(r)/sqrt(length(r)), norm(r)/nr0) < tol && break 

            x -= M \ r

            Vys[num.Vys.>0] .= x[num.Vys[num.Vys.>0]]
            Pt[num.Pt.>0] .= x[num.Pt[num.Pt.>0]]
            Pf[num.Pf.>0] .= x[num.Pf[num.Pf.>0]]
        end

        dϕdt .= [porosity_rate(Pt[j], Pf[j], ϕ0[j], p) for j in eachindex(dϕdt)]
        ϕ[2:end-1] .+= dϕdt[2:end-1] * Δt

        Darcy!(qDy, Vyf, ϕ, Pf, Vys, BC, num, p, Δy, Δt)


        # ------------------------------- #
        if mod(it, 50) == 0 || it==1
            fig = Figure()
            
            ax1 = Axis(fig[1,1], xlabel=L"$Pt$, $Pf$ (MPa)", ylabel=L"$y$ (km)")
            lines!(ax1, Pt[2:end-1]./1e6, yce[2:end-1]./1e3)
            lines!(ax1, Pf[2:end-1]./1e6, yce[2:end-1]./1e3, linestyle=:dash)
            

            # τyy = 2 * p.ηs .* 2/3 .* diff(Vys)/Δy
            # ϕv = 1/2*(ϕ[1:end-1] .+ ϕ[2:end])
            # ρt = p.ρl .* ϕv .+ (1 .- ϕv) .* p.ρs
            # ry = diff(τyy[2:end-1])/Δy .- 0*diff(Pt[2:end-1])/Δy .+ 0*ρt[2:end-1] .* p.gy
            # # ry = zero(Vys)
            # # ry[num.Vys.>0] = x[num.Vys[num.Vys.>0]]

            # ax1 = Axis(fig[1,1], xlabel=L"$Pt$, $Pf$ (MPa)", ylabel=L"$y$ (km)")
            # lines!(ax1, ry[1:end-0], yv[2:end-1]./1e3)
            # # lines!(ax1, Pf[2:end-1]./1e6, yce[2:end-1]./1e3, linestyle=:dash)
            

            ax2 = Axis(fig[1,2], xlabel=L"$\Delta P$ (MPa)", ylabel=L"$y$ (km)")
            lines!(ax2, ((Pf .- Pt) ./ (1 .-ϕ))[2:end-1]./1e6, yce[2:end-1]./1e3)

            ax3 = Axis(fig[2,1], xlabel=L"$Vys$ (cm/y)", ylabel=L"$y$ (km)")
            # lines!(ax3, Vys[2:end-1]*cmy, yv./1e3)
            lines!(ax3, Vyf[3:end-2]*cmy, yv[2:end-1]./1e3)
            # lines!(ax3, [3:end-2]*cmy, yv[2:end-1]./1e3)

            @show Vyf[end-3]*cmy, ϕ[end-3], qDy[end-3]*cmy
            vs = (Vyf[end-3] - qDy[end-3]/ϕ[end-3])*cmy
            @show vs

            @show Vyf[3]*cmy, ϕ[3], qDy[3]*cmy
            vs = (Vyf[3] - qDy[3]/ϕ[3])*cmy
            @show vs

            ax4 = Axis(fig[2,2], xlabel=L"$\phi$", ylabel=L"$y$ (km)")
            lines!(ax4, por_snapshot[2:end-1], -z[2:end-1]./1e3, color=:green, label=L"$\phi$ Paris")
            step = 20
            scatter!(ax4, ϕ[2:step:end-1], yce[2:step:end-1]./1e3, label=L"$\phi$ Frankfurt")
            axislegend(position=:rb)

            display(fig)
        end

        # @save "Havin_test0.jld2"  yce yv Pf Pt ϕ Vys
    end

end

main_Havlin(2000)