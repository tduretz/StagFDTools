using CairoMakie, StaticArrays, ExtendableSparse, LinearAlgebra, Printf, JLD2
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian

# Try to get bulk elasticity running but it does not !

yr  = 365.25*24*3600
cmy = 100*yr

perm(ϕ, a) = a^2 * abs(ϕ)^2.7 / 58 

bulk(ϕ, ηs, m) = ηs*abs(ϕ)^m

function compaction_length(ϕ0, p)
    k0 = perm(ϕ0, p.a)
    χ0 = bulk(ϕ0, p.ηs, p.m)
    return sqrt((k0/p.μl) * (χ0 + 4/3*p.ηs)) 
end

function porosity_rate(ϕ, Pt, Pf, Pt0, Pf0, p, Δt)
    χ       = bulk(ϕ, p.ηs, p.m)
    KΦ      = p.KΦ
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    return ((Pf - Pt)/χ + (dPtdt - dPfdt)/KΦ) * 1, χ 
end

function PorosityResidual(Φ, Φ0, Pt, Pf, Pt0, Pf0, p, Δt) 
    dΦdt = porosity_rate(Φ, Pt, Pf, Pt0, Pf0, p, Δt)[1]
    r    = Φ - (Φ0  + dΦdt * Δt)  
    return r 
end

function porosity(Φ0, Pt, Pf, Pt0, Pf0, p, Δt) 
    dΦdt, ηΦ = porosity_rate(Φ0, Pt, Pf, Pt0, Pf0, p, Δt)
    Φ        = Φ0  + dΦdt * Δt
    r0       = 1.0
    for iter=1:2
        J     = forwarddiff_gradient(PorosityResidual, Φ, Const(Φ0), Const(Pt), Const(Pf), Const(Pt0), Const(Pf0), Const(p), Const(Δt) )
        r     = J.val[1]
        if iter==1 r0 = abs(r) + 1e-10 end
        # @show iter, abs(r), abs(r)/r0
        # if min(abs(r), abs(r)/r0 ) < 1e-10 break end
        Φ    -=  J.derivs[1] \ J.val[1]
    end
    dΦdt, ηΦ = porosity_rate(Φ, Pt, Pf, Pt0, Pf0, p, Δt)
    return Φ, dΦdt, ηΦ 
end

function update_porosity( ϕ0, dϕdt, Δt )
    return ϕ0 + Δt * dϕdt
end

function deviator!(τyy, Vys, τyy0, BC, num, p, Δy, Δt)
    for j = 2:length(τyy)-1

        Vy  = MVector{2}(   Vys[jj]   for jj in j:j+1   )
        tag = SVector{2}( BC.Vy[jj]   for jj in j:j+1)

        # Neumann BC for surface
        if tag[end] == 2
            Vy[end] = Vy[2]
        end

        ∂Vy∂y = (Vy[2] - Vy[1]) / Δy
        divV  = ∂Vy∂y
        ε̇yy′  = ∂Vy∂y - 1/3*divV

        ηe    = p.Gs*Δt
        ηve   = 1 / (1/(ηe) + 1/(p.ηs) )

        τyy[j] = 2 * ηve * (ε̇yy′ + τyy0[j]/(2*ηe))

    end
end

function momentum_local(Vy, Pt, Pf, τyy0, Pt0, Pf0, ϕ0, tag, tag_pt, tag_pf, p, Δy, Δt)

    # Neumann BC for surface
    if tag[end] == 2
        Vy[end] = Vy[2]
    end

    if tag_pt[end] == 1
        Pt[end] = -Pt[end-1]
    end

    if tag_pf[end] == 1
        Pf[end] = -Pf[end-1]
    end

    # Phi on Vy points
    dϕdt    = SVector{2}( porosity(ϕ0[i], Pt[i], Pf[i], Pt0[i], Pf0[i], p, Δt)[2] for i in 1:2 )
    ϕ       = SVector{2}( porosity(ϕ0[i], Pt[i], Pf[i], Pt0[i], Pf0[i], p, Δt)[1] for i in 1:2 )
    ϕy      = ((ϕ[2:end] + ϕ[1:end-1]) / 2)[1] 

    # Kinematics
    ∂Vy∂y = SVector{2}( @. (Vy[2:end] - Vy[1:end-1]) / Δy )
    ε̇yy′  = SVector{2}( @. ∂Vy∂y - 1/3*(∂Vy∂y) )

    # Rheology
    ηe    = p.Gs*Δt
    ηve   = 1 / (1/(ηe) + 1/(p.ηs) )
    τyy   = SVector{2}( @. 2 * ηve * (ε̇yy′ + τyy0/(2*ηe)) )  #* (1-ϕy)

    # Rheology
    ∂τyy∂y = ((τyy[2:end] - τyy[1:end-1]) / Δy)[1] 
    ∂Pt∂y  = (( Pt[2:end] -  Pt[1:end-1]) / Δy)[1] 
    
    # Body force
    ρt     = (1 - ϕy) * p.ρs + ϕy * p.ρl

    # if tag_pt[end] == 1
    #     @show Vy
    #     @show ε̇yy′ + τyy0/(2*ηe)
    #     @show ηve
    #     @show τyy
    #     @show ∂τyy∂y
    # end 

    fy = - (∂τyy∂y - ∂Pt∂y + ρt*p.gy)

    # @show fy / (1e6 / 1000)

    return fy
end

function momentum!(M, r, Vys, Pt, Pf, τyy0, Pt0, Pf0, ϕ0, BC, num, p, Δy, Δt)

    ∂R∂Vy   = @MVector zeros(3)
    ∂R∂Pt   = @MVector zeros(2)
    ∂R∂Pf   = @MVector zeros(2)

    for j = 2:length(Vys)-1

        # Local stencil
        Vyˡ  = MVector{3}(   Vys[jj]   for jj in j-1:j+1 )
        Ptˡ  = MVector{2}(    Pt[jj]   for jj in j-1:j   )
        Pfˡ  = MVector{2}(    Pf[jj]   for jj in j-1:j   )
        Pt0ˡ = SVector{2}(   Pt0[jj]   for jj in j-1:j   )
        Pf0ˡ = SVector{2}(   Pf0[jj]   for jj in j-1:j   )
        τyy0ˡ= SVector{2}(  τyy0[jj]   for jj in j-1:j   )
        ϕ0ˡ  = SVector{2}(    ϕ0[jj]   for jj in j-1:j   )
        tagˡ = SVector{3}( BC.Vy[jj]   for jj in j-1:j+1 )
        tag_ptˡ = SVector{2}( BC.Pt[jj]   for jj in j-1:j )
        tag_pfˡ = SVector{2}( BC.Pf[jj]   for jj in j-1:j )

        # Residual
        if num.Vy[j]>0
            r[num.Vy[j]] = momentum_local(Vyˡ, Ptˡ, Pfˡ, τyy0ˡ, Pt0ˡ, Pf0ˡ, ϕ0ˡ, tagˡ, tag_ptˡ, tag_pfˡ, p, Δy, Δt)
        end

        # Jacobian
        fill!(∂R∂Vy, 0.0)
        fill!(∂R∂Pt, 0.0)
        fill!(∂R∂Pf, 0.0)
        forwarddiff_gradients!(momentum_local, Duplicated(Vyˡ, ∂R∂Vy), Duplicated(Ptˡ, ∂R∂Pt), Duplicated(Pfˡ, ∂R∂Pf), Const(τyy0ˡ), Const(Pt0ˡ), Const(Pf0ˡ), Const(ϕ0ˡ), Const(tagˡ), Const(tag_ptˡ), Const(tag_pfˡ), Const(p), Const(Δy), Const(Δt))

        # Vy --- Vy
        connect = SVector{3}( num.Vy[jj]   for jj in j-1:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Vy[j]>0
                M[num.Vy[j], connect[jj]] = ∂R∂Vy[jj] 
            end
        end

        # Vy --- Pt
        connect = SVector{2}( num.Pt[jj]   for jj in j-1:j )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Vy[j]>0
                M[num.Vy[j], connect[jj]] = ∂R∂Pt[jj] 
            end
        end

        # Vy --- Pf
        connect = SVector{2}( num.Pf[jj]   for jj in j-1:j )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Vy[j]>0
                M[num.Vy[j], connect[jj]] = ∂R∂Pf[jj] 
            end
        end
        
    end
end

function continuity_local(Vy, Pt, Pf, Pt0, Pf0, ϕ0, tag, tag_pt, p, Δy, Δt)

    dlnρsdt = @SVector zeros(3)
   
    # Phi 
    dϕdt    = SVector{3}( porosity(ϕ0[i], Pt[i], Pf[i], Pt0[i], Pf0[i], p, Δt)[2] for i in 1:3 )
    ϕ       = SVector{3}( porosity(ϕ0[i], Pt[i], Pf[i], Pt0[i], Pf0[i], p, Δt)[1] for i in 1:3 )

    dPtdt   = SVector{3}(@. (Pt - Pt0) / Δt)
    dPfdt   = SVector{3}(@. (Pf - Pf0) / Δt)
    # dPsdt   = SVector{3}(@. 1/(1-ϕ) * (dPtdt - ϕ*dPfdt) ) # approx
    dPsdt   = SVector{3, Float64}( @. dϕdt*(Pt - Pf*ϕ)/(1-ϕ)^2 + (dPtdt - ϕ*dPfdt - Pf*dϕdt) / (1 - ϕ) )

    dlnρsdt = SVector{3}(dPsdt / p.Ks) 

    # Solid divergence
    divVs   = (Vy[2] - Vy[1]) / Δy

    fp =  dlnρsdt[2] - dϕdt[2]/(1-ϕ[2]) + divVs

    # @info "node"
    # @show dlnρsdt[2]
    # @show ϕ[2]
    # @show dϕdt[2] / (1 / 1e10)
    # @show divVs / (1 / 1e10)
    # @show fp / (1 / 1e10)
    # error()

    if tag_pt[2]==1 # Base
        r = Pt[2] - Pf[2]
    else
        r = dlnρsdt[2] - dϕdt[2]/(1-ϕ[2]) + divVs
    end
    return r
end

function fluid_continuity_local(Vy, Pt, Pf, Pt0, Pf0, ϕ0, tag, p, Δy, Δt)

    dlnρfdt = @SVector zeros(3)

    # # BC
    # if tag[end] == 2 # Top: no flux
    #     Pf[end] = Pf[end-1] + ρlg * Δy
    # elseif tag[end] == 1 # Top: Dirchlet
    #     Pf[end] = -Pf[end-1] 
    # end
    # if tag[1] == 1 # Bottom: try to set Pf = Pt such that ϕ = ϕ0 
    #     # Pf[2] =  Pt[2]/2 # ????????
    #     ϕS     = (ϕ[1] + ϕ[2])/2
    #     ρtg    = ((1-ϕS)*p.ρs + ϕS*p.ρl) * p.gy
    #     lc     = compaction_length(p.ϕ0, p)
    #     y_base = -p.yfact*lc
    #     Pt_bot = (y_base-3Δy/2)*ρtg


    #     # Pt[1]  = 2*Pt_bot - Pt[2]
    #     # Pf[2]  =  (Pt[1]+Pt[2])/2 / 2
    #     # Pf[1]    = Pt[1]+Pt[2]-Pf[2]
    #     Pf[1]  = 2*Pt_bot - Pf[2]

    #     Pf_bot = 1.37962e9
    #     Pf[1]  = 2*Pf_bot - Pf[2]
       
    #     # Pf[2]  =  Pt[2]
    # end

    if tag[1] == 1
        Pf_bot = 1.2841920000000002e9
        Pt_bot = 1.366380288e9
        Pf[1] = 2*Pf_bot - Pf[2]
        Pt[1] = 2*Pt_bot - Pt[2]
    end

    if tag[end] == 1
        Pf[end] = -Pf[end-1]
        Pt[end] = -Pt[end-1]
    end



    # Phi 
    dϕdt    = SVector{3}( porosity(ϕ0[i], Pt[i], Pf[i], Pt0[i], Pf0[i], p, Δt)[2] for i in 1:3 )
    ϕ       = SVector{3}( porosity(ϕ0[i], Pt[i], Pf[i], Pt0[i], Pf0[i], p, Δt)[1] for i in 1:3 )

    dPfdt   = SVector{3}(@. (Pf - Pf0) / Δt)
    dlnρfdt = SVector{3}(dPfdt / p.Kf) 

    # Buoyancy
    ρlg     = p.ρl * p.gy

    # Darcy
    k       = SVector{3}( perm.(ϕ, p.a) )
    k_μ     = SVector{2}( @. (k[2:end] + k[1:end-1]) / 2 / p.μl) 
    qy      = SVector{2}( @. -k_μ .* ((Pf[2:end] - Pf[1:end-1])/ Δy - ρlg) )

    @show k_μ

    # Solid divergence
    divVs   = (Vy[2] - Vy[1]) / Δy

    # Darcy flux divergence
    divqD   = (qy[2] - qy[1]) / Δy

    return ϕ[2]*dlnρfdt[2] + dϕdt[2] + ϕ[2]*divVs + divqD
end

function continuity!(M, r, Vys, Pt, Pf, Pt0, Pf0, ϕ0, BC, num, p, Δy, Δt)

    ∂R∂Vy   = @MVector zeros(2)
    ∂R∂Pt   = @MVector zeros(3)
    ∂R∂Pf   = @MVector zeros(3)

    for j = 2:length(Pt)-1

        # Local stencil
        Vyˡ  = MVector{2}(   Vys[jj]   for jj in j:j+1   )
        Ptˡ  = MVector{3}(    Pt[jj]   for jj in j-1:j+1 )
        Pfˡ  = MVector{3}(    Pf[jj]   for jj in j-1:j+1 )
        Pf0ˡ = SVector{3}(   Pf0[jj]   for jj in j-1:j+1 )
        Pt0ˡ = SVector{3}(   Pt0[jj]   for jj in j-1:j+1 )
        ϕ0ˡ  = SVector{3}(    ϕ0[jj]   for jj in j-1:j+1 )
        tagˡ = SVector{3}( BC.Pf[jj]   for jj in j-1:j+1 )
        tag_ptˡ = SVector{3}( BC.Pt[jj]   for jj in j-1:j+1 )

        # Residuals
        r[num.Pt[j]] = continuity_local(Vyˡ, Ptˡ, Pfˡ, Pt0ˡ, Pf0ˡ, ϕ0ˡ, tagˡ, tag_ptˡ, p, Δy, Δt)

        # Jacobian
        fill!(∂R∂Vy, 0.0)
        fill!(∂R∂Pt, 0.0)
        fill!(∂R∂Pf, 0.0)
        forwarddiff_gradients!(continuity_local, Duplicated(Vyˡ, ∂R∂Vy), Duplicated(Ptˡ, ∂R∂Pt), Duplicated(Pfˡ, ∂R∂Pf), Const(Pt0ˡ), Const(Pf0ˡ), Const(ϕ0ˡ), Const(tagˡ), Const(tag_ptˡ), Const(p), Const(Δy), Const(Δt))

        # Pt --- Vy
        connect = SVector{2}( num.Vy[jj]   for jj in j:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pt[j]>0
                M[num.Pt[j], connect[jj]] = ∂R∂Vy[jj] 
            end
        end

        # Vy --- Pt
        connect = SVector{3}( num.Pt[jj]   for jj in j-1:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pt[j]>0
                M[num.Pt[j], connect[jj]] = ∂R∂Pt[jj] 
            end
        end

        # Vy --- Pf
        connect = SVector{3}( num.Pf[jj]   for jj in j-1:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pt[j]>0
                M[num.Pt[j], connect[jj]] = ∂R∂Pf[jj] 
            end
        end

        # Residuals
        r[num.Pf[j]] = fluid_continuity_local(Vyˡ, Ptˡ, Pfˡ, Pt0ˡ, Pf0ˡ, ϕ0ˡ, tagˡ, p, Δy, Δt)
        
        # Jacobian
        fill!(∂R∂Vy, 0.0)
        fill!(∂R∂Pt, 0.0)
        fill!(∂R∂Pf, 0.0)
        forwarddiff_gradients!(fluid_continuity_local, Duplicated(Vyˡ, ∂R∂Vy), Duplicated(Ptˡ, ∂R∂Pt), Duplicated(Pfˡ, ∂R∂Pf), Const(Pt0ˡ), Const(Pf0ˡ), Const(ϕ0ˡ), Const(tagˡ), Const(p), Const(Δy), Const(Δt))

        # Pt --- Vy
        connect = SVector{2}( num.Vy[jj]   for jj in j:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pf[j]>0
                M[num.Pf[j], connect[jj]] = ∂R∂Vy[jj] 
            end
        end

        # Vy --- Pt
        connect = SVector{3}( num.Pt[jj]   for jj in j-1:j+1 )
        for jj in eachindex(connect)
            if (connect[jj]>0) && num.Pf[j]>0
                M[num.Pf[j], connect[jj]] = ∂R∂Pt[jj] 
            end
        end

        # Vy --- Pf
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

    open_top = true

    # Parameters
    p = (
        m     = -1.0*1,
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

    # Boundary conditions
    BC  = ( Vy = zeros(Int64, nc+3), Pf = zeros(Int64, nc+2), Pt = zeros(Int64, nc+2))  
    BC.Vy[[1]]   .= 1 # set Dirichlet
    if open_top
        BC.Vy[[end]] .= 2 # set Neumann
    else
        BC.Vy[[end]] .= 1 # set Neumann
    end
    # BC.Pf[[end]] .= 2 # set Neumann
    BC.Pf[[end]] .= 1 # set Dirichlet

    BC.Pf[[end]] .= 1 # set Dirichlet
    BC.Pf[[1]]   .= 1 # set Dirichlet

    if !open_top
        # BC.Pt[[end-1]]   .= 1 # set constraint
    end

    BC.Pt[[end]]   .= 1
    BC.Pt[[1]]     .= 1

    # Initial conditions
    ρt    = p.ρs*(1-p.ϕ0) + p.ϕ0*p.ρl
    P_top = 0*20e6
    for j in (nc+2-1):-1:2
        Pf[j] = Pf[j+1] - p.ρl * p.gy .* Δy
        Pt[j] = Pt[j+1] - ρt   * p.gy .* Δy
    end

    # Vy[3] = 1e-6

    # Numbering
    num = (Vy = zeros(Int64, nc+3), Pt = zeros(Int64, nc+2), Pf = zeros(Int64, nc+2))
    
    if open_top
        num.Vy[3:end-1] .= 1:nc # assumes the lower BC is conforming Dirichlet, so it's not a dof
    else
        num.Vy[3:end-2] .= 1:nc-1 # assumes the lower BC is conforming Dirichlet, so it's not a dof
    end
    num.Pt[2:end-1] .= maximum(num.Vy)+1:maximum(num.Vy)+nc 
    num.Pf[2:end-1] .= maximum(num.Pt)+1:maximum(num.Pt)+nc
    ndof = (Vy=sum(num.Vy.!=0), Pt=sum(num.Pt.!=0), Pf=sum(num.Pf.!=0), tot=maximum(num.Pf))

    # Sparse matrices
    r = zeros(ndof.tot)
    x = zeros(ndof.tot)
    M = ExtendableSparseMatrix(ndof.tot, ndof.tot)
    
    # Initial guess
    x[num.Vy[num.Vy.>0]] .= Vy[num.Vy.>0]
    x[num.Pt[num.Pt.>0]] .= Pt[num.Pt.>0]
    x[num.Pf[num.Pf.>0]] .= Pf[num.Pf.>0]
    
    r0 = 1

    # Time loop
    for it=1:nt
    
        @printf("Time step %04d --- time %1.3f y --- Pt = %1.2f MPa --- Pf = %1.2f MPa --- ϕ = %1.2e\n", it, it*Δt/yr, Pt[2]/1e6, Pf[2]/1e6, ϕ[2]) 
        ϕ0   .= ϕ
        Pf0  .= Pf
        Pt0  .= Pt
        τyy0 .= τyy
        
        # Newton iterations
        for iter = 1:niter
 
            momentum!(M, r, Vy, Pt, Pf, τyy0, Pt0, Pf0, ϕ0, BC, num, p, Δy, Δt)
            continuity!(M, r, Vy, Pt, Pf, Pt0, Pf0, ϕ0, BC, num, p, Δy, Δt)

            if iter==1
                r0 = copy(r)
            end

            if iter==1 nr0 = norm(r) end
            @printf("Time step %04d ---Iteration: %3d - abs. res. = %1.4e - rel. res. = %1.4e\n", it, iter, norm(r)/sqrt(length(r)), norm(r)/nr0 )
            min(norm(r)/sqrt(length(r)), norm(r)/nr0) < tol && break 

            # Full Newton correction
            δx = .- M \ r

            # Line search find α such that r(x + α * δx) is mimimized
            x_i  = copy(x)
            αvec = [0.01 0.05 0.1 0.2 0.5 0.75 1.0]
            rvec = zero(αvec)
            for ils in eachindex(αvec)
                x .= x_i + αvec[ils] * δx
                Vy[num.Vy.>0] .= x[num.Vy[num.Vy.>0]]
                Pt[num.Pt.>0] .= x[num.Pt[num.Pt.>0]]
                Pf[num.Pf.>0] .= x[num.Pf[num.Pf.>0]]
                momentum!(M, r, Vy, Pt, Pf, τyy0, Pt0, Pf0, ϕ0, BC, num, p, Δy, Δt)
                continuity!(M, r, Vy, Pt, Pf, Pt0, Pf0, ϕ0, BC, num, p, Δy, Δt)
                rvec[ils] = norm(r)
            end
            imin = argmin(rvec)
            
            # Apply correction
            x = x_i .+ αvec[imin] * δx

            Vy[num.Vy.>0] .= x[num.Vy[num.Vy.>0]]
            Pt[num.Pt.>0] .= x[num.Pt[num.Pt.>0]]
            Pf[num.Pf.>0] .= x[num.Pf[num.Pf.>0]]
        end

        deviator!(τyy, Vy, τyy0, BC, num, p, Δy, Δt)

        dϕdt       .= [porosity(ϕ0[j], Pt[j], Pf[j], Pt0[j], Pf0[j], p, Δt)[2] for j in eachindex(dϕdt)]
        ϕ[2:end-1] .= [porosity(ϕ0[j], Pt[j], Pf[j], Pt0[j], Pf0[j], p, Δt)[1] for j in 2:nc+2-1 ]

        # ------------------------------- #

        rVy0 = zero(Vy)
        for i in eachindex(Vy)
            if num.Vy[i]>0
                rVy0[i] = r0[num.Vy[i]]
            end
        end
        rPt0 = zero(Pt)
        for i in eachindex(Pt)
            if num.Pt[i]>0
                rPt0[i] = r0[num.Pt[i]]
            end
        end
        rPf0 = zero(Pf)
        for i in eachindex(Pf)
            if num.Pf[i]>0
                rPf0[i] = r0[num.Pf[i]]
            end
        end

        @save "havlin_Stag1D_debug.jld2" Pt Pf τyy ϕ Vy Pt0 Pf0 τyy0 ϕ0 rVy0 rPt0 rPf0

       if mod(it, 100) == 0 || it==1
            fig = Figure()
            
            # ax1 = Axis(fig[1,1], xlabel=L"$Pt$, $Pf$ (MPa)", ylabel=L"$y$ (km)")
            # lines!(ax1, Pt[2:end-1]./1e6, yce[2:end-1]./1e3)
            # lines!(ax1, Pf[2:end-1]./1e6, yce[2:end-1]./1e3, linestyle=:dash)

            ax1 = Axis(fig[1,1], xlabel=L"$\tau_{yy}$ (MPa)", ylabel=L"$y$ (km)")
            lines!(ax1, τyy[2:end-1]./1e6, yce[2:end-1]./1e3)

            
            ax2 = Axis(fig[1,2], xlabel=L"$\Delta P$ (MPa)", ylabel=L"$y$ (km)")
            lines!(ax2, ((Pf .- Pt) ./ (1 .-ϕ))[2:end-1]./1e6, yce[2:end-1]./1e3)


            ax3 = Axis(fig[2,1], xlabel=L"$Vy$ (cm/y)", ylabel=L"$y$ (km)")
            lines!(ax3, Vy[2:end-1]*cmy, yv./1e3)


            ax4 = Axis(fig[2,2], xlabel=L"$\phi$", ylabel=L"$y$ (km)")
            # lines!(ax4, por_snapshot[2:end-1], -z[2:end-1]./1e3, color=:green, label=L"$\phi$ Paris")
            step = 1
            lines!(ax4, ϕ[2:step:end-1], yce[2:step:end-1]./1e3, label=L"$\phi$ Stag 1D")
            
        
            @load "havlin_DR_debug.jld2" Pt Pf τyy ϕ Vy Pt0 Pf0 τyy0 ϕ0  
            lines!(ax1, τyy[2:end-1]./1e6, yce[2:end-1]./1e3)
            lines!(ax2, ((Pf  .- Pt ) ./ (1 .-ϕ ))[2:end-1]./1e6, yce[2:end-1]./1e3)
            lines!(ax3, Vy[2:end-1] *cmy, yv./1e3)
            lines!(ax4,  ϕ[2:step:end-1], yce[2:step:end-1]./1e3, label=L"$\phi$ DR")
            
            axislegend(position=:rb)

            display(fig)

        end
    end

end

main_Havlin(51) #100