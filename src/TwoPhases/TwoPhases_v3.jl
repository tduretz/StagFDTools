struct Fields{Tx,Ty,Tp,Tpf}
    Vx::Tx
    Vy::Ty
    Pt::Tp
    Pf::Tpf
end

function Base.getindex(x::Fields, i::Int64)
    @assert 0 < i < 5 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
    i == 4 && return x.Pf
end

function Ranges(nc)     
    return (inx_Vx = 2:nc.x+2, iny_Vx = 3:nc.y+2, inx_Vy = 3:nc.x+2, iny_Vy = 2:nc.y+2, inx_c = 2:nc.x+1, iny_c = 2:nc.y+1, inx_v = 2:nc.x+2, iny_v = 2:nc.y+2, size_x = (nc.x+3, nc.y+4), size_y = (nc.x+4, nc.y+3), size_c = (nc.x+2, nc.y+2), size_v = (nc.x+3, nc.y+3))
end

function SMomentum_x_Generic(Vx_loc, Vy_loc, Pt, Pf, ΔP, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
    invΔx, invΔy = 1 / Δ.x, 1 / Δ.y

    # BC
    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)

    # Velocity gradient
    Dxx = ∂x(Vx) * invΔx
    Dyy = ∂y_inn(Vy) * invΔy
    Dxy = ∂y(Vx) * invΔy
    Dyx = ∂x_inn(Vy) * invΔx

    # Strain rate
    ε̇kk = @. Dxx + Dyy
    ε̇xx = @. Dxx - 1/3*ε̇kk
    ε̇yy = @. Dyy - 1/3*ε̇kk
    ε̇xy = @. 1/2 * ( Dxy + Dyx )

    # Average vertex to centroid
    ε̇̄xy  = av(ε̇xy)
    # Average centroid to vertex
    ε̇̄xx  = av(ε̇xx)
    ε̇̄yy  = av(ε̇yy)
    P̄t   = av(Pt)
    P̄f   = av(Pf)
    τ̄0xx = av(τ0.xx)
    τ̄0yy = av(τ0.yy)
    τ̄0xy = av(τ0.xy)

    # Effective strain rate
    Gc   = SVector{2, Float64}( materials.G[phases.c] )
    Gv   = SVector{2, Float64}( materials.G[phases.v] )
    tmpc = @. inv(2 * Gc * Δ.t)
    tmpv = @. inv(2 * Gv * Δ.t)
    ϵ̇xx  = @. ε̇xx[:,2] + τ0.xx[:,2] * tmpc
    ϵ̇yy  = @. ε̇yy[:,2] + τ0.yy[:,2] * tmpc
    ϵ̇̄xy  = @. ε̇̄xy[:]   + τ̄0xy[:]    * tmpc
    ϵ̇̄xx  = @. ε̇̄xx[:]   + τ̄0xx[:]    * tmpv
    ϵ̇̄yy  = @. ε̇̄yy[:]   + τ̄0yy[:]    * tmpv
    ϵ̇xy  = @. ε̇xy[2,:] + τ0.xy[2,:] * tmpv

    # Corrected pressure
    comp = materials.compressible
    Ptc  = SVector{2, Float64}( @. Pt[:,2] + comp * ΔP[:] )

    # Stress
    τxx = @MVector zeros(2)
    τxy = @MVector zeros(2)
    for i=1:2
        τxx[i] = (𝐷.c[i][1,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][1,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][1,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][1,4] + (1 - 𝐷.c[i][4,4])) * Pt[i,2] + 𝐷.c[i][1,5] * Pf[i,2]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                       * P̄t[i]  +  𝐷.v[i][3,5] * P̄f[i]
    end

    # Residual
    fx  = ( τxx[2]  - τxx[1] ) * invΔx
    fx += ( τxy[2]  - τxy[1] ) * invΔy
    fx -= ( Ptc[2]  - Ptc[1] ) * invΔx
    fx *= -1 * Δ.x * Δ.y

    return fx
end

function SMomentum_y_Generic(Vx_loc, Vy_loc, Pt, Pf, ΔP, Pt0, Pf0, Φ0, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
    invΔx, invΔy = 1 / Δ.x, 1 / Δ.y

    # BC
    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)

    # Velocity gradient
    Dxx = ∂x_inn(Vx) * invΔx
    Dyy = ∂y(Vy) * invΔy
    Dxy = ∂y_inn(Vx) * invΔy
    Dyx = ∂x(Vy) * invΔx

    # Strain rate
    ε̇kk = @. Dxx + Dyy
    ε̇xx = @. Dxx - 1/3*ε̇kk      
    ε̇yy = @. Dyy - 1/3*ε̇kk      
    ε̇xy = @. 1/2 * (Dxy + Dyx)

    # Average vertex to centroid
    ε̇̄xy  = av(ε̇xy)
    # Average centroid to vertex
    ε̇̄xx  = av(ε̇xx)
    ε̇̄yy  = av(ε̇yy)
    P̄t   = av( Pt)
    P̄f   = av( Pf)
    τ̄0xx = av(τ0.xx)
    τ̄0yy = av(τ0.yy)
    τ̄0xy = av(τ0.xy)
    
    # Effective strain rate
    Gc   = SVector{2, Float64}( materials.G[phases.c])
    Gv   = SVector{2, Float64}( materials.G[phases.v])
    tmpc = (2*Gc.*Δ.t)
    tmpv = (2*Gv.*Δ.t)
    ϵ̇xx  = @. ε̇xx[2,:] + τ0.xx[2,:] / tmpc
    ϵ̇yy  = @. ε̇yy[2,:] + τ0.yy[2,:] / tmpc
    ϵ̇̄xy  = @. ε̇̄xy[:]   + τ̄0xy[:]    / tmpc
    ϵ̇̄xx  = @. ε̇̄xx[:]   + τ̄0xx[:]    / tmpv
    ϵ̇̄yy  = @. ε̇̄yy[:]   + τ̄0yy[:]    / tmpv
    ϵ̇xy  = @. ε̇xy[:,2] + τ0.xy[:,2] / tmpv

    # Corrected pressure
    comp = materials.compressible
    Ptc  = SVector{2, Float64}( @. Pt[2,:]  + comp * ΔP.t[:] )
    Ptc0 = SVector{2, Float64}( @. Pt0[2,:] )
    Pfc  = SVector{2, Float64}( @. Pf[2,:]  + comp * ΔP.f[:] )
    Pfc0 = SVector{2, Float64}( @. Pf0[2,:] )

    # Porosity
    ηΦ      = SVector{2, Float64}( materials.ηΦ[phases.c])
    KΦ      = SVector{2, Float64}( materials.KΦ[phases.c])

    dPtdt   = (Ptc - Ptc0) / Δ.t
    dPfdt   = (Pfc - Pfc0) / Δ.t
    dΦdt    = @. (dPfdt - dPtdt)/KΦ + (Pfc - Ptc)/ηΦ

    # # @show size(dPtdt), size(dPfdt), size(Ptc), size(Pfc),size(KΦ),size(ηΦ), size(dΦdt)

    # THIS IF STATEMENT DOES NOT COMPILE WITH ENZYME
    # if materials.linearizeϕ
    #     Φ       = @. Φ0 
    # else
        Φ       = @. Φ0 + dΦdt*Δ.t
    # end

    # Density
    ρs   = SVector{2, Float64}( materials.ρs[phases.c])
    ρf   = SVector{2, Float64}( materials.ρf[phases.c])
    ρt   = @. (1-Φ) * ρs + Φ * ρf
    ρg   = materials.g[2] * 0.5*(ρt[1] + ρt[2])

    # @show ρg

    # error()

    # Stress
    τyy = @MVector zeros(2)
    τxy = @MVector zeros(2)
    for i=1:2
        τyy[i] = (𝐷.c[i][2,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][2,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][2,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][2,4] + (1 - 𝐷.c[i][4,4])) * Pt[2,i] + 𝐷.c[i][2,5] * Pf[2,i]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                       * P̄t[i]   + 𝐷.v[i][3,5] * P̄f[i]
    end

    # Residual
    fy  = ( τyy[2]  -  τyy[1] ) * invΔy
    fy += ( τxy[2]  -  τxy[1] ) * invΔx
    fy -= ( Ptc[2]  -  Ptc[1])  * invΔy
    fy += ρg
    fy *= -1 * Δ.x * Δ.y
    
    return fy
end

function Continuity(Vx, Vy, Pt, Pf, old, phase, materials, type_loc, bcv_loc, Δ)
    Pt0, Pf0, Φ0, ρs0, ρf0 = old
    invΔx   = 1 / Δ.x
    invΔy   = 1 / Δ.y
    Δt      = Δ.t
    ηΦ      = materials.ηΦ[phase]
    KΦ      = materials.KΦ[phase]
    Ks      = materials.Ks[phase]

    dPtdt   = SMatrix{3, 3, Float64}( @. (Pt - Pt0) / Δt )
    dPfdt   = SMatrix{3, 3, Float64}( @. (Pf - Pf0) / Δt )
    dΦdt    = SMatrix{3, 3, Float64}( @. (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ )
    if materials.linearizeϕ
        Φ       = SMatrix{3, 3, Float64}( Φ0*ones(3,3) ) 
    else
        Φ       = SMatrix{3, 3, Float64}( @. Φ0 + dΦdt*Δt )
    end

    if materials.single_phase
        Φ    = SMatrix{3, 3, Float64}( Φ0*zeros(3,3) )
        dΦdt = SMatrix{3, 3, Float64}( Φ0*zeros(3,3) )
    end

    dPsdt   = SMatrix{3, 3, Float64}( @. dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ) )
    dlnρsdt = SMatrix{3, 3, Float64}( 1/Ks * ( dPsdt ) )
    # dlnρsdt = (1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks) # approximation in Yarushina's paper

    # Single phase
    if materials.single_phase
        dΦdt    = 0.0
        dPsdt   = dPtdt 
        dlnρsdt = dPsdt / Ks
    end

    divVs   = (Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy 
    
    # if materials.oneway
    #     fp      = divVs
    # else
        # fp      = dlnρsdt[2,2] - dΦdt[2,2]/(1-Φ[2,2]) +   divVs
    # end

    # Solid mass / immobile solid mass: ∂ρim∂t  + ∇⋅(q) with q = ρim⋅Vs
    ρim0   = SMatrix{3, 3, Float64}( @. (1-Φ0) * ρs0 )
    lnρs   = SMatrix{3, 3, Float64}( @. log(ρs0) + Δt*dlnρsdt)
    ρs     = SMatrix{3, 3, Float64}( @. exp(lnρs) )
    ρim    = SMatrix{3, 3, Float64}( @. (1-Φ ) * ρs )
    ∂ρim∂t = (ρim[2,2] - ρim0[2,2]) / Δt
    qx     = SMatrix{2, 1, Float64}( @. (ρim[1:end-1,2] .+  ρim[2:end,2])/2 .* Vx[:,2] ) # Brucite paper, Fowler (1985)
    qy     = SMatrix{1, 2, Float64}( @. (ρim[2,1:end-1] .+  ρim[2,2:end])/2 .* Vy[2,:] ) # Brucite paper, Fowler (1985)
    fp     = ∂ρim∂t  +  (qx[2] - qx[1]) * invΔx + (qy[2] - qy[1]) * invΔy

    return fp
end


function FluidContinuity(Vx, Vy, Pt, Pf_loc, ΔPf_loc, old, phase, materials, kμ, type_loc, bcv_loc, Δ)
    
    Pt0, Pf0, Φ0, ρs0, ρf0 = old
    invΔx   = 1 / Δ.x
    invΔy   = 1 / Δ.y
    Δt      = Δ.t
    ηΦ      = materials.ηΦ[phase[2,2]]
    KΦ      = materials.KΦ[phase[2,2]] 
    Kf      = materials.Kf[phase[2,2]]
    Ks      = materials.Ks[phase[2,2]]
    n       = materials.n_CK[phase[2,2]] # Carman-Kozeny

    # Density - currently explicit in time (= using old fluid density)
    ρ0f  = SMatrix{3,3, Float64}( materials.ρf[phase])
    # ρgS  = materials.g[2] * 0.5*(ρ0f[2,2] + ρ0f[2,1])
    # ρgN  = materials.g[2] * 0.5*(ρ0f[2,2] + ρ0f[2,3])
    ρfg  = SVector{2, Float64}(@. materials.g[2] * 0.5*(ρ0f[2,1:end-1] + ρ0f[2,2:end]) )

    Pf    = SetBCPf1(Pf_loc, type_loc, bcv_loc, Δ, ρfg)

    dPtdt   = SMatrix{3,3, Float64}( (Pt .- Pt0) / Δt )
    dPfdt   = SMatrix{3,3, Float64}( (Pf .- Pf0) / Δt )
    dΦdt    = SMatrix{3,3, Float64}( (dPfdt .- dPtdt)/KΦ .+ (Pf .- Pt)/ηΦ )
    if materials.linearizeϕ
        Φ       = SMatrix{3,3, Float64}( Φ0 ) 
    else
        Φ       = SMatrix{3,3, Float64}( Φ0  .+ dΦdt*Δt )
    end

    dPsdt   = SMatrix{3, 3, Float64}( @. dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ) )
    dlnρsdt = SMatrix{3, 3, Float64}( 1/Ks * ( dPsdt ) )
    dlnρfdt = dPfdt[2,2] / Kf

    # Interpolate porosity to velocity nodes
    Φx = SVector{2, Float64}(@. (Φ[1:end-1,2] + Φ[2:end,2])/2 )
    Φy = SVector{2, Float64}(@. (Φ[2,1:end-1] + Φ[2,2:end])/2 )

    # ΦxW = Φ_x[1]#1/2*(Φ[1,2] + Φ[2,2])
    # ΦxE = Φ_x[2]#1/2*(Φ[3,2] + Φ[2,2])
    # ΦyS = Φ_y[1]#1/2*(Φ[2,1] + Φ[2,2])
    # ΦyN = Φ_y[2]#1/2*(Φ[2,3] + Φ[2,2])

    # Pf1 = SetBCPf1(Pf_loc.+ΔPf_loc, type_loc, bcv_loc, Δ, ρfg)

    # qxW = -kμ.xx[1]* ΦxW^n * ((Pf[2,2] - Pf[1,2]) * invΔx      )
    # qxE = -kμ.xx[2]* ΦxE^n * ((Pf[3,2] - Pf[2,2]) * invΔx      )
    # qyS = -kμ.yy[1]* ΦyS^n * ((Pf[2,2] - Pf[2,1]) * invΔy - ρgS) 
    # qyN = -kμ.yy[2]* ΦyN^n * ((Pf[2,3] - Pf[2,2]) * invΔy - ρgN) 

    # divqD = (    qxE -     qxW) * invΔx + (    qyN -     qyS) * invΔy

    qx = SVector{2, Float64}(@. -kμ.xx * Φx^n * ((Pf[2:end,2] - Pf[1:end-1,2]) * invΔx      )  )
    qy = SVector{2, Float64}(@. -kμ.yy * Φy^n * ((Pf[2,2:end] - Pf[2,1:end-1]) * invΔy - ρfg)  )

    divqD = (  qx[2] -   qx[1]) * invΔx + (  qy[2] -   qy[1]) * invΔy
    divVs = (Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy 

    # if type_loc[2,1] ==:Dirichlet
    #     printxy(type_loc)
    #     printxy(Pf)
    #     printxy(1/2*(Pf[:,1:end-1] .+ Pf[:,2:end]))
    # end

    if materials.oneway
        fp   = divqD
    else
        # fp = (Φ[2,2]*dlnρfdt + dΦdt[2,2]       + Φ[2,2]*divVs + divqD)

        # Total mass: ∂ρt∂t + ∇⋅(q) with q = ρf⋅qD + ρt⋅qD⋅V
        
        lnρs   = SMatrix{3, 3, Float64}( @. log(ρs0) + Δt*dlnρsdt)
        ρs     = SMatrix{3, 3, Float64}( @. exp(lnρs) )
        lnρf   = SMatrix{3, 3, Float64}( @. log(ρf0) + Δt*dlnρsdt)
        ρf     = SMatrix{3, 3, Float64}( @. exp(lnρf) )
        ρt     = SMatrix{3, 3, Float64}( @. (1-Φ ) * ρs  + Φ  * ρf  )
        ρt0    = SMatrix{3, 3, Float64}( @. (1-Φ0 )* ρs0 + Φ0 * ρf0 )
        
        ∂ρt∂t  = (ρt[2,2] - ρt0[2,2]) / Δt
        ρfx    = SVector{2, Float64}( @. (ρf[1:end-1,2] + ρf[2:end,2])/2 )
        ρfy    = SVector{2, Float64}( @. (ρf[2,1:end-1] + ρf[2,2:end])/2 )
        ρtx    = SVector{2, Float64}( @. (ρt[1:end-1,2] + ρt[2:end,2])/2 )
        ρty    = SVector{2, Float64}( @. (ρt[2,1:end-1] + ρt[2,2:end])/2 )
        qρx    = SVector{2, Float64}( @. ρfx * qx +  ρtx * Vx[:,2] )     # Brucite paper, Fowler (1985)
        qρy    = SVector{2, Float64}( @. ρfy * qy +  ρty * Vy[2,:] )     # Brucite paper, Fowler (1985)    
        
        fp     = ∂ρt∂t  +  (qρx[2] - qρx[1]) * invΔx + (qρy[2] - qρy[1]) * invΔy 
    end

    return fp
end

function ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
        Pt_loc     = SMatrix{2,3}(      P.t[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        Pf_loc     = SMatrix{2,3}(      P.f[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        ΔPt_loc    = SMatrix{2,1}(     ΔP.t[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        ΔPf_loc    = SMatrix{2,1}(     ΔP.t[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-1:i+1, jj in j-1:j  )

        Dc         = SMatrix{2,1}(      𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        Dv         = SMatrix{1,2}(      𝐷.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)
        D          = (c=Dc, v=Dv)
        τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

        if type.Vx[i,j] == :in
            R.x[i,j]   = SMomentum_x_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPt_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, P0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,3)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
    ∂R∂Pf = @MMatrix zeros(2,3)

    Vx_loc  = @MMatrix zeros(3,3)
    Vy_loc  = @MMatrix zeros(4,4)
    Pt_loc  = @MMatrix zeros(2,3)
    Pf_loc  = @MMatrix zeros(2,3)
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc    .= SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc    .= SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0) 
        
        Pt_loc    .= SMatrix{2,3}(      P.t[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        Pf_loc    .= SMatrix{2,3}(      P.f[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        ΔPt_loc    = SMatrix{2,1}(     ΔP.t[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-1:i+1, jj in j-1:j  )
      
        Dc         = SMatrix{2,1}(      𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        Dv         = SMatrix{1,2}(      𝐷.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)
        D          = (c=Dc, v=Dv)
        τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

        if type.Vx[i,j] == :in
     
            fill!(∂R∂Vx, 0.0)
            fill!(∂R∂Vy, 0.0)
            fill!(∂R∂Pt, 0.0)
            fill!(∂R∂Pf, 0.0)

            autodiff(Enzyme.Reverse, SMomentum_x_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt_loc, ∂R∂Pt), Duplicated(Pf_loc, ∂R∂Pf), Const(ΔPt_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
            # Vx --- Vx
            Local = num.Vx[i-1:i+1,j-1:j+1] .* pattern[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][1][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vx --- Vy
            Local = num.Vy[i-1:i+2,j-2:j+1] .* pattern[1][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][2][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vx --- Pt
            Local = num.Pt[i-1:i,j-2:j] .* pattern[1][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][3][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
            # Vx --- Pf
            Local = num.Pf[i-1:i,j-2:j] .* pattern[1][4]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][4][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
                end
            end 
        end
    end
    return nothing
end

function ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = SMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = SMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-1:i-0, jj in j-0:j-0) 
        Pt_loc     = SMatrix{3,2}(      P.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf_loc     = SMatrix{3,2}(      P.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        ΔPt_loc    = SMatrix{1,2}(     ΔP.t[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        ΔPf_loc    = SMatrix{1,2}(     ΔP.f[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        Pt0_loc    = SMatrix{3,2}(     P0.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf0_loc    = SMatrix{3,2}(     P0.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Φ0_loc     = SMatrix{1,2}(     Φ0.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        τxx0       = SMatrix{3,2}(    τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        τyy0       = SMatrix{3,2}(    τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        τxy0       = SMatrix{2,3}(    τ0.xy[ii,jj] for ii in i-1:i,   jj in j-1:j+1)
        Dc         = SMatrix{1,2}(      𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
        Dv         = SMatrix{2,1}(      𝐷.v[ii,jj] for ii in i-1:i-0,   jj in j-0:j-0)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)
        ΔP_loc     = (t=ΔPt_loc, f=ΔPf_loc)
        D          = (c=Dc, v=Dv)
        τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)
        if type.Vy[i,j] == :in
            R.y[i,j]   = SMomentum_y_Generic(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔP_loc, Pt0_loc, Pf0_loc, Φ0_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(3,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    ∂R∂Pf = @MMatrix zeros(3,2)

    Vx_loc  = @MMatrix zeros(4,4)
    Vy_loc  = @MMatrix zeros(3,3)
    Pt_loc  = @MMatrix zeros(3,2)
    Pf_loc  = @MMatrix zeros(3,2)

    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc    .= SMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc    .= SMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        phc_loc    = @inline SMatrix{1,2}(@inbounds  phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        phv_loc    = @inline SMatrix{2,1}(@inbounds  phases.v[ii,jj] for ii in i-1:i-0, jj in j-0:j-0) 
        Pt_loc    .= SMatrix{3,2}(      P.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf_loc    .= SMatrix{3,2}(      P.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        ΔPt_loc    = @inline SMatrix{1,2}(@inbounds     ΔP.t[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        ΔPf_loc    = SMatrix{1,2}(     ΔP.f[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        Pt0_loc    = SMatrix{3,2}(     P0.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf0_loc    = SMatrix{3,2}(     P0.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Φ0_loc     = SMatrix{1,2}(     Φ0.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        τxx0       = @inline SMatrix{3,2}(@inbounds     τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        τyy0       = @inline SMatrix{3,2}(@inbounds     τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        τxy0       = @inline SMatrix{2,3}(@inbounds     τ0.xy[ii,jj] for ii in i-1:i,   jj in j-1:j+1)
        Dc         = @inline SMatrix{1,2}(@inbounds       𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
        Dv         = @inline SMatrix{2,1}(@inbounds       𝐷.v[ii,jj] for ii in i-1:i-0,   jj in j-0:j-0)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)
        ΔP_loc     = (t=ΔPt_loc, f=ΔPf_loc)
        D          = (c=Dc, v=Dv)
        τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

        if type.Vy[i,j] == :in

            fill!(∂R∂Vx, 0.0)
            fill!(∂R∂Vy, 0.0)
            fill!(∂R∂Pt, 0.0)
            fill!(∂R∂Pf, 0.0)

            autodiff(Enzyme.Reverse, SMomentum_y_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt_loc, ∂R∂Pt), Duplicated(Pf_loc, ∂R∂Pf), Const(ΔP_loc), Const(Pt0_loc), Const(Pf0_loc), Const(Φ0_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))

            Local = num.Vx[i-2:i+1,j-1:j+2] .* pattern[2][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][1][num.Vy[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vy --- Vy
            Local = num.Vy[i-1:i+1,j-1:j+1] .* pattern[2][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][2][num.Vy[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vy --- Pt
            Local = num.Pt[i-2:i,j-1:j] .* pattern[2][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][3][num.Vy[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
            # Vy --- Pf
            Local = num.Pf[i-2:i,j-1:j] .* pattern[2][4]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][4][num.Vy[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
                end
            end       
        end
    end
    return nothing
end

function ResidualContinuity2D!(R, V, P, old, phases, materials, number, type, BC, nc, Δ) 
    
    P0, ϕ0, ρ0   = old
    shift    = (x=1, y=1)
    # (; bc_val, type, pattern, num) = numbering
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        ρs0        = SMatrix{3,3}(     ρ0.s[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρf0        = SMatrix{3,3}(     ρ0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf         = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt         = SMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
        bcx_loc    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        bcy_loc    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        typex_loc  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        typey_loc  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        old_loc    = (Pt = Pt0, Pf=Pf0, ϕ=Φ0, ρs=ρs0, ρf=ρf0 )
        R.pt[i,j]  = Continuity(Vx_loc, Vy_loc, Pt, Pf, old_loc, phases.c[i,j], materials, type_loc, bcv_loc, Δ)

    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, old, phases, materials, num, pattern, type, BC, nc, Δ) 
         
    P0, ϕ0, ρ0   = old
    shift    = (x=1, y=1)
    ∂R∂Vx = @MMatrix zeros(2,3)
    ∂R∂Vy = @MMatrix zeros(3,2)
    ∂R∂Pt = @MMatrix zeros(3,3)
    ∂R∂Pf = @MMatrix zeros(3,3)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        ρs0        = SMatrix{3,3}(     ρ0.s[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρf0        = SMatrix{3,3}(     ρ0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf         = MMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt         = MMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = MMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc     = MMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
        bcx_loc    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        bcy_loc    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        typex_loc  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) 
        typey_loc  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        old_loc    = (Pt = Pt0, Pf=Pf0, ϕ=Φ0, ρs=ρs0, ρf=ρf0 )

        ∂R∂Vx .= 0.
        ∂R∂Vy .= 0.
        ∂R∂Pt .= 0.
        ∂R∂Pf .= 0.

        autodiff(Enzyme.Reverse, Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt, ∂R∂Pt), Duplicated(Pf, ∂R∂Pf), Const(old_loc), Const(phases.c[i,j]), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))

        # Pt --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[3][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pt --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end
        # Pt --- Pt
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[3][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][3][num.Pt[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
            end
        end
        # Pt --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[3][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][4][num.Pt[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
            end
        end
    end
    return nothing
end


# function ResidualFluidContinuity2D!(R, V, P, ΔP, old, phases, materials, number, type, BC, nc, Δ) 
      
#     P0, ϕ0, ρ0 = old
#     shift    = (x=1, y=1)
#     for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
#         if type.Pf[i,j] !== :constant 
#             phase      = SMatrix{3,3}( phases.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#             Pt_loc     = SMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#             Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#             ΔPf_loc    = SMatrix{3,3}(     ΔP.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#             Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#             Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#             Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#             type_loc   = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#             bcv_loc    = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#             Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
#             Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
#             # k_loc_xx   = @SVector [rheo.kμf.x[i,j+1], rheo.kμf.x[i+1,j+1]]
#             # k_loc_yy   = @SVector [rheo.kμf.y[i+1,j], rheo.kμf.y[i+1,j+1]]
#             k_loc_xx   = @SVector [materials.k_ηf0[phases.x[i,j+1]], materials.k_ηf0[phases.x[i+1,j+1]]]
#             k_loc_yy   = @SVector [materials.k_ηf0[phases.y[i+1,j]], materials.k_ηf0[phases.y[i+1,j+1]]]
#             k_loc      = (xx = k_loc_xx,    xy = 0.,
#                           yx = 0.,          yy = k_loc_yy)
#             R.pf[i,j]  = FluidContinuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPf_loc, Pt0, Pf0, Φ0, phase, materials, k_loc, type_loc, bcv_loc, Δ)

#         end
#     end
#     return nothing
# end

# function AssembleFluidContinuity2D!(K, V, P, ΔP, old, phases, materials, num, pattern, type, BC, nc, Δ) 
         
#     P0, ϕ0, ρ0 = old
#     shift    = (x=1, y=1)
#     ∂R∂Vx = @MMatrix zeros(2,3)
#     ∂R∂Vy = @MMatrix zeros(3,2)
#     ∂R∂Pt = @MMatrix zeros(3,3)
#     ∂R∂Pf = @MMatrix zeros(3,3)

#     for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
#         phase      = SMatrix{3,3}( phases.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#         Pt_loc     = MMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#         Pf_loc     = MMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#         ΔPf_loc    = MMatrix{3,3}(     ΔP.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#         Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#         Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#         Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)        
#         type_loc   = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#         bcv_loc    = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
#         Vx_loc     = MMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
#         Vy_loc     = MMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
#         k_loc_xx   = @SVector [materials.k_ηf0[phases.x[i,j+1]], materials.k_ηf0[phases.x[i+1,j+1]]]
#         k_loc_yy   = @SVector [materials.k_ηf0[phases.y[i+1,j]], materials.k_ηf0[phases.y[i+1,j+1]]]
#         k_loc      = (xx = k_loc_xx,    xy = 0.,
#                       yx = 0.,          yy = k_loc_yy)

#         ∂R∂Vx .= 0.
#         ∂R∂Vy .= 0.
#         ∂R∂Pt .= 0.
#         ∂R∂Pf .= 0.
#         autodiff(Enzyme.Reverse, FluidContinuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt_loc, ∂R∂Pt), Duplicated(Pf_loc, ∂R∂Pf), Const(ΔPf_loc), Const(Pt0), Const(Pf0), Const(Φ0), Const(phase), Const(materials), Const(k_loc), Const(type_loc), Const(bcv_loc), Const(Δ))
             
#         # Pf --- Vx
#         Local = num.Vx[i:i+1,j:j+2] .* pattern[4][1]
#         for jj in axes(Local,2), ii in axes(Local,1)
#             if Local[ii,jj]>0 && num.Pf[i,j]>0
#                 K[4][1][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
#             end
#         end
#         # Pf --- Vy
#         Local = num.Vy[i:i+2,j:j+1] .* pattern[4][2]
#         for jj in axes(Local,2), ii in axes(Local,1)
#             if Local[ii,jj]>0 && num.Pf[i,j]>0
#                 K[4][2][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
#             end
#         end
#         # Pf --- Pt
#         Local = num.Pt[i-1:i+1,j-1:j+1] .* pattern[4][3]
#         for jj in axes(Local,2), ii in axes(Local,1)
#             if (Local[ii,jj]>0) && num.Pf[i,j]>0
#                 K[4][3][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
#             end
#         end
#         # Pf --- Pf
#         Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[4][4]
#         for jj in axes(Local,2), ii in axes(Local,1)
#             if (Local[ii,jj]>0) && num.Pf[i,j]>0
#                 K[4][4][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
#             end
#         end
           
#     end
#     return nothing
# end

function ResidualFluidContinuity2D!(R, V, P, ΔP, old, phases, materials, number, type, BC, nc, Δ) 
                
    P0, ϕ0, ρ0   = old
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        if type.Pf[i,j] !== :constant 
            phase      = SMatrix{3,3}( phases.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pt_loc     = SMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ΔPf_loc    = SMatrix{3,3}(     ΔP.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ρs0        = SMatrix{3,3}(     ρ0.s[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            ρf0        = SMatrix{3,3}(     ρ0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            type_loc   = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcv_loc    = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
            Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
            # k_loc_xx   = @SVector [rheo.kμf.x[i,j+1], rheo.kμf.x[i+1,j+1]]
            # k_loc_yy   = @SVector [rheo.kμf.y[i+1,j], rheo.kμf.y[i+1,j+1]]
            k_loc_xx   = @SVector [materials.k_ηf0[phases.x[i,j+1]], materials.k_ηf0[phases.x[i+1,j+1]]]
            k_loc_yy   = @SVector [materials.k_ηf0[phases.y[i+1,j]], materials.k_ηf0[phases.y[i+1,j+1]]]
            k_loc      = (xx = k_loc_xx,    xy = 0.,
                          yx = 0.,          yy = k_loc_yy)
            old_loc    = (Pt = Pt0, Pf=Pf0, ϕ=Φ0, ρs=ρs0, ρf=ρf0 )
            R.pf[i,j]  = FluidContinuity(Vx_loc, Vy_loc, Pt_loc, Pf_loc, ΔPf_loc, old_loc, phase, materials, k_loc, type_loc, bcv_loc, Δ)

        end
    end
    return nothing
end

function AssembleFluidContinuity2D!(K, V, P, ΔP, old, phases, materials, num, pattern, type, BC, nc, Δ) 
              
    P0, ϕ0, ρ0 = old
    shift    = (x=1, y=1)
    ∂R∂Vx = @MMatrix zeros(2,3)
    ∂R∂Vy = @MMatrix zeros(3,2)
    ∂R∂Pt = @MMatrix zeros(3,3)
    ∂R∂Pf = @MMatrix zeros(3,3)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        phase      = SMatrix{3,3}( phases.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt_loc     = MMatrix{3,3}(      P.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf_loc     = MMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ΔPf_loc    = MMatrix{3,3}(     ΔP.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt0        = SMatrix{3,3}(     P0.t[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pf0        = SMatrix{3,3}(     P0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Φ0         = SMatrix{3,3}(     ϕ0.c[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) 
        ρs0        = SMatrix{3,3}(     ρ0.s[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ρf0        = SMatrix{3,3}(     ρ0.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)       
        type_loc   = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc    = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = MMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc     = MMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
        k_loc_xx   = @SVector [materials.k_ηf0[phases.x[i,j+1]], materials.k_ηf0[phases.x[i+1,j+1]]]
        k_loc_yy   = @SVector [materials.k_ηf0[phases.y[i+1,j]], materials.k_ηf0[phases.y[i+1,j+1]]]
        k_loc      = (xx = k_loc_xx,    xy = 0.,
                      yx = 0.,          yy = k_loc_yy)
        old_loc    = (Pt = Pt0, Pf=Pf0, ϕ=Φ0, ρs=ρs0, ρf=ρf0 )

        ∂R∂Vx .= 0.
        ∂R∂Vy .= 0.
        ∂R∂Pt .= 0.
        ∂R∂Pf .= 0.
        autodiff(Enzyme.Reverse, FluidContinuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt_loc, ∂R∂Pt), Duplicated(Pf_loc, ∂R∂Pf), Const(ΔPf_loc), Const(old_loc), Const(phase), Const(materials), Const(k_loc), Const(type_loc), Const(bcv_loc), Const(Δ))
             
        # Pf --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[4][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pf[i,j]>0
                K[4][1][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pf --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[4][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pf[i,j]>0
                K[4][2][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end
        # Pf --- Pt
        Local = num.Pt[i-1:i+1,j-1:j+1] .* pattern[4][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][3][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
            end
        end
        # Pf --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[4][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][4][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
            end
        end
           
    end
    return nothing
end


function UpdatePorosity2D!(R, V, P, P0, Φ, Φ0, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        if type.Pf[i,j] !== :constant 
            KΦ        = materials.KΦ[phases.c[i,j]]
            ηΦ        = materials.ηΦ[phases.c[i,j]]
            dPtdt     = (P.t[i,j] - P0.t[i,j]) / Δ.t
            dPfdt     = (P.f[i,j] - P0.f[i,j]) / Δ.t
            dΦdt      = (dPfdt - dPtdt)/KΦ + (P.f[i,j] - P.t[i,j])/ηΦ
            Φ.c[i,j]  = Φ0.c[i,j] + dΦdt*Δ.t
        end
    end
    return nothing
end

function ResidualPorosity2D!(R, V, P, P0, Φ, Φ0, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        if type.Pf[i,j] !== :constant 
            KΦ        = materials.KΦ[phases.c[i,j]]
            ηΦ        = materials.ηΦ[phases.c[i,j]]
            dPtdt     = (P.t[i,j] - P0.t[i,j]) / Δ.t
            dPfdt     = (P.f[i,j] - P0.f[i,j]) / Δ.t
            dΦdt      = (dPfdt - dPtdt)/KΦ + (P.f[i,j] - P.t[i,j])/ηΦ
            R.Φ[i,j]  = Φ.c[i,j] - (Φ0.c[i,j] + dΦdt*Δ.t)
        end
    end
    return nothing
end

function SetBCPf1(Pf, type, bc, Δ, ρfg)

    MPf =  MMatrix(Pf)

    # N/S
    for ii in axes(type, 1)
        # South
        if type[ii,1] === :Dirichlet
            MPf[ii,1] = fma(2, bc[ii,1], -Pf[ii,2])
        elseif type[ii,1] === :Neumann 
            MPf[ii,1] = fma(Δ.y, bc[ii,1], Pf[ii,2])
        elseif type[ii,1] === :no_flux
            MPf[ii,1] = Pf[ii,2] - ρfg*Δ.y
        elseif type[ii,1] === :periodic || type[ii,1] === :in || type[ii,1] === :constant
            MPf[ii,1] = Pf[ii,1]
        # else
        #     MPf[ii,1] = 1.0
        end

        # North
        if type[ii,end] === :Dirichlet
            MPf[ii,end] = fma(2, bc[ii,end], -Pf[ii,end-1])
        elseif type[ii,end] === :Neumann
            MPf[ii,end] = fma(-Δ.y, bc[ii,end], Pf[ii,end-1])
        elseif type[ii,end] === :no_flux
            MPf[ii,end] = Pf[ii,end-1] + ρfg*Δ.y
        elseif type[ii,end] === :periodic || type[ii,end] === :in || type[ii,end] === :constant
            MPf[ii,end] = Pf[ii,end]
        # else
        #     MPf[ii,end] = 1.0
        end
    end


    # E/W
    for jj in axes(type, 2)
        # West
        if type[1,jj] === :Dirichlet
            MPf[1,jj] = fma(2, bc[1,jj], - Pf[2,jj])
        elseif type[1,jj] === :Neumann
            MPf[1,jj] = fma(Δ.x, bc[1,jj], Pf[2,jj])
        elseif type[1,jj] === :periodic || type[1,jj] === :in || type[1,jj] === :constant
            MPf[1,jj] = Pf[1,jj] 
        # else
        #     MPf[1,jj] =  1.0
        end

        # East
        if type[end,jj] === :Dirichlet
            MPf[end,jj] = fma(2, bc[end,jj], - Pf[end-1,jj])
        elseif type[end,jj] === :Neumann
            MPf[end,jj] = fma(-Δ.x, bc[end,jj], Pf[end-1,jj])
        elseif type[end,jj] === :periodic || type[end,jj] === :in || type[end,jj] === :constant
            MPf[end,jj] = Pf[end,jj] 
        # else
        #     MPf[end,jj] =  1.0
        end
    end

    return SMatrix(MPf)
end

function SetBCVx1(Vx, typex, bcx, Δ)

    MVx = MMatrix(Vx)
    # N/S
    for ii in axes(typex, 1)
        if typex[ii,1] == :Dirichlet_tangent
            MVx[ii,1] = fma(2, bcx[ii,1], -Vx[ii,2])
        elseif typex[ii,1] == :Neumann_tangent
            MVx[ii,1] = fma(Δ.y, bcx[ii,1], Vx[ii,2])
        end

        if typex[ii,end] == :Dirichlet_tangent
            MVx[ii,end] = fma(2, bcx[ii,end], -Vx[ii,end-1])
        elseif typex[ii,end] == :Neumann_tangent
            MVx[ii,end] = fma(Δ.y, bcx[ii,end], Vx[ii,end-1])
        end
    end
    # E/W
    for jj in axes(typex, 2)
        if typex[1,jj] == :Neumann_normal
            MVx[1,jj] = fma(2, Δ.x*bcx[1,jj], Vx[2,jj])
        end
        if typex[end,jj] == :Neumann_normal
            MVx[end,jj] = fma(2,-Δ.x*bcx[end,jj], Vx[end-1,jj])
        end
    end
    return SMatrix(MVx)
end

function SetBCVy1(Vy, typey, bcy, Δ)
    MVy = MMatrix(Vy)
    # E/W
    for jj in axes(typey, 2)
        if typey[1,jj] == :Dirichlet_tangent
            MVy[1,jj] = fma(2, bcy[1,jj], -Vy[2,jj])
        elseif typey[1,jj] == :Neumann_tangent
            MVy[1,jj] = fma(Δ.y, bcy[1,jj], Vy[2,jj])
        end

        if typey[end,jj] == :Dirichlet_tangent
            MVy[end,jj] = fma(2, bcy[end,jj], -Vy[end-1,jj])
        elseif typey[end,jj] == :Neumann_tangent
            MVy[end,jj] = fma(Δ.y, bcy[end,jj], Vy[end-1,jj])
        end
    end
    # N/S
    for ii in axes(typey, 1)
        if typey[ii,1] == :Neumann_normal
            MVy[ii,1] = fma(2, Δ.y*bcy[ii,1], Vy[ii,2])
        end
        if typey[ii,end] == :Neumann_normal
            MVy[ii,end] = fma(2,-Δ.y*bcy[ii,end], Vy[ii,end-1])
        end
    end
    return SMatrix(MVy)
end

function Numbering!(N, type, nc)
    
    ndof  = 0
    neq   = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west  = sum(any(i->i==:periodic, type.Vx[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vx[:,2], dims=1)) > 0

    shift  = (periodic_west) ? 1 : 0 
    # Loop through inner nodes of the mesh
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vx[i,j] == :Dirichlet_normal || (type.Vx[i,j] != :periodic && i==nc.x+3-1) || type.Vx[i,j] == :constant 
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vx[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx[1,:] .= N.Vx[end-2,:]
    end

    # Copy equation indices for periodic cases
    if periodic_south
        # South
        N.Vx[:,1] .= N.Vx[:,end-3]
        N.Vx[:,2] .= N.Vx[:,end-2]
        # North
        N.Vx[:,end]   .= N.Vx[:,4]
        N.Vx[:,end-1] .= N.Vx[:,3]
    end
    noisy ? printxy(N.Vx) : nothing

    neq = maximum(N.Vx)

    ############ Numbering Vy ############
    ndof  = 0
    periodic_west  = sum(any(i->i==:periodic, type.Vy[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vy[:,2], dims=1)) > 0
    shift = periodic_south ? 1 : 0
    # Loop through inner nodes of the mesh
    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] != :periodic && j==nc.y+3-1) || type.Vy[i,j] == :constant 
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vy[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_south
        N.Vy[:,1]     .= N.Vy[:,end-2]
        N.Vy[:,end-1] .= N.Vy[:,2]
        N.Vy[:,end]   .= N.Vy[:,3]
    end

    # Copy equation indices for periodic cases
    if periodic_west
        # West
        N.Vy[1,:] .= N.Vy[end-3,:]
        N.Vy[2,:] .= N.Vy[end-2,:]
        # East
        N.Vy[end,:]   .= N.Vy[4,:]
        N.Vy[end-1,:] .= N.Vy[3,:]
    end
    noisy ? printxy(N.Vy) : nothing

    neq = maximum(N.Vy)

    ############ Numbering Pt ############
    # neq_Pt                     = nc.x * nc.y
    # N.Pt[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ 0*neq, nc.x, nc.y)
    ii = 0
    for j=1:nc.y, i=1:nc.x
        if type.Pt[i+1,j+1] != :constant
            ii += 1
            N.Pt[i+1,j+1] = ii
        end
    end

    if periodic_west
        N.Pt[1,:]   .= N.Pt[end-1,:]
        N.Pt[end,:] .= N.Pt[2,:]
    end

    if periodic_south
        N.Pt[:,1]   .= N.Pt[:,end-1]
        N.Pt[:,end] .= N.Pt[:,2]
    end
    noisy ? printxy(N.Pt) : nothing

    neq = maximum(N.Pt)

    ############ Numbering Pf ############

    # neq_Pf                    = nc.x * nc.y
    # N.Pf[2:end-1,2:end-1] .= reshape(1:neq_Pf, nc.x, nc.y)
    ii = 0
    for j=1:nc.y, i=1:nc.x
        if type.Pf[i+1,j+1] != :constant
            ii += 1
            N.Pf[i+1,j+1] = ii
        end
    end

    # Make periodic in x
    for j in axes(type.Pf,2)
        if type.Pf[1,j] === :periodic
            N.Pf[1,j] = N.Pf[end-1,j]
        end
        if type.Pf[end,j] === :periodic
            N.Pf[end,j] = N.Pf[2,j]
        end
    end

    # Make periodic in y
    for i in axes(type.Pf,1)
        if type.Pf[i,1] === :periodic
            N.Pf[i,1] = N.Pf[i,end-1]
        end
        if type.Pf[i,end] === :periodic
            N.Pf[i,end] = N.Pf[i,2]
        end
    end

end

function SetRHS!(r, R, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            r[ind] = R.x[i,j]
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            r[ind] = R.y[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            r[ind] = R.pt[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            r[ind] = R.pf[i,j]
        end
    end
end

function UpdateSolution!(V, P, dx, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            V.x[i,j] += dx[ind] 
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            V.y[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            P.t[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            P.f[i,j] += dx[ind]
        end
    end
end

@views function SparsityPattern!(K, num, pattern, nc) 
    ############ Fields Vx ############
    shift  = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vx --- Vx
        Local = num.Vx[i-1:i+1,j-1:j+1] .* pattern[1][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][1][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Vy
        Local = num.Vy[i-1:i+2,j-2:j+1] .* pattern[1][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][2][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Pt
        Local = num.Pt[i-1:i,j-2:j] .* pattern[1][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][3][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Pf
        Local = num.Pf[i-1:i,j-2:j] .* pattern[1][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][4][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Vy ############
    shift  = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vy --- Vx
        Local = num.Vx[i-2:i+1,j-1:j+2] .* pattern[2][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][1][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Vy
        Local = num.Vy[i-1:i+1,j-1:j+1] .* pattern[2][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][2][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Pt
        Local = num.Pt[i-2:i,j-1:j] .* pattern[2][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][3][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Pf
        Local = num.Pf[i-2:i,j-1:j] .* pattern[2][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][4][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Pt ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pt --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[3][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Pt
        Local = num.Pt[i,j] .* pattern[3][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][3][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Pf
        Local = num.Pf[i,j] .* pattern[3][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][4][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Pf ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pf --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[4][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][1][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[4][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][2][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Pt
        Local = num.Pt[i,j] .* pattern[4][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][3][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[4][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][4][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ End ############
end

function LineSearch!(rvec, α, dx, R, V, P, ε̇, τ, Vi, Pi, ΔP, Φ, old, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    
    τ0, P0, Φ0, ρ0 = old
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x 
    Vi.y .= V.y 
    Pi.t .= P.t
    Pi.f .= P.f

    for i in eachindex(α)
        V.x .= Vi.x 
        V.y .= Vi.y
        P.t .= Pi.t
        P.f .= Pi.f
        UpdateSolution!(V, P, α[i].*dx, number, type, nc)
        TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)
        ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualContinuity2D!(R, V, P, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 
        ResidualFluidContinuity2D!(R, V, P, ΔP, (P0, Φ0, ρ0), phases, materials, number, type, BC, nc, Δ) 
        rvec[i] = @views norm(R.x[inx_Vx,iny_Vx])/length(R.x[inx_Vx,iny_Vx]) + norm(R.y[inx_Vy,iny_Vy])/length(R.y[inx_Vy,iny_Vy]) + norm(R.pt[inx_c,iny_c])/length(R.pt[inx_c,iny_c]) + norm(R.pf[inx_c,iny_c])/length(R.pf[inx_c,iny_c])  
    end
    imin = argmin(rvec)
    V.x .= Vi.x 
    V.y .= Vi.y
    P.t .= Pi.t
    P.f .= Pi.f
    return imin
end

function GlobalResidual!(α, dx, R, V, P, ε̇, τ, ΔP, P0, Φ, Φ0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    UpdateSolution!(V, P, α.*dx, number, type, nc)
    TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)
    ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
    ResidualContinuity2D!(R, V, P, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 
    ResidualFluidContinuity2D!(R, V, P, ΔP, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 
end

@inline fnorm(R, inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c) = @views (norm(R.x[inx_Vx,iny_Vx])/sqrt(length(R.x[inx_Vx,iny_Vx])))^2 + (norm(R.y[inx_Vy,iny_Vy])/sqrt(length(R.y[inx_Vy,iny_Vy])))^2 + 1*(norm(R.pt[inx_c,iny_c])/length(R.pt[inx_c,iny_c]))^2 + 1*(norm(R.pf[inx_c,iny_c])/length(R.pf[inx_c,iny_c]))^2

function BackTrackingLineSearch!(rvec, α, dx, R0, R, V, P, ε̇, τ, Vi, Pi, ΔP, P0, Φ, Φ0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ; α_init=1.0, β=0.5, c=1e-4)
    
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x 
    Vi.y .= V.y 
    Pi.t .= P.t
    Pi.f .= P.f

    α = α_init
    GlobalResidual!(0.0, dx, R0, V, P, ε̇, τ, ΔP, P0, Φ, Φ0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    
    f0_norm_sq = fnorm(R, inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c) 

    k = 0
    max_iters = 5

    for iter in 1:max_iters
    # # while f_norm_sq >= (1 - c * α * slope) * f0_norm_sq

        k    += 1

        V.x .= Vi.x 
        V.y .= Vi.y
        P.t .= Pi.t
        P.f .= Pi.f

        GlobalResidual!(  α, dx, R, V, P, ε̇, τ, ΔP, P0, Φ, Φ0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
        
        f_norm_sq = fnorm(R, inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c) 

        slope = -2 * ( sum(R0.x[inx_Vx,iny_Vx].*R.x[inx_Vx,iny_Vx]) + sum(R0.y[inx_Vy,iny_Vy].*R.y[inx_Vy,iny_Vy]) + 1*sum(R0.pt[inx_c,iny_c].*R.pt[inx_c,iny_c]) + 1*sum(R0.pf[inx_c,iny_c].*R.pf[inx_c,iny_c]) )
    
         if f_norm_sq <= (1 - c * α * slope) * f0_norm_sq
            break        
        end

        # @show α, f_norm_sq, f0_norm_sq, (1 - c * α * slope) * f0_norm_sq


        @show α, f_norm_sq, f0_norm_sq, f_norm_sq/f0_norm_sq

        α *= β

    end

    V.x .= Vi.x 
    V.y .= Vi.y
    P.t .= Pi.t
    P.f .= Pi.f

    @info k, α

    return α
end

    
# function backtracking_line_search(f, x, δx; α_init=1.0, β=0.5, c=1e-4)
#     α = α_init
#     fx = f(x)
#     f_norm_sq = norm(fx)^2
#     slope = -2 * real(dot(fx, f(x + α * δx)))  # approximation to directional derivative

    # while norm(f(x + α * δx))^2 > f_norm_sq - c * α * slope
    #     α *= β
    # end

#     return α
# end