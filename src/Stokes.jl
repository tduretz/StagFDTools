struct Fields{Tx,Ty,Tp}
    Vx::Tx
    Vy::Ty
    Pt::Tp
end

function Base.getindex(x::Fields, i::Int64)
    @assert 0 < i < 4 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
end

function Ranges(nc)     
    return (inx_Vx = 2:nc.x+2, iny_Vx = 3:nc.y+2, inx_Vy = 3:nc.x+2, iny_Vy = 2:nc.y+2, inx_c = 2:nc.x+1, iny_c = 2:nc.y+1, inx_v = 2:nc.x+2, iny_v = 2:nc.y+2, size_x = (nc.x+3, nc.y+4), size_y = (nc.x+4, nc.y+3), size_c = (nc.x+2, nc.y+2), size_v = (nc.x+3, nc.y+3))
end

function set_boundaries_template!(type, config, nc)
    
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    @info "Setting $(string(config))"

    if config == :all_Dirichlet
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in       
        type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
        type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
        type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
        type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in       
        type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
        type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
        type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
        type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in

    elseif config == :EW_periodic # East/West periodic
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]    .= :in       
        type.Vx[1,iny_Vx]         .= :periodic 
        type.Vx[end-1:end,iny_Vx] .= :periodic 
        type.Vx[inx_Vx,2]         .= :Dirichlet_tangent
        type.Vx[inx_Vx,end-1]     .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]    .= :in       
        type.Vy[1:2,iny_Vy]       .= :periodic
        type.Vy[end-1:end,iny_Vy] .= :periodic
        type.Vy[inx_Vy,2]         .= :Dirichlet_normal 
        type.Vy[inx_Vy,end-1]     .= :Dirichlet_normal 
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
        type.Pt[[1 end],2:end-1] .= :periodic

    elseif config == :NS_periodic  # North/South periodic
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]    .= :in       
        type.Vx[2,iny_Vx]         .= :Dirichlet_normal
        type.Vx[end-1,iny_Vx]     .= :Dirichlet_normal
        type.Vx[inx_Vx,1:2]       .= :periodic 
        type.Vx[inx_Vx,end-1:end] .= :periodic 
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]    .= :in       
        type.Vy[2,iny_Vy]         .= :Dirichlet_tangent 
        type.Vy[end-1,iny_Vy]     .= :Dirichlet_tangent 
        type.Vy[inx_Vy,1]         .= :periodic
        type.Vy[inx_Vy,end-1:end] .= :periodic
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
        type.Pt[2:end-1,[1 end]] .= :periodic

    elseif config == :NS_Neumann
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in       
        type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
        type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
        type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
        type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in       
        type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
        type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
        type.Vy[inx_Vy,1]       .= :Neumann_normal
        type.Vy[inx_Vy,end]     .= :Neumann_normal
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
    elseif config == :N_StressFree
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in       
        type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
        type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
        type.Vx[inx_Vx,2]       .= :Neumann_tangent
        type.Vx[inx_Vx,end-1]   .= :Neumann_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in       
        type.Vy[2,iny_Vy]       .= :Neumann_tangent
        type.Vy[end-1,iny_Vy]   .= :Neumann_tangent
        type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
        type.Vy[inx_Vy,end]     .= :Neumann_normal
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in

    elseif config == :EW_Neumann
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in      
        type.Vx[1,iny_Vx]       .= :Neumann_normal
        type.Vx[end,iny_Vx]     .= :Neumann_normal
        type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
        type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in       
        type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
        type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
        type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
        type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
        # type.Pt[[1,end],2:end-1] .= :Neumann_normal

    elseif config == :free_slip
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in       
        type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
        type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
        type.Vx[inx_Vx,2]       .= :Neumann_tangent
        type.Vx[inx_Vx,end-1]   .= :Neumann_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in       
        type.Vy[2,iny_Vy]       .= :Neumann_tangent
        type.Vy[end-1,iny_Vy]   .= :Neumann_tangent
        type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
        type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
        
    elseif config == :no_slip
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in       
        type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
        type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
        type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
        type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
        # -------- Vy -------- #
        type.Vy[inx_Vy,iny_Vy]  .= :in       
        type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
        type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
        type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
        type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
        # -------- Pt -------- #
        type.Pt[2:end-1,2:end-1] .= :in
        
    end
end

function SMomentum_x_Generic(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
    invΔx, invΔy = 1 / Δ.x, 1 / Δ.y

    # BC
    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)

    # @show type.p

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
    Ptc  = SVector{2}( @. Pt[:,2] + comp * ΔP[:] )

    # Stress
    Tstress = promote_type(eltype(Vx_loc), eltype(Vy_loc), eltype(Pt), eltype(Ptc))
    σxx = MVector{2, Tstress}(undef)
    τxy = MVector{2, Tstress}(undef)
    for i=1:2
        σxx[i] = (𝐷.c[i][1,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][1,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][1,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][1,4] - (𝐷.c[i][4,4] - 1)) * Pt[i,2]  - Ptc[i]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                       * P̄t[i]
    end
    # if type.p[1] == :Neumann_normal
    #     σxx[1] = 2*(200) - σxx[2]
    #     τxy[:] = 0.0
    # end
    # if type.p[2] == :Neumann_normal
    #     σxx[2] = 2*(200) - σxx[1]
    #     τxy[:] = 0.0
    # end

    # Residual
    fx  = ( σxx[2]  - σxx[1] ) * invΔx
    fx += ( τxy[2]  - τxy[1] ) * invΔy
    fx *= -1* Δ.x * Δ.y

    return fx
end

function SMomentum_y_Generic(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
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
    Ptc  = SVector{2}( @. Pt[2,:] + comp * ΔP[:] )

    # Stress
    Tstress = promote_type(eltype(Vx_loc), eltype(Vy_loc), eltype(Pt), eltype(Ptc))
    τyy = MVector{2, Tstress}(undef)
    τxy = MVector{2, Tstress}(undef)
    for i=1:2
        τyy[i] = (𝐷.c[i][2,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][2,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][2,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][2,4] - (𝐷.c[i][4,4] - 1.)) * Pt[2,i]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                        * P̄t[i]
    end

    # Gravity
    ρ    = SVector{2, Float64}( materials.ρ[phases.c])
    ρg   = materials.g[2] * 0.5*(ρ[1] + ρ[2])

    # Residual
    fy  = ( τyy[2]  -  τyy[1] ) * invΔy
    fy += ( τxy[2]  -  τxy[1] ) * invΔx
    fy -= ( Ptc[2]  -  Ptc[1])  * invΔy
    fy += ρg
    fy *= -1 * Δ.x * Δ.y
    
    return fy
end

function Continuity(Vx, Vy, Pt, Pt0, D, phase, materials, type_loc, bcv_loc, Δ)
    invΔx = 1 / Δ.x
    invΔy = 1 / Δ.y
    invΔt = 1 / Δ.t
    β     = materials.β[phase]
    η     = materials.β[phase]
    comp  = materials.compressible
    f     = ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy) + comp * β * (Pt[1] - Pt0) * invΔt #+ 1/(1000*η)*Pt[1]
    # f    *= max(invΔx, invΔy)
    return f
end

function ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        if type.Vx[i,j] == :in
            Vx_loc     = SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc     = SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typep_loc  = SMatrix{2,1}(  type.Pt[ii,jj] for ii in i-1:i-0, jj in j-1:j-1  )
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
            P_loc      = SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            ΔP_loc     = SMatrix{2,1}(       ΔP.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-1:i+1, jj in j-1:j  )

            Dc         = SMatrix{2,1}(      𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            Dv         = SMatrix{1,2}(      𝐷.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc, p=typep_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)
    
            R.x[i,j]   = SMomentum_x_Generic(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, P0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,3)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
                
    Vx_loc = @MMatrix zeros(3,3)
    Vy_loc = @MMatrix zeros(4,4)
    P_loc  = @MMatrix zeros(2,3)
    ΔP_loc = @MMatrix zeros(2,1)

    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        
        if type.Vx[i,j] == :in

            bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typep_loc  = SMatrix{2,1}(  type.Pt[ii,jj] for ii in i-1:i-0, jj in j-1:j-1  )
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0) 
            
            Vx_loc    .= SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc    .= SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            P_loc     .= SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            ΔP_loc    .= SMatrix{2,1}(       ΔP.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)

            τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-1:i+1, jj in j-1:j  )
            
            Dc         = SMatrix{2,1}(      𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            Dv         = SMatrix{1,2}(      𝐷.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc, p=typep_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

            fill!(∂R∂Vx, 0e0)
            fill!(∂R∂Vy, 0e0)
            fill!(∂R∂Pt, 0e0)
            ∂Vx, ∂Vy, ∂Pt = ad_partial_gradients(SMomentum_x_Generic, (Vx_loc, Vy_loc, P_loc), ΔP_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
            ∂R∂Vx .= ∂Vx
            ∂R∂Vy .= ∂Vy
            ∂R∂Pt .= ∂Pt
            # Vx --- Vx
            Local = SMatrix{3,3}(num.Vx[ii, jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][1][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vx --- Vy
            Local = SMatrix{4,4}(num.Vy[ii, jj] for ii in i-1:i+2, jj in j-2:j+1) .* pattern[1][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][2][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vx --- Pt
            Local = SMatrix{2,3}(num.Pt[ii, jj] for ii in i-1:i, jj in j-2:j) .* pattern[1][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][3][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
        end
    end
    return nothing
end

function ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        if type.Vy[i,j] == :in
            Vx_loc     = SMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            Vy_loc     = SMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-1:i-0, jj in j-0:j-0) 
            P_loc      = SMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            ΔP_loc     = SMatrix{1,2}(       ΔP.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            τxx0       = SMatrix{3,2}(    τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τyy0       = SMatrix{3,2}(    τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τxy0       = SMatrix{2,3}(    τ0.xy[ii,jj] for ii in i-1:i,   jj in j-1:j+1)
            Dc         = SMatrix{1,2}(      𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
            Dv         = SMatrix{2,1}(      𝐷.v[ii,jj] for ii in i-1:i-0,   jj in j-0:j-0)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

            R.y[i,j]   = SMomentum_y_Generic(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, P0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(3,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    
    Vx_loc = @MMatrix zeros(4,4)
    Vy_loc = @MMatrix zeros(3,3)
    P_loc  = @MMatrix zeros(3,2)
    ΔP_loc = @MMatrix zeros(1,2)
       
    shift    = (x=2, y=1)
    K21 = K[2][1]
    K22 = K[2][2]
    K23 = K[2][3]

    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x

        if type.Vy[i,j] === :in

            Vx_loc    .= @inline SMatrix{4,4}(@inbounds       V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            Vy_loc    .= @inline SMatrix{3,3}(@inbounds       V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcx_loc    = @inline SMatrix{4,4}(@inbounds     BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            bcy_loc    = @inline SMatrix{3,3}(@inbounds     BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typex_loc  = @inline SMatrix{4,4}(@inbounds   type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            typey_loc  = @inline SMatrix{3,3}(@inbounds   type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            phc_loc    = @inline SMatrix{1,2}(@inbounds  phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            phv_loc    = @inline SMatrix{2,1}(@inbounds  phases.v[ii,jj] for ii in i-1:i-0, jj in j-0:j-0) 
            P_loc     .= @inline SMatrix{3,2}(@inbounds         P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            ΔP_loc    .= @inline SMatrix{1,2}(@inbounds        ΔP.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            τxx0       = @inline SMatrix{3,2}(@inbounds     τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τyy0       = @inline SMatrix{3,2}(@inbounds     τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τxy0       = @inline SMatrix{2,3}(@inbounds     τ0.xy[ii,jj] for ii in i-1:i,   jj in j-1:j+1)
            Dc         = @inline SMatrix{1,2}(@inbounds       𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
            Dv         = @inline SMatrix{2,1}(@inbounds       𝐷.v[ii,jj] for ii in i-1:i-0,   jj in j-0:j-0)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

            fill!(∂R∂Vx, 0.0)
            fill!(∂R∂Vy, 0.0)
            fill!(∂R∂Pt, 0.0)
            ∂Vx, ∂Vy, ∂Pt = ad_partial_gradients(SMomentum_y_Generic, (Vx_loc, Vy_loc, P_loc), ΔP_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
            ∂R∂Vx .= ∂Vx
            ∂R∂Vy .= ∂Vy
            ∂R∂Pt .= ∂Pt
            
            num_Vy = @inbounds num.Vy[i,j]
            bounds_Vy = num_Vy > 0
            # Vy --- Vx
            Local1 = SMatrix{4,4}(num.Vx[ii, jj] for ii in i-2:i+1, jj in j-1:j+2) .* pattern[2][1]
            # for jj in axes(Local1,2), ii in axes(Local1,1)
            #     if (Local1[ii,jj]>0) && bounds_Vy
            #         @inbounds K21[num_Vy, Local1[ii,jj]] = ∂R∂Vx[ii,jj] 
            #     end
            # end
            # Vy --- Vy
            Local2 = SMatrix{3,3}(num.Vy[ii, jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern[2][2]
            # for jj in axes(Local2,2), ii in axes(Local2,1)
            #     if (Local2[ii,jj]>0) && bounds_Vy
            #         @inbounds K22[num_Vy, Local2[ii,jj]] = ∂R∂Vy[ii,jj]  
            #     end
            # end
            # Vy --- Pt
            Local3 = SMatrix{3,2}(num.Pt[ii, jj] for ii in i-2:i, jj in j-1:j) .* pattern[2][3]
            # for jj in axes(Local3,2), ii in axes(Local3,1)
            #     if (Local3[ii,jj]>0) && bounds_Vy
            #         @inbounds K23[num_Vy, Local3[ii,jj]] = ∂R∂Pt[ii,jj]  
            #     end
            # end 

            Base.@nexprs 4 jj -> begin
                Base.@nexprs 4 ii -> begin
                    bounds_Vy && (Local1[ii,jj]>0) && 
                        (@inbounds K21[num_Vy, Local1[ii,jj]] = ∂R∂Vx[ii,jj])
                    
                    bounds_Vy && ii<4 && jj<4 && (Local2[ii,jj]>0) &&
                        (@inbounds K22[num_Vy, Local2[ii,jj]] = ∂R∂Vy[ii,jj])

                    bounds_Vy && ii<4 && jj<3 && (Local3[ii,jj]>0) && 
                        (@inbounds K23[num_Vy, Local3[ii,jj]] = ∂R∂Pt[ii,jj])
                end
            end
        end
    end 
    return nothing
end

function ResidualContinuity2D!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    for j in 2:size(R.p,2)-1, i in 2:size(R.p,1)-1
        if type.Pt[i,j] !== :constant 
            Vx_loc     = SMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
            Vy_loc     = SMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
            bcv_loc    = (;)
            type_loc   = (;)
            D          = (;)
            R.p[i,j]   = Continuity(Vx_loc, Vy_loc, P[i,j], P0[i,j], D, phases.c[i,j], materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, Pt0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
                
    ∂R∂Vx = @MMatrix zeros(2,3)
    ∂R∂Vy = @MMatrix zeros(3,2)
    ∂R∂P  = @MMatrix zeros(1,1)
    
    Vx_loc= @MMatrix zeros(2,3)
    Vy_loc= @MMatrix zeros(3,2)
    P_loc = @MMatrix zeros(1,1)

    for j in 2:size(P, 2)-1, i in 2:size(P, 1)-1
        Vx_loc    .= SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc    .= SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
        P_loc     .= SMatrix{1,1}(        P[ii,jj] for ii in i:i,   jj in j:j  )
        bcv_loc    = (;)
        type_loc   = (;)
        D          = (;)
        
        fill!(∂R∂Vx, 0e0)
        fill!(∂R∂Vy, 0e0)
        fill!(∂R∂P , 0e0)
        ∂Vx, ∂Vy, ∂P = ad_partial_gradients(Continuity, (Vx_loc, Vy_loc, P_loc), Pt0[i,j], D, phases.c[i,j], materials, type_loc, bcv_loc, Δ)
        ∂R∂Vx .= ∂Vx
        ∂R∂Vy .= ∂Vy
        ∂R∂P  .= ∂P

        # Pt --- Vx
        Local = SMatrix{2,3}(num.Vx[ii,jj] for ii in i:i+1, jj in j:j+2)# .* pattern[3][1]        
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pt --- Vy
        Local = SMatrix{3,2}(num.Vy[ii,jj] for ii in i:i+2, jj in j:j+1) #.* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end

        # Pt --- Pt
        if num.Pt[i,j]>0
            K[3][3][num.Pt[i,j], num.Pt[i,j]] = ∂R∂P[1,1]
        end
    end
    return nothing
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
    end
    # ############ Fields Pt ############
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
    end
    ############ End ############
end

function SetRHS!(r, R, number, type, nc)

    nVx, nVy   = maximum(number.Vx), maximum(number.Vy)

    for j=2:nc.y+3-1, i=2:nc.x+3-1
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            r[ind] = R.x[i,j]
        end
    end
    for j=2:nc.y+3-1, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            r[ind] = R.y[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            r[ind] = R.p[i,j]
        end
    end
end

function UpdateSolution!(V, Pt, dx, number, type, nc)

    nVx, nVy   = maximum(number.Vx), maximum(number.Vy)

    for j=1:size(V.x,2), i=1:size(V.x,1)
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            V.x[i,j] += dx[ind]
        end
    end
 
    for j=1:size(V.y,2), i=1:size(V.y,1)
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            V.y[i,j] += dx[ind]
        end
    end
    
    for I in eachindex(Pt)
        if type.Pt[I] == :in
            ind = number.Pt[I] + nVx + nVy
            Pt[I] += dx[ind]
        end
    end

    # Set E/W periodicity
    for j=2:nc.y+3-1
        if type.Vx[nc.x+3-1,j] == :periodic
            V.x[nc.x+3-1,j] = V.x[2,j]
            V.x[nc.x+3-0,j] = V.x[3,j]
            V.x[       1,j] = V.x[nc.x+3-2,j]
        end
        if type.Vy[nc.x+3,j] == :periodic
            V.y[nc.x+3-0,j] = V.y[3,j]
            V.y[nc.x+3+1,j] = V.y[4,j]
            V.y[1,j]        = V.y[nc.x+3-2,j]
            V.y[2,j]        = V.y[nc.x+3-1,j]
        end
        if j<=nc.y+2
            if type.Pt[nc.x+2,j] == :periodic
                Pt[nc.x+2,j] = Pt[2,j]
                Pt[1,j]      = Pt[nc.x+1,j]
            end
        end
    end 

    # Set S/N periodicity
    for i=2:nc.x+3-1
        if type.Vx[i,nc.y+3] == :periodic
            V.x[i,nc.y+3-0] = V.x[i,3]
            V.x[i,nc.y+3+1] = V.x[i,4]
            V.x[i,1]        = V.x[i,nc.y+3-2]
            V.x[i,2]        = V.x[i,nc.y+3-1]
        end
        if type.Vy[i,nc.y+3-1] == :periodic
            V.y[i,nc.y+3-1] = V.y[i,2]
            V.y[i,nc.y+3-0] = V.y[i,3]
            V.y[i,       1] = V.y[i,nc.y+3-2]
        end
        if i<=nc.x+2
            if type.Pt[i,nc.y+2] == :periodic
                Pt[i,nc.y+2] = Pt[i,2]
                Pt[i,1]      = Pt[i,nc.y+1]
            end
        end
    end

end

function Numbering!(N, type, nc)
    
    ndof  = 0
    neq   = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west  = sum(any(i->i==:periodic, type.Vx[1,3:end-2], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vx[3:end-2,2], dims=1)) > 0

    shift  = (periodic_west) ? 1 : 0 
    # Loop through inner nodes of the mesh
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vx[i,j] == :Dirichlet_normal || (type.Vx[i,j] == :periodic && i==nc.x+3-1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vx[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx[1,:]     .= N.Vx[end-2,:]
        N.Vx[end-1,:] .= N.Vx[2,:]
        N.Vx[end,:]   .= N.Vx[3,:]
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
    periodic_west  = sum(any(i->i==:periodic, type.Vy[2,3:end-2], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vy[3:end-2,1], dims=1)) > 0
    shift = periodic_south ? 1 : 0
    # Loop through inner nodes of the mesh
    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] == :periodic && j==nc.y+3-1)
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
end


function LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x 
    Vi.y .= V.y 
    Pti  .= Pt
    for i in eachindex(α)
        V.x .= Vi.x 
        V.y .= Vi.y
        Pt  .= Pti
        UpdateSolution!(V, Pt, α[i].*dx, number, type, nc)
        TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)
        ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
        ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        rvec[i] = @views norm(R.x[inx_Vx,iny_Vx])/length(R.x[inx_Vx,iny_Vx]) + norm(R.y[inx_Vy,iny_Vy])/length(R.y[inx_Vy,iny_Vy]) + norm(R.p[inx_c,iny_c])/length(R.p[inx_c,iny_c])  
    end
    imin = argmin(rvec)
    V.x .= Vi.x 
    V.y .= Vi.y
    Pt  .= Pti
    return imin
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)

    _ones = @SVector ones(4)
    D_test = @MMatrix ones(4,4)
    s = 1 

    periodic_west  = sum(any(i->i==:periodic, type.Vx[1,3:end-2], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vx[3:end-2,2], dims=1)) > 0

    # Loop over centroids
    for j=1+s:size(ε̇.xx,2)-s, i=1+s:size(ε̇.xx,1)-s
        if (i==1 && j==1) || (i==size(ε̇.xx,1) && j==1) || (i==1 && j==size(ε̇.xx,2)) || (i==size(ε̇.xx,1) && j==size(ε̇.xx,2))
            # Avoid the outer corners - nothing is well defined there ;)
        else
            Vx     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1,   jj in j:j+2)
            Vy     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2,   jj in j:j+1)
            bcx    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
            bcy    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
            typex  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
            typey  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
            τxy0   = SMatrix{2,2}(    τ0.xy[ii,jj] for ii in i:i+1,   jj in j:j+1)

            Vx = SetBCVx1(Vx, typex, bcx, Δ)
            Vy = SetBCVy1(Vy, typey, bcy, Δ)

            # if i==2
            #     printxy(typex)
            #     printxy(typey)
            # end

            Dxx = ∂x_inn(Vx) / Δ.x 
            Dyy = ∂y_inn(Vy) / Δ.y 
            Dxy = ∂y(Vx) / Δ.y
            Dyx = ∂x(Vy) / Δ.x
            
            Dkk = Dxx .+ Dyy
            ε̇xx = @. Dxx - Dkk ./ 3
            ε̇yy = @. Dyy - Dkk ./ 3
            ε̇xy = @. (Dxy + Dyx) ./ 2
            ε̇̄xy = av(ε̇xy)
        
            # Visco-elasticity
            G     = materials.G[phases.c[i,j]]
            τ̄xy0  = av(τxy0)
            ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), Pt[i,j]])

            # beta = materials.β[phases.c[i,j]]
            # @show Dkk[1] + beta[1]*(Pt[i,j]-Pt0[i,j])/Δ.t

            # Tangent operator used for Newton Linearisation
            stress_state, τ_vec, jac = ad_value_and_jacobian_first(StressVector!, ε̇vec, Dkk[1], Pt0[i,j], materials, phases.c[i,j], Δ)
            _, η_local, λ̇_local, τII_local = stress_state

            @views 𝐷_ctl.c[i,j] .= jac

            # Tangent operator used for Picard Linearisation
            𝐷.c[i,j] .= diagm(2 * η_local * _ones)
            𝐷.c[i,j][4,4] = 1

            # ############### TEST
            # ε̇vec   = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), Dkk[1]])
            # jac2   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector_div!, ε̇vec, Const(Dkk[1]), Const(Pt0[i,j]), Const(materials), Const(phases.c[i,j]), Const(Δ))

            # @views D_test[:,1] .= jac2.derivs[1][1][1]
            # @views D_test[:,2] .= jac2.derivs[1][2][1]
            # @views D_test[:,3] .= jac2.derivs[1][3][1]
            # @views D_test[:,4] .= jac2.derivs[1][4][1]

            # K = 1 / materials.β[phases.c[i,j]]
            # C = @SMatrix[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1/(K*Δ.t)]
            # # 𝐷.c[i,j][4,4] = -K*Δ.t

            # 𝐷_ctl.c[i,j] .= D_test*C
            # ############### TEST

            # Update stress
            τ.xx[i,j]  = τ_vec[1]
            τ.yy[i,j]  = τ_vec[2]
            τ.II[i,j]  = τII_local
            ε̇.xx[i,j]  = ε̇xx[1]
            ε̇.yy[i,j]  = ε̇yy[1]
            λ̇.c[i,j]   = λ̇_local
            η.c[i,j]   = η_local
            ΔPt.c[i,j] = (τ_vec[4] - Pt[i,j])
        end
    end

    # for j=2:size(ε̇.xx,2)-1 
    #         i = 1
    #         @views 𝐷_ctl.c[i,j] .= -𝐷_ctl.c[2,j]
    #         @views 𝐷.c[i,j]     .= -𝐷.c[2,j]
    #         i = size(ε̇.xx,1)
    #         @views 𝐷_ctl.c[i,j] .= -𝐷_ctl.c[1,j]
    #         @views 𝐷.c[i,j]     .= -𝐷.c[1,j]
    # end

    # # For periodic cases
    if periodic_west
        for j=2:size(ε̇.xx,2)-1 
            i = 1
            @views 𝐷_ctl.c[i,j] .= 𝐷_ctl.c[end-1,j]
            @views 𝐷.c[i,j]     .= 𝐷.c[end-1,j]
            i = size(ε̇.xx,1)
            @views 𝐷_ctl.c[i,j] .= 𝐷_ctl.c[2,j]
            @views 𝐷.c[i,j]     .= 𝐷.c[2,j]
        end
    end
    if periodic_south
        for i=2:size(ε̇.xx,1)-1 
            j = 1
            @views 𝐷_ctl.c[i,j] .= 𝐷_ctl.c[i,end-1]
            @views 𝐷.c[i,j]     .= 𝐷.c[i,end-1]
            j = size(ε̇.xx,2)
            @views 𝐷_ctl.c[i,j] .= 𝐷_ctl.c[i,2]
            @views 𝐷.c[i,j]     .= 𝐷.c[i,2]
        end
    end

    # @show "vertices"

    # Loop over vertices
    for j=1+s:size(ε̇.xy,2)-s, i=1+s:size(ε̇.xy,1)-s
        Vx     = SMatrix{3,2}(      V.x[ii,jj] for ii in i-1:i+1, jj in j:j+1  )
        Vy     = SMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1  , jj in j-1:j+1)
        bcx    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j:j+1  )
        bcy    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i:i+1  , jj in j-1:j+1)
        typex  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j:j+1  )
        typey  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i:i+1  , jj in j-1:j+1)
        τxx0   = SMatrix{2,2}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-1:j)
        τyy0   = SMatrix{2,2}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-1:j)
        P      = SMatrix{2,2}(       Pt[ii,jj] for ii in i-1:i,   jj in j-1:j)
        P0     = SMatrix{2,2}(       Pt0[ii,jj] for ii in i-1:i,   jj in j-1:j)

        Vx     = SetBCVx1(Vx, typex, bcx, Δ)
        Vy     = SetBCVy1(Vy, typey, bcy, Δ)
    
        Dxx    = ∂x(Vx) / Δ.x
        Dyy    = ∂y(Vy) / Δ.y
        Dxy    = ∂y_inn(Vx) / Δ.y
        Dyx    = ∂x_inn(Vy) / Δ.x

        Dkk   = @. Dxx + Dyy
        ε̇xx   = @. Dxx - Dkk / 3
        ε̇yy   = @. Dyy - Dkk / 3
        ε̇xy   = @. (Dxy + Dyx) /2
        ε̇̄xx   = av(ε̇xx)
        ε̇̄yy   = av(ε̇yy)
        
        # Visco-elasticity
        G     = materials.G[phases.v[i,j]]
        τ̄xx0  = av(τxx0)
        τ̄yy0  = av(τyy0)
        P̄     = av(   P)
        P̄0    = av(  P0)
        D̄kk   = av( Dkk)
 
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t), P̄[1]])
        
        # Tangent operator used for Newton Linearisation
        stress_state, τ_vec, jac = ad_value_and_jacobian_first(StressVector!, ε̇vec, D̄kk[1], P̄0[1], materials, phases.v[i,j], Δ)
        _, η_local, λ̇_local, _ = stress_state

        @views 𝐷_ctl.v[i,j] .= jac

        # Tangent operator used for Picard Linearisation
        𝐷.v[i,j] .= diagm(2 * η_local * _ones)
        𝐷.v[i,j][4,4] = 1

        # ############### TEST
        # ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t), D̄kk[1]])
        # jac2   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector_div!, ε̇vec, Const(D̄kk[1]), Const(P̄0[1]), Const(materials), Const(phases.v[i,j]), Const(Δ))

        # @views D_test[:,1] .= jac2.derivs[1][1][1]
        # @views D_test[:,2] .= jac2.derivs[1][2][1]
        # @views D_test[:,3] .= jac2.derivs[1][3][1]
        # @views D_test[:,4] .= jac2.derivs[1][4][1]

        # K = 1 / materials.β[phases.c[i,j]]
        # C = @SMatrix[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1/(K*Δ.t)]

        # 𝐷_ctl.v[i,j] .= D_test*C
        # ############### TEST

        # Update stress
        τ.xy[i,j] = τ_vec[3]
        ε̇.xy[i,j] = ε̇xy[1]
        λ̇.v[i,j]  = λ̇_local
        η.v[i,j]  = η_local
        # τ.xy[i,j] = 2*jac.val[2]*(ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t))
    end
end
