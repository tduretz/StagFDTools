using StagFDTools, Enzyme
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

    elseif config == :EW_Neumann
        # -------- Vx -------- #
        type.Vx[inx_Vx,iny_Vx]  .= :in       
        type.Vx[1,iny_Vx]       .= :Neumann_normal
        type.Vx[end-0,iny_Vx]   .= :Neumann_normal
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
        
    end
end

function SMomentum_x_Generic(Vx_loc, Vy_loc, Pt, λ̇, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
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
    β    = SVector{2, Float64}( materials.β[phases.c[:]] )
    ψ    = SVector{2, Float64}( materials.ψ[phases.c[:]] )
    Ptc  = SVector{2, Float64}( @. Pt[:,2] + comp * λ̇[:] * Δ.t / β * sind(ψ) )

    # Stress
    τxx = @MVector zeros(2)
    τxy = @MVector zeros(2)
    for i=1:2
        τxx[i] = (𝐷.c[i][1,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][1,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][1,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][1,4] - (𝐷.c[i][4,4] - 1)) * Pt[i,2]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                       * P̄t[i]
    end

    # Residual
    fx  = ( τxx[2]  - τxx[1] ) * invΔx
    fx += ( τxy[2]  - τxy[1] ) * invΔy
    fx -= ( Ptc[2]  - Ptc[1] ) * invΔx
    # fx *= -1 * Δ.x * Δ.y

    return fx
end

function SMomentum_y_Generic(Vx_loc, Vy_loc, Pt, λ̇, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
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
    β    = SVector{2, Float64}( materials.β[phases.c[:]] )
    ψ    = SVector{2, Float64}( materials.ψ[phases.c[:]] )
    Ptc  = SVector{2, Float64}( @. Pt[2,:] + comp * λ̇[:] * Δ.t / β * sind(ψ) )

    # Stress
    τyy = @MVector zeros(2)
    τxy = @MVector zeros(2)
    for i=1:2
        τyy[i] = (𝐷.c[i][2,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][2,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][2,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][2,4] - (𝐷.c[i][4,4] - 1.)) * Pt[2,i]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                        * P̄t[i]
    end

    # Residual
    fy  = ( τyy[2]  -  τyy[1] ) * invΔy
    fy += ( τxy[2]  -  τxy[1] ) * invΔx
    fy -= ( Ptc[2]  -  Ptc[1])  * invΔy
    # fy *= -1 * Δ.x * Δ.y
    
    return fy
end


function Continuity(Vx, Vy, Pt, Pt0, D, phase, materials, type_loc, bcv_loc, Δ)
    invΔx = 1 / Δ.x
    invΔy = 1 / Δ.y
    invΔt = 1 / Δ.t
    β     = materials.β[phase]
    comp  = materials.compressible
    return ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy) + comp * β * (Pt[1] - Pt0) * invΔt
end

function ResidualMomentum2D_x!(R, V, P, P0, λ̇, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        if type.Vx[i,j] == :in
            Vx_loc     = SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc     = SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
            P_loc      = SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            λ̇_loc      = SMatrix{2,1}(      λ̇.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
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
    
            R.x[i,j]   = SMomentum_x_Generic(Vx_loc, Vy_loc, P_loc, λ̇_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, P0, λ̇, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,3)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
                
    Vx_loc = @MMatrix zeros(3,3)
    Vy_loc = @MMatrix zeros(4,4)
    P_loc  = @MMatrix zeros(2,3)
    λ̇_loc  = @MMatrix zeros(2,1)

    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        
        if type.Vx[i,j] == :in

            bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0) 
            
            Vx_loc    .= SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc    .= SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            P_loc     .= SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            λ̇_loc     .= SMatrix{2,1}(      λ̇.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)

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

            fill!(∂R∂Vx, 0e0)
            fill!(∂R∂Vy, 0e0)
            fill!(∂R∂Pt, 0e0)
            autodiff(Enzyme.Reverse, SMomentum_x_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(λ̇_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
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

function ResidualMomentum2D_y!(R, V, P, P0, λ̇, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)                 
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
            λ̇_loc      = SMatrix{1,2}(      λ̇.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
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

            R.y[i,j]   = SMomentum_y_Generic(Vx_loc, Vy_loc, P_loc, λ̇_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, P0, λ̇, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(3,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    
    Vx_loc = @MMatrix zeros(4,4)
    Vy_loc = @MMatrix zeros(3,3)
    P_loc  = @MMatrix zeros(3,2)
    λ̇_loc  = @MMatrix zeros(1,2)
       
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
            λ̇_loc     .= @inline SMatrix{1,2}(@inbounds       λ̇.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
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
            autodiff(Enzyme.Reverse, SMomentum_y_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(λ̇_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
            
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

function ResidualContinuity2D!(R, V, P, P0, λ̇, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    for j in 2:size(R.p,2)-1, i in 2:size(R.p,1)-1
        Vx_loc     = SMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = SMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (;)
        type_loc   = (;)
        D          = (;)
        R.p[i,j]   = Continuity(Vx_loc, Vy_loc, P[i,j], P0[i,j], D, phases.c[i,j], materials, type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, Pt0, λ̇, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
                
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)
    ∂R∂P  = @MMatrix zeros(1,1)
    
    Vx_loc= @MMatrix zeros(3,2)
    Vy_loc= @MMatrix zeros(2,3)
    P_loc = @MMatrix zeros(1,1)

    for j in 2:size(P, 2)-1, i in 2:size(P, 1)-1
        Vx_loc    .= SMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc    .= SMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        P_loc     .= SMatrix{1,1}(        P[ii,jj] for ii in i:i,   jj in j:j  )
        bcv_loc    = (;)
        type_loc   = (;)
        D          = (;)
        
        fill!(∂R∂Vx, 0e0)
        fill!(∂R∂Vy, 0e0)
        fill!(∂R∂P , 0e0)
        autodiff(Enzyme.Reverse, Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂P), Const(Pt0[i,j]), Const(D), Const(phases.c[i,j]), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))

        # Pt --- Vx
        Local = SMatrix{2,3}(num.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) .* pattern[3][1]        
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pt --- Vy
        Local = SMatrix{3,2}(num.Vy[ii,jj] for ii in i:i+2, jj in j:j+1) .* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end

        # Pt --- Pt
        K[3][3][num.Pt[i,j], num.Pt[i,j]] = ∂R∂P[1,1]
    end
    return nothing
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


function SetBCVx!(Vx_loc, bcx_loc, bcv, Δ)

    for ii in axes(Vx_loc, 1)

        # Set Vx boundaries at S (this must be done 1st)
        if bcx_loc[ii,begin] == :Neumann 
            Vx_loc[ii,begin] =  Vx_loc[ii,begin+1] - Δ.y*bcv.∂Vx∂y_BC[ii,1]
        elseif bcx_loc[ii,begin] == :Dirichlet 
            Vx_loc[ii,begin] = -Vx_loc[ii,begin+1] + 2*bcv.Vx_BC[ii,1]
        end
        if bcx_loc[ii,begin] == :out 
            if bcx_loc[ii,begin+1] == :Neumann
                Vx_loc[ii,begin+1] =  Vx_loc[ii,begin+2] -   Δ.y*bcv.∂Vx∂y_BC[ii,1]
                Vx_loc[ii,begin]   =  Vx_loc[ii,begin+3] - 3*Δ.y*bcv.∂Vx∂y_BC[ii,1] 
            elseif bcx_loc[ii,begin+1] == :Dirichlet
                Vx_loc[ii,begin+1] = -Vx_loc[ii,begin+2] + 2*bcv.Vx_BC[ii,1]
                Vx_loc[ii,begin]   = -Vx_loc[ii,begin+3] + 2*bcv.Vx_BC[ii,1] 
            end
        end

        # Set Vx boundaries at N (this must be done 1st)
        if bcx_loc[ii,end] == :Neumann 
            Vx_loc[ii,end] =  Vx_loc[ii,end-1] + Δ.y*bcv.∂Vx∂y_BC[ii,2] 
        elseif bcx_loc[ii,end] == :Dirichlet 
            Vx_loc[ii,end] = -Vx_loc[ii,end-1] + 2*bcv.Vx_BC[ii,2]
        end
        if bcx_loc[ii,end] == :out
            if bcx_loc[ii,end-1] == :Neumann
                Vx_loc[ii,end-1] =  Vx_loc[ii,end-2] +   Δ.y*bcv.∂Vx∂y_BC[ii,2] 
                Vx_loc[ii,end]   =  Vx_loc[ii,end-3] + 3*Δ.y*bcv.∂Vx∂y_BC[ii,2]   
            elseif bcx_loc[ii,3] == :Dirichlet
                Vx_loc[ii,end-1] = -Vx_loc[ii,end-2] + 2*bcv.Vx_BC[ii,2] 
                Vx_loc[ii,end]   = -Vx_loc[ii,end-3] + 2*bcv.Vx_BC[ii,2]  
            end
        end
    end

    # for jj in axes(Vx_loc, 2)
    #     # Set Vx boundaries at W (this must be done 2nd)
    #     if bcx_loc[1,jj] == :out
    #         Vx_loc[1,jj] = Vx_loc[2,jj] - Δ.x*bcv.∂Vx∂x_BC[1,jj] 
    #     end
    #     # Set Vx boundaries at E (this must be done 2nd)
    #     if bcx_loc[3,jj] == :out
    #         Vx_loc[3,jj] = Vx_loc[2,jj] + Δ.x*bcv.∂Vx∂x_BC[2,jj] 
    #     end
    # end
end

function SetBCVy!(Vy_loc, bcy_loc, bcv, Δ)
    
    for jj in axes(Vy_loc, 2)

        # Set Vy boundaries at W (this must be done 1st)
        if bcy_loc[begin,jj] == :Neumann 
            Vy_loc[begin,jj] =  Vy_loc[begin+1,jj] - Δ.x*bcv.∂Vy∂x_BC[1,jj] 
        elseif bcy_loc[begin,jj] == :Dirichlet 
            Vy_loc[begin,jj] = -Vy_loc[begin+1,jj] + 2*bcv.Vy_BC[1,jj]
        end
        if bcy_loc[begin,jj] == :out
            if bcy_loc[begin+1,jj] == :Neumann 
                Vy_loc[begin+1,jj] = Vy_loc[begin+2,jj] -   Δ.y*bcv.∂Vy∂x_BC[1,jj] 
                Vy_loc[begin,jj]   = Vy_loc[begin+3,jj] - 3*Δ.y*bcv.∂Vy∂x_BC[1,jj] 
            elseif bcy_loc[begin+1,jj] == :Dirichlet
                Vy_loc[begin+1,jj] = -Vy_loc[begin+2,jj] + 2*bcv.Vy_BC[1,jj]
                Vy_loc[begin,jj]   = -Vy_loc[begin+3,jj] + 2*bcv.Vy_BC[1,jj]
            end 
        end

        # Set Vy boundaries at E (this must be done 1st)
        if bcy_loc[end,jj] == :Neumann 
            Vy_loc[end,jj] = Vy_loc[end-1,jj] + Δ.x*bcv.∂Vy∂x_BC[1,jj] 
        elseif bcy_loc[end,jj] == :Dirichlet 
            Vy_loc[end,jj] = -Vy_loc[end-1,jj] + 2*bcv.Vy_BC[2,jj]
        end
        if bcy_loc[end,jj] == :out
            if bcy_loc[end-1,jj] == :Neumann 
                Vy_loc[end-1,jj] = Vy_loc[end-2,jj] +   Δ.y*bcv.∂Vy∂x_BC[1,jj]
                Vy_loc[end,jj]   = Vy_loc[end-3,jj] + 3*Δ.y*bcv.∂Vy∂x_BC[1,jj]
            elseif bcy_loc[3,jj] == :Dirichlet 
                Vy_loc[end-1,jj] = -Vy_loc[end-2,jj] + 2*bcv.Vy_BC[2,jj]
                Vy_loc[end,jj]   = -Vy_loc[end-3,jj] + 2*bcv.Vy_BC[2,jj]
            end
        end
    end

    # for ii in axes(Vy_loc, 1)
    #     # Set Vy boundaries at S (this must be done 2nd)
    #     if bcy_loc[ii,1] == :out
    #         Vy_loc[ii,1] = Vy_loc[ii,2] - Δ.y*bcv.∂Vy∂y_BC[ii,1]
    #     end
    #     # Set Vy boundaries at S (this must be done 2nd)
    #     if bcy_loc[ii,3] == :out
    #         Vy_loc[ii,3] = Vy_loc[ii,2] + Δ.y*bcv.∂Vy∂y_BC[ii,2]
    #     end
    # end
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
    
    for j=1:size(Pt,2), i=1:size(Pt,1)
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            Pt[i,j] += dx[ind]
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
    neq_Pt                     = nc.x * nc.y
    N.Pt[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ 0*neq, nc.x, nc.y)

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

# function Numbering!(N, type, nc)
    
#     ndof  = 0
#     neq   = 0
#     noisy = false

#     ############ Fields Vx ############
#     periodic_west  = sum(any(i->i==:periodic, type.Vx[2,:], dims=2)) > 0
#     periodic_south = sum(any(i->i==:periodic, type.Vx[:,2], dims=1)) > 0

#     shift  = (periodic_west) ? 1 : 0 
#     # Loop through inner nodes of the mesh
#     for j=3:nc.y+4-2, i=2:nc.x+3-1
#         if type.Vx[i,j] == :Dirichlet_normal || (type.Vx[i,j] != :periodic && i==nc.x+3-1)
#             # Avoid nodes with constant velocity or redundant periodic nodes
#         else
#             ndof+=1
#             N.Vx[i,j] = ndof  
#         end
#     end

#     # Copy equation indices for periodic cases
#     if periodic_west
#         N.Vx[1,:] .= N.Vx[end-2,:]
#     end

#     # Copy equation indices for periodic cases
#     if periodic_south
#         # South
#         N.Vx[:,1] .= N.Vx[:,end-3]
#         N.Vx[:,2] .= N.Vx[:,end-2]
#         # North
#         N.Vx[:,end]   .= N.Vx[:,4]
#         N.Vx[:,end-1] .= N.Vx[:,3]
#     end
#     noisy ? printxy(N.Vx) : nothing

#     neq = maximum(N.Vx)

#     ############ Fields Vy ############
#     ndof  = 0
#     periodic_west  = sum(any(i->i==:periodic, type.Vy[2,:], dims=2)) > 0
#     periodic_south = sum(any(i->i==:periodic, type.Vy[:,2], dims=1)) > 0
#     shift = periodic_south ? 1 : 0
#     # Loop through inner nodes of the mesh
#     for j=2:nc.y+3-1, i=3:nc.x+4-2
#         if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] != :periodic && j==nc.y+3-1)
#             # Avoid nodes with constant velocity or redundant periodic nodes
#         else
#             ndof+=1
#             N.Vy[i,j] = ndof  
#         end
#     end

#     # Copy equation indices for periodic cases
#     if periodic_south
#         N.Vy[:,1] .= N.Vy[:,end-2]
#     end

#     # Copy equation indices for periodic cases
#     if periodic_west
#         # West
#         N.Vy[1,:] .= N.Vy[end-3,:]
#         N.Vy[2,:] .= N.Vy[end-2,:]
#         # East
#         N.Vy[end,:]   .= N.Vy[4,:]
#         N.Vy[end-1,:] .= N.Vy[3,:]
#     end
#     noisy ? printxy(N.Vy) : nothing

#     neq = maximum(N.Vy)

#     ############ Fields Pt ############
#     neq_Pt                     = nc.x * nc.y
#     N.Pt[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ 0*neq, nc.x, nc.y)

#     if periodic_west
#         N.Pt[1,:]   .= N.Pt[end-1,:]
#         N.Pt[end,:] .= N.Pt[2,:]
#     end

#     if periodic_south
#         N.Pt[:,1]   .= N.Pt[:,end-1]
#         N.Pt[:,end] .= N.Pt[:,2]
#     end
#     noisy ? printxy(N.Pt) : nothing

#     neq = maximum(N.Pt)

# end