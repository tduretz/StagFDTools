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

function SetRHS!(r, R, number, type, nc)

    nVx, nVy   = maximum(number.Vx), maximum(number.Vy)

    for j=2:nc.y+3-1, i=2:nc.x+3-1
        if type.Vx[i,j] === :in
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
    noisy && printxy(N.Vx)

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

###################################################################################
###################################################################################
###################################################################################

function Continuity(Vx_loc, Vy_loc, Pt, Pt0, D, J, phase, materials, type_loc, bcv_loc, Δ)
    _Δx = 1 / Δ.ξ
    _Δy = 1 / Δ.η
    _Δt = 1 / Δ.t
    # BC
    Vx    = SetBCVx1(Vx_loc, type_loc.x, bcv_loc.x, Δ)
    Vy    = SetBCVy1(Vy_loc, type_loc.y, bcv_loc.y, Δ)
    V̄x    = av(Vx)
    V̄y    = av(Vy)
    β     = materials.β[phase]
    η     = materials.β[phase]
    comp  = materials.compressible
    ∂Vx∂x = (Vx[2,2] - Vx[1,2]) * _Δx * J[1,1][1,1] + (V̄x[1,2] - V̄x[1,1]) * _Δy * J[1,1][1,2]
    ∂Vy∂y = (V̄y[2,1] - V̄y[1,1]) * _Δx * J[1,1][2,1] + (Vy[2,2] - Vy[2,1]) * _Δy * J[1,1][2,2] 
    f     =  (∂Vx∂x + ∂Vy∂y) + comp * β * (Pt[1] - Pt0) * _Δt #+ 1/(1000*η)*Pt[1]
    f    *= max(_Δx, _Δy)
    return f
end

function ResidualContinuity2D!(R, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ) 
                
    for j in 2:size(R.p,2)-1, i in 2:size(R.p,1)-1
        if type.Pt[i,j] !== :constant 
            Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
            Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
            bcx_loc    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1, jj in j:j+2)
            bcy_loc    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
            typex_loc  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1, jj in j:j+2)
            typey_loc  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
            Jinv_c     = SMatrix{1,1}(   Jinv.c[ii,jj] for ii in i:i,   jj in j:j  )
            D          = (;)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            R.p[i,j]   = Continuity(Vx_loc, Vy_loc, P[i,j], P0[i,j], D, Jinv_c, phases.c[i,j], materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, Pt0, ΔP, τ0, 𝐷, Jinv, phases, materials, num, pattern, type, BC, nc, Δ) 
                
    ∂R∂Vx = @MMatrix zeros(2,3)
    ∂R∂Vy = @MMatrix zeros(3,2)
    ∂R∂P  = @MMatrix zeros(1,1)
    
    Vx_loc= @MMatrix zeros(2,3)
    Vy_loc= @MMatrix zeros(3,2)
    P_loc = @MMatrix zeros(1,1)

    for j in 2:size(P, 2)-1, i in 2:size(P, 1)-1
        Vx_loc     .= MMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc     .= MMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
        P_loc      .= MMatrix{1,1}(        P[ii,jj] for ii in i:i,   jj in j:j  )
        bcx_loc    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcy_loc    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        typex_loc  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1, jj in j:j+2)
        typey_loc  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
        Jinv_c     = SMatrix{1,1}(   Jinv.c[ii,jj] for ii in i:i,   jj in j:j  )
        D          = (;)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        fill!(∂R∂Vx, 0e0)
        fill!(∂R∂Vy, 0e0)
        fill!(∂R∂P , 0e0)
        autodiff(Enzyme.Reverse, Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂P), Const(Pt0[i,j]), Const(D), Const(Jinv_c), Const(phases.c[i,j]), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))

        K31 = K[3][1]
        K32 = K[3][2]
        K33 = K[3][3]

        # Pt --- Vx
        Local = SMatrix{2,3}(num.Vx[ii,jj] for ii in i:i+1, jj in j:j+2) .* pattern[3][1]        
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K31[num.Pt[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pt --- Vy
        Local = SMatrix{3,2}(num.Vy[ii,jj] for ii in i:i+2, jj in j:j+1) .* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K32[num.Pt[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end
        # Pt --- Pt
        if num.Pt[i,j]>0
            K33[num.Pt[i,j], num.Pt[i,j]] = ∂R∂P[1,1]
        end
    end
    return nothing
end

###################################################################################
###################################################################################
###################################################################################

function SetBCVx1(Vx, typex, bcx, Δ)

    if size(Vx, 2) > 3
        jmax = 2
    else
        jmax = 1
    end

    MVx = MMatrix(Vx)
    # N/S
    for ii in axes(typex, 1)
        for j=1:jmax 
            if typex[ii,j] == :Dirichlet_tangent
                MVx[ii,j] = fma(2, bcx[ii,j], -Vx[ii,j+1])
            elseif typex[ii,j] == :Neumann_tangent
                MVx[ii,j] = fma(Δ.η, bcx[ii,j], Vx[ii,j+1])
            end

            if typex[ii,end-j+1] == :Dirichlet_tangent
                MVx[ii,end-j+1] = fma(2, bcx[ii,end-j+1], -Vx[ii,end-j])
            elseif typex[ii,end-j+1] == :Neumann_tangent
                MVx[ii,end-j+1] = fma(Δ.η, bcx[ii,end-j+1], Vx[ii,end-j])
            end
        end
    end
    # E/W
    for jj in axes(typex, 2)
        if typex[1,jj] == :Neumann_normal
            MVx[1,jj] = fma(2, Δ.ξ*bcx[1,jj], Vx[2,jj])
        end
        if typex[end,jj] == :Neumann_normal
            MVx[end,jj] = fma(2,-Δ.ξ*bcx[end,jj], Vx[end-1,jj])
        end
    end
    return SMatrix(MVx)
end

function SetBCVy1(Vy, typey, bcy, Δ)

    if size(Vy, 1) > 3
        imax = 2
    else
        imax = 1
    end

    MVy = MMatrix(Vy)
    # E/W
    for jj in axes(typey, 2)
        for i=1:imax
            if typey[i,jj] == :Dirichlet_tangent
                MVy[i,jj] = fma(2, bcy[i,jj], -Vy[i+1,jj])
            elseif typey[i,jj] == :Neumann_tangent
                MVy[i,jj] = fma(Δ.ξ, bcy[i,jj], Vy[i+1,jj])
            end

            if typey[end-i+1,jj] == :Dirichlet_tangent
                MVy[end-i+1,jj] = fma(2, bcy[end-i+1,jj], -Vy[end-i,jj])
            elseif typey[end-i+1,jj] == :Neumann_tangent
                MVy[end-i+1,jj] = fma(Δ.ξ, bcy[end-i+1,jj], Vy[end-i,jj])
            end
        end
    end
    # N/S
    for ii in axes(typey, 1)
        if typey[ii,1] == :Neumann_normal
            MVy[ii,1] = fma(2, Δ.η*bcy[ii,1], Vy[ii,2])
        end
        if typey[ii,end] == :Neumann_normal
            MVy[ii,end] = fma(2,-Δ.η*bcy[ii,end], Vy[ii,end-1])
        end
    end
    return SMatrix(MVy)
end

function SMomentum_x_Generic(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, J, phases, materials, type, bcv, Δ)
    
    _Δξ, _Δη = 1 / Δ.ξ, 1 / Δ.η

    # BC
    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)

    V̄x  = av(Vx)
    V̄y  = av(Vy)
    P̄t  = av(Pt)

    Dxxc =  ∂x_inn(Vx) * _Δξ .* getindex.(J.c, 1, 1)        .+ ∂y(V̄x)     * _Δη .* getindex.(J.c, 1, 2)        # centroids (4, 3)
    Dxxv =  ∂x(V̄x)     * _Δξ .* getindex.(J.v, 1, 1)        .+ ∂y_inn(Vx) * _Δη .* getindex.(J.v, 1, 2)        # vertices  (3, 4)
    Dyyc =  ∂x(V̄y)     * _Δξ .* inn_x(getindex.(J.c, 2, 1)) .+ ∂y_inn(Vy) * _Δη .* inn_x(getindex.(J.c, 2, 2)) # centroids (2, 3)
    Dyyv =  ∂x_inn(Vy) * _Δξ .* inn_y(getindex.(J.v, 2, 1)) .+ ∂y(V̄y)     * _Δη .* inn_y(getindex.(J.v, 2, 2)) # vertices  (3, 2)
    Dxyc =  ∂x_inn(Vx) * _Δξ .* getindex.(J.c, 2, 1)        .+ ∂y(V̄x)     * _Δη .* getindex.(J.c, 2, 2)        # centroids (4, 3)
    Dxyv =  ∂x(V̄x)     * _Δξ .* getindex.(J.v, 2, 1)        .+ ∂y_inn(Vx) * _Δη .* getindex.(J.v, 2, 2)        # vertices  (3, 4)
    Dyxc =  ∂x(V̄y)     * _Δξ .* inn_x(getindex.(J.c, 1, 1)) .+ ∂y_inn(Vy) * _Δη .* inn_x(getindex.(J.c, 1, 2)) # centroids (2, 3)
    Dyxv =  ∂x_inn(Vy) * _Δξ .* inn_y(getindex.(J.v, 1, 1)) .+ ∂y(V̄y)     * _Δη .* inn_y(getindex.(J.v, 1, 2)) # vertices  (3, 2)

    ε̇kkc = inn_x(Dxxc) .+ Dyyc
    ε̇kkv = inn_y(Dxxv) .+ Dyyv
    ε̇xxc = inn_x(Dxxc) .- 1/3 .* ε̇kkc
    ε̇xxv = inn_y(Dxxv) .- 1/3 .* ε̇kkv
    ε̇yyc = Dyyc .- 1/3 .* ε̇kkc
    ε̇yyv = Dyyv .- 1/3 .* ε̇kkv
    ε̇xyc = 1/2 .* (inn_x(Dxyc) .+ Dyxc)
    ε̇xyv = 1/2 .* (inn_y(Dxyv) .+ Dyxv)

    ϵ̇xxc = ε̇xxc
    ϵ̇xxv = ε̇xxv
    ϵ̇yyc = ε̇yyc
    ϵ̇yyv = ε̇yyv
    ϵ̇xyc = ε̇xyc
    ϵ̇xyv = ε̇xyv

    D11, D12, D13, D14 = getindex.(𝐷.c, 1, 1) .- getindex.(𝐷.c, 4, 1), getindex.(𝐷.c, 1, 2) .- getindex.(𝐷.c, 4, 2), getindex.(𝐷.c, 1, 3) .- getindex.(𝐷.c, 4, 3),  getindex.(𝐷.c, 1, 4) .- getindex.(𝐷.c, 4, 4) .+ 1
    D31, D32, D33, D34 = getindex.(𝐷.c, 3, 1), getindex.(𝐷.c, 3, 2), getindex.(𝐷.c, 3, 3), getindex.(𝐷.c, 3, 4)
    τxxc = D11 .* ϵ̇xxc .+ D12 .* ϵ̇yyc .+ D13 .* ϵ̇xyc .+  D14 .* inn_x(Pt)
    τxyc = D31 .* ϵ̇xxc .+ D32 .* ϵ̇yyc .+ D33 .* ϵ̇xyc .+  D34 .* inn_x(Pt)

    D11, D12, D13, D14 = getindex.(𝐷.v, 1, 1) .- getindex.(𝐷.v, 4, 1), getindex.(𝐷.v, 1, 2) .- getindex.(𝐷.v, 4, 2), getindex.(𝐷.v, 1, 3) .- getindex.(𝐷.v, 4, 3),  getindex.(𝐷.v, 1, 4) .- getindex.(𝐷.v, 4, 4) .+ 1
    D31, D32, D33, D34 = getindex.(𝐷.v, 3, 1), getindex.(𝐷.v, 3, 2), getindex.(𝐷.v, 3, 3), getindex.(𝐷.v, 3, 4)
    τxxv = D11 .* ϵ̇xxv .+ D12 .* ϵ̇yyv .+ D13 .* ϵ̇xyv .+  D14 .* P̄t
    τxyv = D31 .* ϵ̇xxv .+ D32 .* ϵ̇yyv .+ D33 .* ϵ̇xyv .+  D34 .* P̄t

    fx  = ∂x_inn(τxxc .- inn(Pt)) * _Δξ .* getindex.(J.Vx, 1, 1) .+ ∂y_inn(τxxv .- inn_x(P̄t)) * _Δη .* getindex.(J.Vx, 1, 2)
    fx += ∂x_inn(τxyc) * _Δξ .* getindex.(J.Vx, 2, 1) .+ ∂y_inn(τxyv) * _Δη .* getindex.(J.Vx, 2, 2)
    fx *= -1/(_Δξ*_Δη)
    
    return fx[1]
end

function ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        if type.Vx[i,j] == :in

            bcx_loc    = @inline SMatrix{5,5}(@inbounds    BC.Vx[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            bcy_loc    = @inline SMatrix{4,4}(@inbounds    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = @inline SMatrix{5,5}(@inbounds  type.Vx[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            typey_loc  = @inline SMatrix{4,4}(@inbounds  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            ph_loc     = @inline SMatrix{2,2}(@inbounds phases.Vy[ii,jj] for ii in i:i+1, jj in j-1:j)

            Vx_loc     = @inline SMatrix{5,5}(@inbounds      V.x[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            Vy_loc     = @inline SMatrix{4,4}(@inbounds      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            P_loc      = @inline SMatrix{4,3}(@inbounds        P[ii,jj] for ii in i-2:i+1,   jj in j-2:j  )
            ΔP_loc     = @inline SMatrix{2,3}(@inbounds       ΔP.c[ii,jj] for ii in i-1:i,   jj in j-2:j  )    
            τ0_loc     = @inline SMatrix{2,2}(@inbounds    τ0.Vy[ii,jj] for ii in i:i+1,   jj in j-1:j  )
            D_c       = @inline SMatrix{2,3}(@inbounds        𝐷.c[ii,jj] for ii in i-1:i+0,   jj in j-2:j  )
            D_v       = @inline SMatrix{3,2}(@inbounds        𝐷.v[ii,jj] for ii in i-1:i+1, jj in j-1:j+0  )

            J_Vx       = @inline SMatrix{1,1}(@inbounds    Jinv.Vx[ii,jj] for ii in i:i,   jj in j:j    )
            J_c       = @inline SMatrix{4,3}(@inbounds    Jinv.c[ii,jj] for ii in i-2:i+1,   jj in j-2:j  )
            J_v       = @inline SMatrix{3,4}(@inbounds    Jinv.v[ii,jj] for ii in i-1:i+1, jj in j-2:j+1  )

            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            Jinv_loc   = (Vx=J_Vx, c=J_c, v=J_v)
            D          = (c=D_c, v=D_v)
    
            R.x[i,j]   = SMomentum_x_Generic(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, D, Jinv_loc, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx  = @MMatrix zeros(5,5)
    ∂R∂Vy  = @MMatrix zeros(4,4)
    ∂R∂Pt  = @MMatrix zeros(4,3)
                
    Vx_loc = @MMatrix zeros(5,5)
    Vy_loc = @MMatrix zeros(4,4)
    P_loc  = @MMatrix zeros(4,3)

    shift    = (x=1, y=2)
    K11 = K[1][1]
    K12 = K[1][2]
    K13 = K[1][3]

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        
        if type.Vx[i,j] == :in

            bcx_loc    = @inline SMatrix{5,5}(@inbounds    BC.Vx[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            bcy_loc    = @inline SMatrix{4,4}(@inbounds    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = @inline SMatrix{5,5}(@inbounds  type.Vx[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            typey_loc  = @inline SMatrix{4,4}(@inbounds  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            ph_loc     = @inline SMatrix{2,2}(@inbounds phases.Vy[ii,jj] for ii in i:i+1, jj in j-1:j)

            Vx_loc    .= @inline SMatrix{5,5}(@inbounds      V.x[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            Vy_loc    .= @inline SMatrix{4,4}(@inbounds      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            P_loc     .= @inline SMatrix{4,3}(@inbounds        P[ii,jj] for ii in i-2:i+1,   jj in j-2:j  )
            ΔP_loc     = @inline SMatrix{2,3}(@inbounds       ΔP.c[ii,jj] for ii in i-1:i,   jj in j-2:j  )    
            τ0_loc     = @inline SMatrix{2,2}(@inbounds    τ0.Vy[ii,jj] for ii in i:i+1,   jj in j-1:j  )
            D_c        = @inline SMatrix{2,3}(@inbounds        𝐷.c[ii,jj] for ii in i-1:i+0,   jj in j-2:j  )
            D_v        = @inline SMatrix{3,2}(@inbounds        𝐷.v[ii,jj] for ii in i-1:i+1, jj in j-1:j+0  )

            J_Vx       = @inline SMatrix{1,1}(@inbounds    Jinv.Vx[ii,jj] for ii in i:i,   jj in j:j    )
            J_c        = @inline SMatrix{4,3}(@inbounds    Jinv.c[ii,jj] for ii in i-2:i+1,   jj in j-2:j  )
            J_v        = @inline SMatrix{3,4}(@inbounds    Jinv.v[ii,jj] for ii in i-1:i+1, jj in j-2:j+1  )

            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            Jinv_loc   = (Vx=J_Vx, c=J_c, v=J_v)
            D          = (c=D_c, v=D_v)

            fill!(∂R∂Vx, 0e0)
            fill!(∂R∂Vy, 0e0)
            fill!(∂R∂Pt, 0e0)
            autodiff(Enzyme.Reverse, SMomentum_x_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ΔP_loc), Const(τ0_loc), Const(D), Const(Jinv_loc), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
            
            num_Vx = @inbounds num.Vx[i,j]
            bounds_Vx = num_Vx > 0
            
            # Vx --- Vx
            Local = SMatrix{5,5}(num.Vx[ii, jj] for ii in i-2:i+2, jj in j-2:j+2) .* pattern[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && bounds_Vx
                    @inbounds K11[num_Vx, Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vx --- Vy
            Local = SMatrix{4,4}(num.Vy[ii, jj] for ii in i-1:i+2, jj in j-2:j+1) .* pattern[1][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && bounds_Vx
                    @inbounds K12[num_Vx, Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vx --- Pt
            Local = SMatrix{4,3}(num.Pt[ii, jj] for ii in i-2:i+1, jj in j-2:j) .* pattern[1][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && bounds_Vx
                    @inbounds K13[num_Vx, Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
        end
    end
    return nothing
end

function SMomentum_y_Generic(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, J, phases, materials, type, bcv, Δ)
    

    ρ = materials.ρ0[1]*materials.g[2]

    _Δξ, _Δη = 1 / Δ.ξ, 1 / Δ.η

    # BC
    Vx = SetBCVx1(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1(Vy_loc, type.y, bcv.y, Δ)

    V̄x  = av(Vx)
    V̄y  = av(Vy)
    P̄t  = av(Pt)

    Dxxc = ∂x_inn(Vx) .* _Δξ .* inn_y(getindex.(J.c, 1, 1)) .+ ∂y(inn_x(V̄x)) .* _Δη .* inn_y(getindex.(J.c, 1, 2))        # centroids (3, 2)
    Dxxv = ∂x(V̄x)     .* _Δξ .* inn_x(getindex.(J.v, 1, 1)) .+ ∂y_inn(Vx)    .* _Δη .* inn_x(getindex.(J.v, 1, 2))        # vertices  (2, 3)
    Dyyc = ∂x(V̄y)     .* _Δξ .* getindex.(J.c, 2, 1)        .+ ∂y_inn(Vy)    .* _Δη .* getindex.(J.c, 2, 2)
    Dyyv = ∂x_inn(Vy) .* _Δξ .* getindex.(J.v, 2, 1)        .+ ∂y(V̄y)        .* _Δη .* getindex.(J.v, 2, 2)
    Dxyc = ∂x_inn(Vx) .* _Δξ .* inn_y(getindex.(J.c, 2, 1)) .+ ∂y(inn_x(V̄x)) .* _Δη .* inn_y(getindex.(J.c, 2, 2))        # centroids (3, 2)
    Dxyv = ∂x(V̄x)     .* _Δξ .* inn_x(getindex.(J.v, 2, 1)) .+ ∂y_inn(Vx)    .* _Δη .* inn_x(getindex.(J.v, 2, 2))        # vertices  (2, 3)
    Dyxc = ∂x(V̄y)     .* _Δξ .* getindex.(J.c, 1, 1)        .+ ∂y_inn(Vy)    .* _Δη .* getindex.(J.c, 1, 2)
    Dyxv = ∂x_inn(Vy) .* _Δξ .* getindex.(J.v, 1, 1)        .+ ∂y(V̄y)        .* _Δη .* getindex.(J.v, 1, 2)

    ε̇kkc = Dxxc .+ inn_y(Dyyc)
    ε̇kkv = Dxxv .+ inn_x(Dyyv)
    ε̇yyc = inn_y(Dyyc) .- 1/3 .* ε̇kkc
    ε̇yyv = inn_x(Dyyv) .- 1/3 .* ε̇kkv
    ε̇xxc = Dxxc .- 1/3 .* ε̇kkc
    ε̇xxv = Dxxv .- 1/3 .* ε̇kkv
    ε̇xyc = 1/2 .* (Dxyc .+ inn_y(Dyxc))
    ε̇xyv = 1/2 .* (Dxyv .+ inn_x(Dyxv))

    ϵ̇xxc = ε̇xxc
    ϵ̇xxv = ε̇xxv
    ϵ̇yyc = ε̇yyc
    ϵ̇yyv = ε̇yyv
    ϵ̇xyc = ε̇xyc
    ϵ̇xyv = ε̇xyv

    D21, D22, D23, D24 = getindex.(𝐷.c, 2, 1) .- getindex.(𝐷.c, 4, 1), getindex.(𝐷.c, 2, 2) .- getindex.(𝐷.c, 4, 2), getindex.(𝐷.c, 2, 3) .- getindex.(𝐷.c, 4, 3),  getindex.(𝐷.c, 2, 4) .- getindex.(𝐷.c, 4, 4) .+ 1
    D31, D32, D33, D34 = getindex.(𝐷.c, 3, 1), getindex.(𝐷.c, 3, 2), getindex.(𝐷.c, 3, 3), getindex.(𝐷.c, 3, 4)
    τyyc = D21 .* ϵ̇xxc .+ D22 .* ϵ̇yyc .+ D23 .* ϵ̇xyc .+  D24 .* inn_y(Pt) 
    τxyc = D31 .* ϵ̇xxc .+ D32 .* ϵ̇yyc .+ D33 .* ϵ̇xyc .+  D34 .* inn_y(Pt) 

    D21, D22, D23, D24 = getindex.(𝐷.v, 2, 1) .- getindex.(𝐷.v, 4, 1), getindex.(𝐷.v, 2, 2) .- getindex.(𝐷.v, 4, 2), getindex.(𝐷.v, 2, 3) .- getindex.(𝐷.v, 4, 3),  getindex.(𝐷.v, 2, 4) .- getindex.(𝐷.v, 4, 4) .+ 1
    D31, D32, D33, D34 = getindex.(𝐷.v, 3, 1), getindex.(𝐷.v, 3, 2), getindex.(𝐷.v, 3, 3), getindex.(𝐷.v, 3, 4)
    τyyv = D21 .* ϵ̇xxv  .+ D22 .* ϵ̇yyv .+ D23 .* ϵ̇xyv .+  D24 .* P̄t
    τxyv = D31 .* ϵ̇xxv  .+ D32 .* ϵ̇yyv .+ D33 .* ϵ̇xyv .+  D34 .* P̄t

    fy  = ∂x_inn(τyyv .- P̄t) * _Δξ .* getindex.(J.Vy, 2, 1) .+ ∂y_inn(τyyc .- inn(Pt)) * _Δη .* getindex.(J.Vy, 2, 2)
    fy += ∂x_inn(τxyv) * _Δξ .* getindex.(J.Vy, 1, 1) .+ ∂y_inn(τxyc) * _Δη .* getindex.(J.Vy, 1, 2)
    fy *= -1/(_Δξ*_Δη)

    return fy[1]
end

function ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        if type.Vy[i,j] == :in

            bcx_loc    = @inline SMatrix{4,4}(@inbounds     BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            bcy_loc    = @inline SMatrix{5,5}(@inbounds     BC.Vy[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            typex_loc  = @inline SMatrix{4,4}(@inbounds   type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            typey_loc  = @inline SMatrix{5,5}(@inbounds   type.Vy[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            ph_loc     = @inline SMatrix{2,2}(@inbounds phases.Vx[ii,jj] for ii in i-1:i, jj in j:j+1)

            Vx_loc     = @inline SMatrix{4,4}(@inbounds       V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            Vy_loc     = @inline SMatrix{5,5}(@inbounds       V.y[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            P_loc      = @inline SMatrix{3,4}(@inbounds         P[ii,jj] for ii in i-2:i,   jj in j-2:j+1)
            ΔP_loc     = @inline SMatrix{3,2}(@inbounds        ΔP.c[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τ0_loc     = @inline SMatrix{2,2}(@inbounds     τ0.Vx[ii,jj] for ii in i-1:i, jj in j:j+1    )
            D_c        = @inline SMatrix{3,2}(@inbounds       𝐷.c[ii,jj] for ii in i-2:i,   jj in j-1:j+0)
            D_v        = @inline SMatrix{2,3}(@inbounds       𝐷.v[ii,jj] for ii in i-1:i,   jj in j-1:j+1)

            J_Vy       = @inline SMatrix{1,1}(@inbounds    Jinv.Vy[ii,jj] for ii in i:i,   jj in j:j    )
            J_c        = @inline SMatrix{3,4}(@inbounds    Jinv.c[ii,jj] for ii in i-2:i,   jj in j-2:j+1)
            J_v        = @inline SMatrix{4,3}(@inbounds    Jinv.v[ii,jj] for ii in i-2:i+1, jj in j-1:j+1)

            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            Jinv_loc   = (c=J_c, v=J_v, Vy=J_Vy)
            D          = (c=D_c, v=D_v)

            R.y[i,j]   = SMomentum_y_Generic(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, D, Jinv_loc, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Vy = @MMatrix zeros(5,5)
    ∂R∂Pt = @MMatrix zeros(3,4)
    
    Vx_loc = @MMatrix zeros(4,4)
    Vy_loc = @MMatrix zeros(5,5)
    P_loc  = @MMatrix zeros(3,4)
       
    shift    = (x=2, y=1)
    K21 = K[2][1]
    K22 = K[2][2]
    K23 = K[2][3]

    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x

        if type.Vy[i,j] === :in


            bcx_loc    = @inline SMatrix{4,4}(@inbounds     BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            bcy_loc    = @inline SMatrix{5,5}(@inbounds     BC.Vy[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            typex_loc  = @inline SMatrix{4,4}(@inbounds   type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            typey_loc  = @inline SMatrix{5,5}(@inbounds   type.Vy[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            ph_loc     = @inline SMatrix{2,2}(@inbounds phases.Vx[ii,jj] for ii in i-1:i, jj in j:j+1)

            Vx_loc    .= @inline SMatrix{4,4}(@inbounds       V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            Vy_loc    .= @inline SMatrix{5,5}(@inbounds       V.y[ii,jj] for ii in i-2:i+2, jj in j-2:j+2)
            P_loc     .= @inline SMatrix{3,4}(@inbounds         P[ii,jj] for ii in i-2:i,   jj in j-2:j+1)
            ΔP_loc     = @inline SMatrix{3,2}(@inbounds        ΔP.c[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τ0_loc     = @inline SMatrix{2,2}(@inbounds     τ0.Vx[ii,jj] for ii in i-1:i, jj in j:j+1    )
            D_c        = @inline SMatrix{3,2}(@inbounds       𝐷.c[ii,jj] for ii in i-2:i,   jj in j-1:j+0)
            D_v        = @inline SMatrix{2,3}(@inbounds       𝐷.v[ii,jj] for ii in i-1:i,   jj in j-1:j+1)

            J_Vy       = @inline SMatrix{1,1}(@inbounds    Jinv.Vy[ii,jj] for ii in i:i,   jj in j:j    )
            J_c        = @inline SMatrix{3,4}(@inbounds    Jinv.c[ii,jj] for ii in i-2:i,   jj in j-2:j+1)
            J_v        = @inline SMatrix{4,3}(@inbounds    Jinv.v[ii,jj] for ii in i-2:i+1, jj in j-1:j+1)

            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            Jinv_loc   = (c=J_c, v=J_v, Vy=J_Vy)
            D          = (c=D_c, v=D_v)

            fill!(∂R∂Vx, 0.0)
            fill!(∂R∂Vy, 0.0)
            fill!(∂R∂Pt, 0.0)
            autodiff(Enzyme.Reverse, SMomentum_y_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ΔP_loc), Const(τ0_loc), Const(D), Const(Jinv_loc), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
            
            num_Vy = @inbounds num.Vy[i,j]
            bounds_Vy = num_Vy > 0

            # Vy --- Vx
            Local1 = SMatrix{4,4}(num.Vx[ii, jj] for ii in i-2:i+1, jj in j-1:j+2) .* pattern[2][1]
            for jj in axes(Local1,2), ii in axes(Local1,1)
                if (Local1[ii,jj]>0) && bounds_Vy
                    @inbounds K21[num_Vy, Local1[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vy --- Vy
            Local2 = SMatrix{5,5}(num.Vy[ii, jj] for ii in i-2:i+2, jj in j-2:j+2) .* pattern[2][2]
            for jj in axes(Local2,2), ii in axes(Local2,1)
                if (Local2[ii,jj]>0) && bounds_Vy
                    @inbounds K22[num_Vy, Local2[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vy --- Pt
            Local3 = SMatrix{3,4}(num.Pt[ii, jj] for ii in i-2:i, jj in j-2:j+1) .* pattern[2][3]
            for jj in axes(Local3,2), ii in axes(Local3,1)
                if (Local3[ii,jj]>0) && bounds_Vy
                    @inbounds K23[num_Vy, Local3[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
        end
    end 
    return nothing
end

function LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, Jinv, number, type, BC, materials, phases, nc, Δ)
    
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x 
    Vi.y .= V.y 
    Pti  .= Pt
    for i in eachindex(α)
        V.x .= Vi.x 
        V.y .= Vi.y
        Pt  .= Pti
        UpdateSolution!(V, Pt, α[i].*dx, number, type, nc)
        TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, Jinv, type, BC, materials, phases, Δ)
        ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ) 
        ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ)
        rvec[i] = @views norm(R.x[inx_Vx,iny_Vx])/length(R.x[inx_Vx,iny_Vx]) + norm(R.y[inx_Vy,iny_Vy])/length(R.y[inx_Vy,iny_Vy]) + 0*norm(R.p[inx_c,iny_c])/length(R.p[inx_c,iny_c])  
    end
    imin = argmin(rvec)
    V.x .= Vi.x 
    V.y .= Vi.y
    Pt  .= Pti
    return imin
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, Pt, ΔPt, J, type, BC, materials, phases, Δ)

    _ones = @SVector ones(4)
    _Δξ = 1 / Δ.ξ
    _Δη = 1 / Δ.η
    _Δt = 1 / Δ.t

    # Loop over centroids
    for j=2:size(ε̇.xx,2)-1, i=2:size(ε̇.xx,1)-1
        if (i==1 && j==1) || (i==size(ε̇.xx,1) && j==1) || (i==1 && j==size(ε̇.xx,2)) || (i==size(ε̇.xx,1) && j==size(ε̇.xx,2))
            # Avoid the outer corners - nothing is well defined there ;)
        else
            Vx     = @inline SMatrix{4,3}(@inbounds      V.x[ii,jj] for ii in i-1:i+2,   jj in j:j+2)
            Vy     = @inline SMatrix{3,4}(@inbounds      V.y[ii,jj] for ii in i:i+2,   jj in j-1:j+2)
            bcx    = @inline SMatrix{4,3}(@inbounds    BC.Vx[ii,jj] for ii in i-1:i+2,   jj in j:j+2)
            bcy    = @inline SMatrix{3,4}(@inbounds    BC.Vy[ii,jj] for ii in i:i+2,   jj in j-1:j+2)
            typex  = @inline SMatrix{4,3}(@inbounds  type.Vx[ii,jj] for ii in i-1:i+2,   jj in j:j+2)
            typey  = @inline SMatrix{3,4}(@inbounds  type.Vy[ii,jj] for ii in i:i+2,   jj in j-1:j+2)
            τxy0   = @inline SMatrix{2,2}(@inbounds    τ0.xy[ii,jj] for ii in i:i+1,   jj in j:j+1)

            J_c    = @inline SMatrix{1,1}(@inbounds      J.c[ii,jj] for ii in i:i,   jj in j:j  )
            J_v    = @inline SMatrix{2,2}(@inbounds      J.v[ii,jj] for ii in i:i+1,   jj in j:j+1  )

            Vx = SetBCVx1(Vx, typex, bcx, Δ)
            Vy = SetBCVy1(Vy, typey, bcy, Δ)
            V̄x = av(Vx)
            V̄y = av(Vy)

            Dxx = inn(∂x(Vx)) .* _Δξ .* getindex.(J_c, 1, 1) .+ ∂y_inn(V̄x)  .* _Δη .* getindex.(J_c, 1, 2)  # (1, 1)
            Dyy = ∂x_inn(V̄y)  .* _Δξ .* getindex.(J_c, 2, 1) .+ inn(∂y(Vy)) .* _Δη .* getindex.(J_c, 2, 2)  # (1, 1)
            Dxy = ∂x(V̄x)      .* _Δξ .* getindex.(J_v, 2, 1) .+ ∂y_inn(Vx)  .* _Δη .* getindex.(J_v, 2, 2)  # (2, 2)  
            Dyx = ∂x_inn(Vy)  .* _Δξ .* getindex.(J_v, 1, 1) .+ ∂y(V̄y)      .* _Δη .* getindex.(J_v, 1, 2)  # (2, 2)

            Dkk = Dxx .+ Dyy
            ε̇xx = @. Dxx - Dkk ./ 3
            ε̇yy = @. Dyy - Dkk ./ 3
            ε̇xy = @. (Dxy + Dyx) ./ 2
            ε̇̄xy = av(ε̇xy)
        
            # Visco-elasticity
            G    = materials.G[phases.c[i,j]]
            τ̄xy0 = av(τxy0)
            ε̇vec = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), Pt[i,j]])

            # Tangent operator used for Newton Linearisation
            jac  = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.c[i,j]), Const(Δ))
            
            # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
            @views 𝐷_ctl.c[i,j][:,1] .= jac.derivs[1][1][1]
            @views 𝐷_ctl.c[i,j][:,2] .= jac.derivs[1][2][1]
            @views 𝐷_ctl.c[i,j][:,3] .= jac.derivs[1][3][1]
            @views 𝐷_ctl.c[i,j][:,4] .= jac.derivs[1][4][1]

            # Tangent operator used for Picard Linearisation
            𝐷.c[i,j] .= diagm(2*jac.val[2] * _ones)
            𝐷.c[i,j][4,4] = 1

            # Update stress
            τ.xx[i,j]  = jac.val[1][1]
            τ.yy[i,j]  = jac.val[1][2]
            ε̇.xx[i,j]  = ε̇xx[1]
            ε̇.yy[i,j]  = ε̇yy[1]
            λ̇.c[i,j]   = jac.val[3]
            η.c[i,j]   = jac.val[2]
            ΔPt.c[i,j] = (jac.val[1][4] - Pt[i,j])
        end
    end

    # Loop over vertices
    for j=2:size(ε̇.xy,2)-1, i=2:size(ε̇.xy,1)-1
        Vx     = @inline SMatrix{3,4}(@inbounds      V.x[ii,jj] for ii in i:i+2,   jj in j+0:j+3)
        Vy     = @inline SMatrix{4,3}(@inbounds      V.y[ii,jj] for ii in i+0:i+3, jj in j:j+2  )
        bcx    = @inline SMatrix{3,4}(@inbounds    BC.Vx[ii,jj] for ii in i:i+2,   jj in j+0:j+3)
        bcy    = @inline SMatrix{4,3}(@inbounds    BC.Vy[ii,jj] for ii in i+0:i+3, jj in j:j+2  )
        typex  = @inline SMatrix{3,4}(@inbounds  type.Vx[ii,jj] for ii in i:i+2,   jj in j+0:j+3)
        typey  = @inline SMatrix{4,3}(@inbounds  type.Vy[ii,jj] for ii in i+0:i+3, jj in j:j+2  )
        τxx0   = @inline SMatrix{2,2}(@inbounds    τ0.xx[ii,jj] for ii in i:i+1,   jj in j:j+1)
        τyy0   = @inline SMatrix{2,2}(@inbounds    τ0.yy[ii,jj] for ii in i:i+1,   jj in j:j+1)
        P      = @inline SMatrix{2,2}(@inbounds       Pt[ii,jj] for ii in i-1:i,   jj in j-1:j)

        J_c    = @inline SMatrix{2,2}(@inbounds      J.c[ii,jj] for ii in i-1:i, jj in j-1:j  )
        J_v    = @inline SMatrix{1,1}(@inbounds      J.v[ii,jj] for ii in i:i,   jj in j:j  )

        Vx     = SetBCVx1(Vx, typex, bcx, Δ)
        Vy     = SetBCVy1(Vy, typey, bcy, Δ)
        V̄x     = av(Vx)
        V̄y     = av(Vy)

        Dxx    = ∂x_inn(Vx)  .* _Δξ .* getindex.(J_c, 1, 1) .+ ∂y(V̄x)      .* _Δη .* getindex.(J_c, 1, 2)
        Dyy    = ∂x(V̄y)      .* _Δξ .* getindex.(J_c, 2, 1) .+ ∂y_inn(Vy)  .* _Δη .* getindex.(J_c, 2, 2)
        Dxy    = ∂x_inn(V̄x)  .* _Δξ .* getindex.(J_v, 2, 1) .+ inn(∂y(Vx)) .* _Δη .* getindex.(J_v, 2, 2)
        Dyx    = inn(∂x(Vy)) .* _Δξ .* getindex.(J_v, 1, 2) .+ ∂y_inn(V̄y)  .* _Δη .* getindex.(J_v, 1, 2)

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
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t), P̄[1]])
        
        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.v[i,j]), Const(Δ))

        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        @views 𝐷_ctl.v[i,j][:,1] .= jac.derivs[1][1][1]
        @views 𝐷_ctl.v[i,j][:,2] .= jac.derivs[1][2][1]
        @views 𝐷_ctl.v[i,j][:,3] .= jac.derivs[1][3][1]
        @views 𝐷_ctl.v[i,j][:,4] .= jac.derivs[1][4][1]

        # Tangent operator used for Picard Linearisation
        𝐷.v[i,j] .= diagm(2*jac.val[2] * _ones)
        𝐷.v[i,j][4,4] = 1

        # Update stress
        τ.xy[i,j] = jac.val[1][3]
        ε̇.xy[i,j] = ε̇xy[1]
        λ̇.v[i,j]  = jac.val[3]
        η.v[i,j]  = jac.val[2]
        # τ.xy[i+1,j+1] = 2*jac.val[2]*(ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t))
    end
end