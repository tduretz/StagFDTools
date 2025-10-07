

#----------------------------------------------
#               Grid meshing
#----------------------------------------------
# function to define grid spacing, x and y vertices and x and y cells' center
# outputs initialisation is fone outside of this function
function mesh_grid_normal_distrib(L, μ, σ, nc, Δ, xv, yv, xc, yc)

    inflimit = (x=-L.x/2, y=-L.y/2)
    suplimit = (x=L.x/2, y=L.y/2)

    # nodes
    xv_in = normal_linspace_interval(inflimit.x, suplimit.x, μ.x, σ.x, nc.x+1)
    yv_in = normal_linspace_interval(inflimit.y, suplimit.y, μ.y, σ.y, nc.y+1)

    # grid spacing inside cells
    Δ.x = diff(xv_in)
    Δ.y = diff(yv_in)
    #Δ.x[2:end-1] = diff(xv_in)
    #Δ.y[2:end-1] = diff(yv_in)
    # gris spacing ghost cells
    #=Δ.x[1]   = Δ.x[2]
    Δ.x[end] = Δ.x[end-1]
    Δ.y[1]   = Δ.y[2]
    Δ.y[end] = Δ.y[end-1]=#

    # inside vertices
    xv .= xv_in
    yv .= yv_in
    #xv[2:end-1] .= xv_in
    #yv[2:end-1] .= yv_in
    # ghost vertices
    #=xv[1]   = xv[2] - Δ.x[1]
    xv[end] = xv[end-1] + Δ.x[end]
    yv[1]   = yv[2] - Δ.y[1]
    yv[end] = yv[end-1] + Δ.y[end]=#

    # cells' centers
    xc = 0.5*(xv[2:end] + xv[1:end-1])
    yc = 0.5*(yv[2:end] + yv[1:end-1])

end

# function returning a vector which vertices follow a normal distribution
function normal_linspace_interval(inflimit::Float64, suplimit::Float64, μ::Float64, σ::Float64, ncells::Int)
    dist = Normal(μ, σ)
    inf_cdf = cdf(dist, inflimit)
    sup_cdf = cdf(dist, suplimit)
    vec = range(inf_cdf, sup_cdf; length=ncells)
    return quantile.(dist, vec)
end



#----------------------------------------------
#               Tangent operator
#----------------------------------------------
function TangentOperator_var!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, Pt, ΔPt, type, BC, materials, phases, Δ)

    _ones = @SVector ones(4)

    # Loop over centroids
    for j=1:size(ε̇.xx,2)-0, i=1:size(ε̇.xx,1)-0
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
            # ajout car pour grille variable les deltas ont des valeurs différentes
            Δcx = @SVector [ Δ.x[i], Δ.x[i+1], Δ.x[i+2] ]
            Δcy = @SVector [ Δ.y[j], Δ.y[j+1], Δ.y[j+2] ]

            Vx = SetBCVx1_var(Vx, typex, bcx, Δ)
            Vy = SetBCVy1_var(Vy, typey, bcy, Δ)

            # ajout car pour grille variable les deltas sont des vecteurs
            #display(size(Δ.x))
            #display(size(Vx))
            #display(size(Δ.y))
            #display(size(Vy))
            #Dxx = @SVector{2,3}( 0, 0 ; 0, 0)
            #Dyy = @SVector{3,2}(zeros(3,2))
            #Dxy = @SVector{2,2}(zeros(2,2))
            #Dyx = @SVector{2,2}(zeros(2,2))

            Dxx = zeros(3)
            Dyy = zeros(3)
            display(size(∂y(Vx) ./ Δcy))
            display(size(∂x(Vy) ./ Δcx))
            Dxy = zeros(3)
            Dyx = zeros(3)
            Dxx .= ∂x_inn(Vx) ./ Δcx
            Dyy .= ∂y_inn(Vy) ./ Δcy
            Dxy .= ∂y(Vx) ./ Δcy
            Dyx .= ∂x(Vy) ./ Δcx
            
            Dkk = Dxx .+ Dyy
            ε̇xx = @. Dxx - Dkk ./ 3
            ε̇yy = @. Dyy - Dkk ./ 3
            ε̇xy = @. (Dxy + Dyx) ./ 2
            ε̇̄xy = av(ε̇xy)
        
            # Visco-elasticity
            G     = materials.G[phases.c[i,j]]
            τ̄xy0  = av(τxy0)
            ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), Pt[i,j]])

            # Tangent operator used for Newton Linearisation
            jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.c[i,j]), Const(Δ.x), Const(Δ.y))
            
            # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
            @views 𝐷_ctl.c[i,j][:,1] .= jac.derivs[1][1][1]
            @views 𝐷_ctl.c[i,j][:,2] .= jac.derivs[1][2][1]
            @views 𝐷_ctl.c[i,j][:,3] .= jac.derivs[1][3][1]
            @views 𝐷_ctl.c[i,j][:,4] .= jac.derivs[1][4][1]

            # Tangent operator used for Picard Linearisation
            𝐷.c[i,j] .= diagm(2*jac.val[2] * _ones)
            𝐷.c[i,j][4,4] = 1

            # Update stress
            τ.xx[i,j] = jac.val[1][1]
            τ.yy[i,j] = jac.val[1][2]
            ε̇.xx[i,j] = ε̇xx[1]
            ε̇.yy[i,j] = ε̇yy[1]
            λ̇.c[i,j]  = jac.val[3]
            η.c[i,j]  = jac.val[2]
            ΔPt[i,j]  = (jac.val[1][4] - Pt[i,j])
        end
    end

    # Loop over vertices
    for j=1:size(ε̇.xy,2)-2, i=1:size(ε̇.xy,1)-2
        Vx     = SMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        Vy     = SMatrix{2,3}(      V.y[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        bcx    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        bcy    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        typex  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        typey  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        τxx0   = SMatrix{2,2}(    τ0.xx[ii,jj] for ii in i:i+1,   jj in j:j+1)
        τyy0   = SMatrix{2,2}(    τ0.yy[ii,jj] for ii in i:i+1,   jj in j:j+1)
        P      = SMatrix{2,2}(       Pt[ii,jj] for ii in i:i+1,   jj in j:j+1)

        Vx     = SetBCVx1_var(Vx, typex, bcx, Δ)
        Vy     = SetBCVy1_var(Vy, typey, bcy, Δ)
    
        Dxx    .= ∂x(Vx) ./ Δ.x
        Dyy    .= ∂y(Vy) ./ Δ.y
        Dxy    .= ∂y_inn(Vx) ./ Δ.y
        Dyx    .= ∂x_inn(Vy) ./ Δ.x

        Dkk   = @. Dxx + Dyy
        ε̇xx   = @. Dxx - Dkk / 3
        ε̇yy   = @. Dyy - Dkk / 3
        ε̇xy   = @. (Dxy + Dyx) /2
        ε̇̄xx   = av(ε̇xx)
        ε̇̄yy   = av(ε̇yy)
        
        # Visco-elasticity
        G     = materials.G[phases.v[i+1,j+1]]
        τ̄xx0  = av(τxx0)
        τ̄yy0  = av(τyy0)
        P̄     = av(   P)
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t), P̄[1]])
        
        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.v[i+1,j+1]), Const(Δ.x), Const(Δ.y))

        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        @views 𝐷_ctl.v[i+1,j+1][:,1] .= jac.derivs[1][1][1]
        @views 𝐷_ctl.v[i+1,j+1][:,2] .= jac.derivs[1][2][1]
        @views 𝐷_ctl.v[i+1,j+1][:,3] .= jac.derivs[1][3][1]
        @views 𝐷_ctl.v[i+1,j+1][:,4] .= jac.derivs[1][4][1]

        # Tangent operator used for Picard Linearisation
        𝐷.v[i+1,j+1] .= diagm(2*jac.val[2] * _ones)
        𝐷.v[i+1,j+1][4,4] = 1

        # Update stress
        τ.xy[i+1,j+1] = jac.val[1][3]
        ε̇.xy[i+1,j+1] = ε̇xy[1]
        λ̇.v[i+1,j+1]  = jac.val[3]
        η.v[i+1,j+1]  = jac.val[2]
        # τ.xy[i+1,j+1] = 2*jac.val[2]*(ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t))

    end
end


function SetBCVx1_var(Vx, typex, bcx, Δ)

    MVx = MMatrix(Vx)
    # N/S
    for ii in axes(typex, 1)
        if typex[ii,1] == :Dirichlet_tangent
            MVx[ii,1] = fma(2, bcx[ii,1], -Vx[ii,2])
        elseif typex[ii,1] == :Neumann_tangent
            MVx[ii,1] = fma(Δ.y[ii], bcx[ii,1], Vx[ii,2])
        end

        if typex[ii,end] == :Dirichlet_tangent
            MVx[ii,end] = fma(2, bcx[ii,end], -Vx[ii,end-1])
        elseif typex[ii,end] == :Neumann_tangent
            MVx[ii,end] = fma(Δ.y[ii], bcx[ii,end], Vx[ii,end-1])
        end
    end
    # E/W
    for jj in axes(typex, 2)
        if typex[1,jj] == :Neumann_normal
            MVx[1,jj] = fma(2, Δ.x[jj]*bcx[1,jj], Vx[2,jj])
        end
        if typex[end,jj] == :Neumann_normal
            MVx[end,jj] = fma(2,-Δ.x[jj]*bcx[end,jj], Vx[end-1,jj])
        end
    end
    return SMatrix(MVx)
end


function SetBCVy1_var(Vy, typey, bcy, Δ)
    MVy = MMatrix(Vy)
    # E/W
    for jj in axes(typey, 2)
        if typey[1,jj] == :Dirichlet_tangent
            MVy[1,jj] = fma(2, bcy[1,jj], -Vy[2,jj])
        elseif typey[1,jj] == :Neumann_tangent
            MVy[1,jj] = fma(Δ.y[jj], bcy[1,jj], Vy[2,jj])                       # ERREUR ICI C'EST DELTA X ????
        end

        if typey[end,jj] == :Dirichlet_tangent
            MVy[end,jj] = fma(2, bcy[end,jj], -Vy[end-1,jj])
        elseif typey[end,jj] == :Neumann_tangent
            MVy[end,jj] = fma(Δ.y[jj], bcy[end,jj], Vy[end-1,jj])               # ERREUR ICI C'EST DELTA X ????
        end
    end
    # N/S
    for ii in axes(typey, 1)
        if typey[ii,1] == :Neumann_normal
            MVy[ii,1] = fma(2, Δ.y[ii]*bcy[ii,1], Vy[ii,2])
        end
        if typey[ii,end] == :Neumann_normal
            MVy[ii,end] = fma(2,-Δ.y[ii]*bcy[ii,end], Vy[ii,end-1])
        end
    end
    return SMatrix(MVy)
end

#----------------------------------------------
#               2D Continuity
#----------------------------------------------
function ResidualContinuity2D_var!(R, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ) 
                
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
            R.p[i,j]   = Continuity_var(Vx_loc, Vy_loc, P[i,j], P0[i,j], D, Jinv_c, phases.c[i,j], materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function Continuity_var(Vx, Vy, Pt, Pt0, D, phase, materials, type_loc, bcv_loc, Δ)
    invΔx .= 1 ./ Δ.x
    invΔy .= 1 ./ Δ.y
    invΔt .= 1 ./ Δ.t
    β     = materials.β[phase]
    η     = materials.β[phase]
    comp  = materials.compressible
    f     .= ((Vx[2,2] - Vx[1,2]) .* invΔx .+ (Vy[2,2] - Vy[2,1]) .* invΔy) .+ comp * β * (Pt[1] - Pt0) .* invΔt #+ 1/(1000*η)*Pt[1]
    f    .*= max(invΔx, invΔy)
    return f
end




#----------------------------------------------
#               Momentum
#----------------------------------------------
function ResidualMomentum2D_x_var!(R, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ) 
                
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


function SMomentum_x_Generic_var(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
    #invΔx, invΔy = 1 / Δ.x, 1 / Δ.y
    invΔx .= 1 ./Δ.x
    invΔy .= 1 ./Δ.y

    # BC
    Vx = SetBCVx1_var(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1_var(Vy_loc, type.y, bcv.y, Δ)

    # Velocity gradient
    Dxx .= ∂x(Vx) .* invΔx
    Dyy .= ∂y_inn(Vy) .* invΔy
    Dxy .= ∂y(Vx) .* invΔy
    Dyx .= ∂x_inn(Vy) .* invΔx

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
    Ptc  = SVector{2, Float64}( @. Pt[:,2] + comp * ΔP[:] )

    # Stress
    τxx = @MVector zeros(2)
    τxy = @MVector zeros(2)
    for i=1:2
        τxx[i] = (𝐷.c[i][1,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][1,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][1,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][1,4] - (𝐷.c[i][4,4] - 1)) * Pt[i,2]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                       * P̄t[i]
    end

    # Residual
    fx  .= ( τxx[2]  - τxx[1] ) .* invΔx
    fx .+= ( τxy[2]  - τxy[1] ) .* invΔy
    fx .-= ( Ptc[2]  - Ptc[1] ) .* invΔx
    fx .*= -1 .* Δ.x .* Δ.y

    return fx
end


function ResidualMomentum2D_y_var!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)                 
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
            ΔP_loc     = SMatrix{1,2}(       ΔP[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
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

            R.y[i,j]   = SMomentum_y_Generic_var(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function SMomentum_y_Generic_var(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
    #invΔx, invΔy = 1 / Δ.x, 1 / Δ.y
    invΔx .= 1 ./Δ.x
    invΔy .= 1 ./Δ.y

    # BC
    Vx = SetBCVx1_var(Vx_loc, type.x, bcv.x, Δ)
    Vy = SetBCVy1_var(Vy_loc, type.y, bcv.y, Δ)

    # Velocity gradient
    Dxx .= ∂x_inn(Vx) .* invΔx
    Dyy .= ∂y(Vy) .* invΔy
    Dxy .= ∂y_inn(Vx) .* invΔy
    Dyx .= ∂x(Vy) .* invΔx

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
    Ptc  = SVector{2, Float64}( @. Pt[2,:] + comp * ΔP[:] )

    # Stress
    τyy = @MVector zeros(2)
    τxy = @MVector zeros(2)
    for i=1:2
        τyy[i] = (𝐷.c[i][2,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][2,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][2,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][2,4] - (𝐷.c[i][4,4] - 1.)) * Pt[2,i]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                        * P̄t[i]
    end

    # Residual
    fy  .= ( τyy[2]  -  τyy[1] ) .* invΔy
    fy .+= ( τxy[2]  -  τxy[1] ) .* invΔx
    fy .-= ( Ptc[2]  -  Ptc[1])  .* invΔy
    fy .*= -1 .* Δ.x .* Δ.y
    
    return fy
end





#----------------------------------------------
#               Assembling
#----------------------------------------------

#=function AssembleContinuity2D_var!(K, V, P, Pt0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
                
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

        # vecteur dx
        Δxv = @SVector [ Δ.x[i], Δ.x[i+1], Δ.x[j+2] ]
        # vecteur dy
        Δyv = @SVector [ Δ.y[j], Δ.y[j+1], Δ.y[j+2] ]

        autodiff(Enzyme.Reverse, Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂P), Const(Pt0[i,j]), Const(D), Const(phases.c[i,j]), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δxv), Const(Δyv))

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
        if num.Pt[i,j]>0
            K[3][3][num.Pt[i,j], num.Pt[i,j]] = ∂R∂P[1,1]
        end
    end
    return nothing
end=#


function AssembleContinuity2D_var!(K, V, P, Pt0, ΔP, τ0, 𝐷, Jinv, phases, materials, num, pattern, type, BC, nc, Δ) 
                
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

        # vecteur dx
        Δxv = @SVector [ Δ.x[i-1], Δ.x[i], Δ.x[i+1], Δ.x[j+2] ]
        # vecteur dy
        Δyv = @SVector [ Δ.y[j-2], Δ.y[j-1], Δ.y[j], Δ.y[j+1] ]

        autodiff(Enzyme.Reverse, Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂P), Const(Pt0[i,j]), Const(D), Const(Jinv_c), Const(phases.c[i,j]), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δxv), Const(Δyv))

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


function AssembleMomentum2D_x_var!(K, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, num, pattern, type, BC, nc, Δ) 

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
            
            # vecteur dx
            Δxv = @SVector [ Δ.x[i-1], Δ.x[i], Δ.x[i+1], Δ.x[j+2] ]
            # vecteur dy
            Δyv = @SVector [ Δ.y[j-2], Δ.y[j-1], Δ.y[j], Δ.y[j+1] ]

            autodiff(Enzyme.Reverse, SMomentum_x_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ΔP_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δxv), Const(Δyv))
            
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


function AssembleMomentum2D_y_var!(K, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, num, pattern, type, BC, nc, Δ) 
    
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
            
            # vecteur dx
            Δxv = @SVector [ Δ.x[i-1], Δ.x[i], Δ.x[i+1], Δ.x[j+2] ]
            # vecteur dy
            Δyv = @SVector [ Δ.y[j-2], Δ.y[j-1], Δ.y[j], Δ.y[j+1] ]

            autodiff(Enzyme.Reverse, SMomentum_y_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ΔP_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δxv), Const(Δyv))
            
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