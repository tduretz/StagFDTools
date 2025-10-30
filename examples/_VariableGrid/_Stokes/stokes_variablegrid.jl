
using Distributions

function normal_linspace_interval(inflimit::Float64, suplimit::Float64, μ::Float64, σ::Float64, ncells::Int)
    dist = Normal(μ, σ)
    inf_cdf = cdf(dist, inflimit)
    sup_cdf = cdf(dist, suplimit)
    vec = range(inf_cdf, sup_cdf; length=ncells)
    return quantile.(dist, vec)
end


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

            Δx_Vx     = SVector{2}(Δ.x[ii] for ii in i:i+1)
            Δy_Vx     = SVector{3}(Δ.y[jj] for jj in j:j+2)
            Δx_Vy     = SVector{3}(Δ.x[ii] for ii in i:i+2)
            Δy_Vy     = SVector{2}(Δ.y[jj] for jj in j:j+1)
    
            Vx     = SetBCVx1_var(Vx, typex, bcx, Δx_Vx, Δy_Vx)
            Vy     = SetBCVy1_var(Vy, typey, bcy, Δx_Vy, Δy_Vy)

            Dxx = ∂x_inn(Vx) / ((Δx_Vx[1]+Δx_Vx[2])/2)
            Dyy = ∂y_inn(Vy) / ((Δy_Vy[1]+Δy_Vy[2])/2)
            Dxy = ∂y(Vx) / ((Δy_Vx[1]+Δy_Vx[2])/2)
            Dyx = ∂x(Vy) / ((Δx_Vy[1]+Δx_Vy[2])/2)

            Dkk = Dxx .+ Dyy
            ε̇xx = @. Dxx - Dkk ./ 3
            ε̇yy = @. Dyy - Dkk ./ 3
            ε̇xy = @. (Dxy + Dyx) ./ 2
            ε̇̄xy = av(ε̇xy)
        
            # Visco-elasticity
            G     = materials.G[phases.c[i,j]]
            τ̄xy0  = av(τxy0)
            ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t[1]), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t[1]), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t[1]), Pt[i,j]])
            # Tangent operator used for Newton Linearisation
            Δt =  Δ.t[1]
            jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector_var!, ε̇vec, Const(materials), Const(phases.c[i,j]), Const(Δt))
            
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
    for j=2:size(ε̇.xy,2)-2, i=2:size(ε̇.xy,1)-2
        Vx     = SMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        Vy     = SMatrix{2,3}(      V.y[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        bcx    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        bcy    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        typex  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        typey  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        τxx0   = SMatrix{2,2}(    τ0.xx[ii,jj] for ii in i:i+1,   jj in j:j+1)
        τyy0   = SMatrix{2,2}(    τ0.yy[ii,jj] for ii in i:i+1,   jj in j:j+1)
        P      = SMatrix{2,2}(       Pt[ii,jj] for ii in i:i+1,   jj in j:j+1)

        Δx_Vx     = SVector{3}(Δ.x[ii] for ii in i:i+2)
        Δy_Vx     = SVector{2}(Δ.y[jj] for jj in j+1:j+2)
        Δx_Vy     = SVector{2}(Δ.x[ii] for ii in i+1:i+2)
        Δy_Vy     = SVector{3}(Δ.y[jj] for jj in j:j+2)

        Vx     = SetBCVx1_var(Vx, typex, bcx, Δx_Vx, Δy_Vx)
        Vy     = SetBCVy1_var(Vy, typey, bcy, Δx_Vy, Δy_Vy)
    
        Dxx    = ∂x(Vx) /  ((Δx_Vx[2]+Δx_Vx[3])/2)
        Dyy    = ∂y(Vy) /  ((Δy_Vy[2]+Δy_Vy[3])/2)
        Dxy    = ∂y_inn(Vx) / ((Δy_Vx[1]+Δy_Vx[2])/2)
        Dyx    = ∂x_inn(Vy) / ((Δx_Vy[1]+Δx_Vy[2])/2)

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

        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t[1]), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t[1]), ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t[1]), P̄[1]])
        
        # Tangent operator used for Newton Linearisation
        Δt = Δ.t[1]
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector_var!, ε̇vec, Const(materials), Const(phases.v[i+1,j+1]), Const(Δt))

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



function SetBCVx1_var(Vx, typex, bcx, Δx, Δy)

    MVx = MMatrix(Vx)
    # Vx follows vertices axes for North and South orientations
    # N/S
    for ii in axes(typex, 1)
        if typex[ii,1] == :Dirichlet_tangent
            MVx[ii,1] = fma(2, bcx[ii,1], -Vx[ii,2])
        elseif typex[ii,1] == :Neumann_tangent
            coeff = ((Δy[1]+Δy[2])/2)
            #coeff = Δy[1]
            MVx[ii,1] = fma(coeff, bcx[ii,1], Vx[ii,2])
        end

        if typex[ii,end] == :Dirichlet_tangent
            MVx[ii,end] = fma(2, bcx[ii,end], -Vx[ii,end-1])
        elseif typex[ii,end] == :Neumann_tangent
            coeff = ((Δy[end]+Δy[end-1])/2)
            #coeff = Δy[end]
            MVx[ii,end] = fma(coeff, bcx[ii,end], Vx[ii,end-1])
        end
    end


    
    # E/W
    # Vx follows centers axes for East and West orientations
    for jj in axes(typex, 2)
        if typex[1,jj] == :Neumann_normal
            MVx[1,jj] = fma(2, Δx[1]*bcx[1,jj], Vx[2,jj])
        end
        if typex[end,jj] == :Neumann_normal
            MVx[end,jj] = fma(2,-Δx[end]*bcx[end,jj], Vx[end-1,jj])
        end
    end
    return SMatrix(MVx)
end


function SetBCVy1_var(Vy, typey, bcy, Δx, Δy)
    MVy = MMatrix(Vy)
    # E/W
    # Vy follows vertices axes for East and West orientations
    for jj in axes(typey, 2)
        if typey[1,jj] == :Dirichlet_tangent
            MVy[1,jj] = fma(2, bcy[1,jj], -Vy[2,jj])
        elseif typey[1,jj] == :Neumann_tangent
            coeff = ((Δx[1]+Δx[2])/2)
            #coeff = Δx[1]
            MVy[1,jj] = fma(coeff, bcy[1,jj], Vy[2,jj])
        end

        if typey[end,jj] == :Dirichlet_tangent
            MVy[end,jj] = fma(2, bcy[end,jj], -Vy[end-1,jj])
        elseif typey[end,jj] == :Neumann_tangent
            coeff = ((Δx[end]+Δx[end-1])/2)
            #coeff = Δx[end]
            MVy[end,jj] = fma(coeff, bcy[end,jj], Vy[end-1,jj])
        end
    end
    # N/S
    # Vy follows centers axes for North and South orientations
    for ii in axes(typey, 1)
        if typey[ii,1] == :Neumann_normal
            MVy[ii,1] = fma(2, Δy[1]*bcy[ii,1], Vy[ii,2])
        end
        if typey[ii,end] == :Neumann_normal
            MVy[ii,end] = fma(2,-Δy[end]*bcy[ii,end], Vy[ii,end-1])
        end
    end
    return SMatrix(MVy) 
end


function ResidualContinuity2D_var!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    # loop on centroids
    for j in 2:size(R.p,2)-1, i in 2:size(R.p,1)-1
        if type.Pt[i,j] !== :constant 
            #Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
            #Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
            Vx_loc     = SMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
            Vy_loc     = SMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)

            Δx_Vx_loc     = SVector{3}(Δ.x[ii] for ii in i:i+2)
            Δy_Vx_loc     = SVector{2}(Δ.y[jj] for jj in j:j+1)
            Δx_Vy_loc     = SVector{2}(Δ.x[ii] for ii in i:i+1)
            Δy_Vy_loc     = SVector{3}(Δ.y[jj] for jj in j:j+2)

            #=Δx_Vx_loc     = SVector{2}(Δ.x[ii] for ii in i:i+1)
            Δy_Vx_loc     = SVector{3}(Δ.y[jj] for jj in j:j+2)
            Δy_Vy_loc     = SVector{2}(Δ.y[jj] for jj in j:j+1)
            Δx_Vy_loc     = SVector{3}(Δ.x[ii] for ii in i:i+2)=#

            Δt_loc = Δ.t[1]

            bcv_loc    = (;)
            type_loc   = (;)
            D          = (;)
            R.p[i,j]   = Continuity_var(Vx_loc, Vy_loc, P[i,j], P0[i,j], D, phases.c[i,j], materials, type_loc, bcv_loc, Δx_Vx_loc, Δy_Vx_loc, Δy_Vy_loc, Δx_Vy_loc, Δt_loc)
        end
    end
    return nothing
end


function Continuity_var(Vx, Vy, Pt, Pt0, D, phase, materials, type_loc, bcv_loc, Δx_Vx, Δy_Vx, Δy_Vy, Δx_Vy, Δt)
    invΔx = 1 / Δx_Vx[1]
    invΔy = 1 / Δx_Vy[1]
    invArea = invΔx * invΔy
    invΔt = 1 / Δt
    β     = materials.β[phase]
    η     = materials.β[phase]
    comp  = materials.compressible

    f     = ((Vx[2,2] - Vx[1,2]) * invΔy + (Vy[2,2] - Vy[2,1]) * invΔx) + comp * β * (Pt[1] - Pt0) * invΔt #+ 1/(1000*η)*Pt[1]
    f    *= invArea
    
    #=invΔx_Vx = 1 / Δx_Vx[1] #2 / (Δx_Vx[1]+Δx_Vx[2])
    invΔy_Vy = 1 / Δy_Vy[1] #2 / (Δy_Vy[1]+Δy_Vy[2])
    #invΔy_Vx = 2 / (Δy_Vx[1]+Δy_Vx[2])
    #invΔx_Vy = 2 / (Δx_Vy[1]+Δx_Vy[2])
    invAreaVx = 2 / ( (Δy_Vx[1]+Δy_Vx[2]) * Δx_Vx[1] ) #1 / ( Δx_Vx[2] * Δy_Vy[2] ) # 2 / ( (Δy_Vx[1]+Δy_Vx[2]) * Δx_Vx[1] )
    invAreaVy = 2 / ( (Δx_Vy[1]+Δx_Vy[2]) * Δy_Vy[1] ) #1 / ( Δx_Vx[2] * Δy_Vy[2] ) #2 / ( (Δx_Vy[1]+Δx_Vy[2]) * Δy_Vy[1] )
    invArea = 1 / (Δx_Vx[2] * Δy_Vy[2])
    invΔt = 1 / Δt
    β     = materials.β[phase]
    η     = materials.β[phase]
    comp  = materials.compressible

    f     = (Vx[2,2] - Vx[1,2]) * invΔx_Vx + (Vy[2,2] - Vy[2,1]) * invΔy_Vy + comp * β * (Pt[1] - Pt0) * invΔt
    #f     = (Vx[2,2] - Vx[1,2]) * Δx_Vx[1] * ((Δy_Vx[1]+Δy_Vx[2]) / 2) + (Vy[2,2] - Vy[2,1]) * Δy_Vy[1] * ((Δx_Vy[1]+Δx_Vy[2]) / 2) + comp * β * (Pt[1] - Pt0) * invΔt #+ 1/(1000*η)*Pt[1]
    #f     = (Vx[2,2] - Vx[1,2]) * Δx_Vx[1] * invAreaVx + (Vy[2,2] - Vy[2,1]) * Δy_Vy[1] * invAreaVy + comp * β * (Pt[1] - Pt0) * invΔt #+ 1/(1000*η)*Pt[1]
    #f     = (((Vx[2,2] - Vx[1,2]) * invΔx_Vx) + ((Vy[2,2] - Vy[2,1]) * invΔy_Vy)) + comp * β * (Pt[1] - Pt0) * invΔt #+ 1/(1000*η)*Pt[1]
    #f     = ( (Vx[2,2] - Vx[1,2]) * invΔx_Vx * ( 1 / ( ( (Δy_Vx[1]+Δy_Vx[2]) / 2 ) * Δx_Vx[1] ) ) + (Vy[2,2] - Vy[2,1]) * invΔy_Vy * ( 1 / ( ( (Δx_Vy[1]+Δx_Vy[2]) / 2 ) * Δy_Vy[1]) ) ) + comp * β * (Pt[1] - Pt0) * invΔt #+ 1/(1000*η)*Pt[1]
    f    *= 1 / Δx_Vx[1] # * Δy_Vy[1])#max(invΔx_Vx, invΔy_Vy)=#
    return f
end



function ResidualMomentum2D_x_var!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                
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
            ΔP_loc     = SMatrix{2,1}(     ΔP.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
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

            Δx         = SVector{2}(        Δ.x[ii] for ii in i-1:i)
            Δy         = SVector{1}(        Δ.y[jj] for jj in j:j)
    
            Δx_Vx_loc     = SVector{3}(Δ.x[ii] for ii in i-1:i+1)
            Δy_Vx_loc     = SVector{3}(Δ.y[jj] for jj in j-1:j+1)
            Δx_Vy_loc     = SVector{4}(Δ.x[ii] for ii in i-1:i+2)
            Δy_Vy_loc     = SVector{4}(Δ.y[jj] for jj in j-2:j+1)

            R.x[i,j]   = SMomentum_x_Generic_var(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δx_Vx_loc, Δy_Vx_loc, Δx_Vy_loc, Δy_Vy_loc, Δ.t[1])
        end
    end
    return nothing
end


function SMomentum_x_Generic_var(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, phases, materials, type, bcv, Δx_Vx, Δy_Vx, Δx_Vy, Δy_Vy, Δt)

    # BC
    Vx = SetBCVx1_var(Vx_loc, type.x, bcv.x, Δx_Vx, Δy_Vx)
    Vy = SetBCVy1_var(Vy_loc, type.y, bcv.y, Δx_Vy, Δy_Vy)

    # Velocity gradient
    Dxx = ∂x(Vx) * (1/Δx_Vx[2])
    Dyy = ∂y_inn(Vy) * (1/Δy_Vy[3])
    Dxy = ∂y(Vx) * (2/(Δy_Vx[3]+Δy_Vx[2]))
    Dyx = ∂x_inn(Vy) * (2/(Δx_Vy[3]+Δx_Vy[2]))

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
    tmpc = @. inv(2 * Gc * Δt)
    tmpv = @. inv(2 * Gv * Δt)
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
    fx  = ( τxx[2]  - τxx[1] ) * (1/Δx_Vx[2])
    fx += ( τxy[2]  - τxy[1] ) * (2/(Δy_Vx[3]+Δy_Vx[2]))
    fx -= ( Ptc[2]  - Ptc[1] ) * (1/Δx_Vx[2])
    fx *= -1 * Δx_Vx[2] * ((Δy_Vx[3]+Δy_Vx[2])/2)

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
            ΔP_loc     = SMatrix{1,2}(     ΔP.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
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

            Δx_Vx_loc     = SVector{4}(Δ.x[ii] for ii in i-2:i+1)
            Δy_Vx_loc     = SVector{4}(Δ.y[jj] for jj in j-1:j+2)
            Δx_Vy_loc     = SVector{3}(Δ.x[ii] for ii in i-1:i+1)
            Δy_Vy_loc     = SVector{3}(Δ.y[jj] for jj in j-1:j+1)

            R.y[i,j]   = SMomentum_y_Generic_var(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δx_Vx_loc, Δy_Vx_loc, Δx_Vy_loc, Δy_Vy_loc, Δ.t[1])
        end
    end
    return nothing
end


function SMomentum_y_Generic_var(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, phases, materials, type, bcv, Δx_Vx, Δy_Vx, Δx_Vy, Δy_Vy, Δt)
    
    # BC
    Vx = SetBCVx1_var(Vx_loc, type.x, bcv.x, Δx_Vx, Δy_Vx)
    Vy = SetBCVy1_var(Vy_loc, type.y, bcv.y, Δx_Vy, Δy_Vy)

    # Velocity gradient
    Dxx = ∂x_inn(Vx) * (1/Δx_Vx[3])
    Dyy = ∂y(Vy) * (1/Δy_Vy[2])
    Dxy = ∂y_inn(Vx) * (2/(Δy_Vx[3]+Δy_Vx[2]))
    Dyx = ∂x(Vy) * (2/(Δx_Vy[3]+Δx_Vy[2]))

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
    tmpc = (2*Gc.*Δt)
    tmpv = (2*Gv.*Δt)
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
    fy  = ( τyy[2]  -  τyy[1] ) * (1/Δy_Vy[2])
    fy += ( τxy[2]  -  τxy[1] ) * (2/(Δx_Vy[3]+Δx_Vy[2]))
    fy -= ( Ptc[2]  -  Ptc[1])  * (1/Δy_Vy[2])
    fy *= -1 * ((Δx_Vy[3]+Δx_Vy[2])/2) * Δy_Vy[2]
    
    return fy
end



#----------------------------------
#           Assembling
#----------------------------------

function AssembleContinuity2D_var!(K, V, P, Pt0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 

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
        
        Δx_Vx_loc     = SVector{2}(Δ.x[ii] for ii in i:i+1)
        Δy_Vx_loc     = SVector{3}(Δ.y[jj] for jj in j:j+2)
        Δy_Vy_loc     = SVector{2}(Δ.y[jj] for jj in j:j+1)
        Δx_Vy_loc     = SVector{3}(Δ.x[ii] for ii in i:i+2)

        Δt_loc        = Δ.t[1]

        autodiff(Enzyme.Reverse, Continuity_var, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂P), Const(Pt0[i,j]), Const(D), Const(phases.c[i,j]), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δx_Vx_loc), Const(Δy_Vx_loc), Const(Δy_Vy_loc), Const(Δx_Vy_loc), Const(Δt_loc))

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




function AssembleMomentum2D_x_var!(K, V, P, P0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 

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
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0) 
            
            Vx_loc    .= SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc    .= SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            P_loc     .= SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            ΔP_loc    .= SMatrix{2,1}(     ΔP.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)

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

            Δx_Vx_loc     = SVector{3}(Δ.x[ii] for ii in i-1:i+1)
            Δy_Vx_loc     = SVector{3}(Δ.y[jj] for jj in j-1:j+1)
            Δx_Vy_loc     = SVector{4}(Δ.x[ii] for ii in i-1:i+2)
            Δy_Vy_loc     = SVector{4}(Δ.y[jj] for jj in j-2:j+1)

            Δt_loc        = Δ.t[1]

            fill!(∂R∂Vx, 0e0)
            fill!(∂R∂Vy, 0e0)
            fill!(∂R∂Pt, 0e0)
            
            autodiff(Enzyme.Reverse, SMomentum_x_Generic_var, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ΔP_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δx_Vx_loc), Const(Δy_Vx_loc), Const(Δx_Vy_loc), Const(Δy_Vy_loc), Const(Δt_loc))
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



function AssembleMomentum2D_y_var!(K, V, P, P0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
    
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
            ΔP_loc    .= @inline SMatrix{1,2}(@inbounds      ΔP.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
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

            Δx_Vx_loc     = SVector{4}(Δ.x[ii] for ii in i-2:i+1)
            Δy_Vx_loc     = SVector{4}(Δ.y[jj] for jj in j-1:j+2)
            Δx_Vy_loc     = SVector{3}(Δ.x[ii] for ii in i-1:i+1)
            Δy_Vy_loc     = SVector{3}(Δ.y[jj] for jj in j-1:j+1)

            Δt_loc        = Δ.t[1]

            fill!(∂R∂Vx, 0.0)
            fill!(∂R∂Vy, 0.0)
            fill!(∂R∂Pt, 0.0)
            
            autodiff(Enzyme.Reverse, SMomentum_y_Generic_var, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ΔP_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δx_Vx_loc), Const(Δy_Vx_loc), Const(Δx_Vy_loc), Const(Δy_Vy_loc), Const(Δt_loc))
            
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


#----------------------------
#      Line search
#----------------------------
function LineSearch_var!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x 
    Vi.y .= V.y 
    Pti  .= Pt
    for i in eachindex(α)
        V.x .= Vi.x 
        V.y .= Vi.y
        Pt  .= Pti
        UpdateSolution!(V, Pt, α[i].*dx, number, type, nc)
        TangentOperator_var!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, type, BC, materials, phases, Δ)
        ResidualContinuity2D_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
        ResidualMomentum2D_x_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        rvec[i] = @views norm(R.x[inx_Vx,iny_Vx])/length(R.x[inx_Vx,iny_Vx]) + norm(R.y[inx_Vy,iny_Vy])/length(R.y[inx_Vy,iny_Vy]) + 0*norm(R.p[inx_c,iny_c])/length(R.p[inx_c,iny_c])  
    end
    imin = argmin(rvec)
    V.x .= Vi.x 
    V.y .= Vi.y
    Pt  .= Pti
    return imin
end
