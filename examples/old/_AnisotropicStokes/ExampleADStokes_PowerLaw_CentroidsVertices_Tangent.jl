using StagFDTools, StagFDTools.Stokes, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays
import Statistics:mean
using DifferentiationInterface
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian
function Momentum_x_Generic(Vx, Vy, Pt, 𝐷, phases, materials, type, bcv, Δ)
    
    invΔx, invΔy = 1 / Δ.x, 1 / Δ.y

    # BC
    SetBCVx1!(Vx, type.x, bcv.x, Δ)
    SetBCVy1!(Vy, type.y, bcv.y, Δ)

    # Velocity gradient
    Dxx = (Vx[2:end,:] - Vx[1:end-1,:]) * invΔx             # Static Arrays ???
    Dyy = (Vy[2:end-1,2:end] - Vy[2:end-1,1:end-1]) * invΔy             
    Dxy = (Vx[:,2:end] - Vx[:,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,2:end-1] - Vy[1:end-1,2:end-1]) * invΔx 

    # Strain rate
    ε̇kk = Dxx + Dyy
    ε̇xx = Dxx - 1/3*ε̇kk
    ε̇yy = Dyy - 1/3*ε̇kk
    ε̇xy = 1/2 * ( Dxy + Dyx )

    # Average vertex to centroid
    ε̇̄xy = 0.25*(ε̇xy[1:end-1,1:end-1] .+ ε̇xy[1:end-1,2:end-0] .+ ε̇xy[2:end-0,1:end-1] .+ ε̇xy[2:end,2:end])
    # Average centroid to vertex
    ε̇̄xx = 0.25*(ε̇xx[1:end-1,1:end-1] .+ ε̇xx[1:end-1,2:end-0] .+ ε̇xx[2:end-0,1:end-1] .+ ε̇xx[2:end,2:end])
    ε̇̄yy = 0.25*(ε̇yy[1:end-1,1:end-1] .+ ε̇yy[1:end-1,2:end-0] .+ ε̇yy[2:end-0,1:end-1] .+ ε̇yy[2:end,2:end])

    # Stress
    τxx = @MMatrix zeros(2,1)
    τxy = @MMatrix zeros(1,2)
    for i=1:2
        τxx[i,1] = 𝐷.c[i][1,1] .* ε̇xx[i,2] + 𝐷.c[i][1,2] .* ε̇yy[i,2] + 𝐷.c[i][1,3] .* ε̇̄xy[i,1]
        τxy[1,i] = 𝐷.v[i][3,1] .* ε̇̄xx[1,i] + 𝐷.v[i][3,2] .* ε̇̄yy[1,i] + 𝐷.v[i][3,3] .* ε̇xy[2,i]
    end

    # Residual
    fx  = (τxx[2,1] - τxx[1,1]) * invΔx
    fx += (τxy[1,2] - τxy[1,1]) * invΔy
    fx -= ( Pt[2,2] -  Pt[1,2]) * invΔx
    fx *= -1 * Δ.x * Δ.y

    return fx
end

function Momentum_y_Generic(Vx, Vy, Pt, 𝐷, phases, materials, type, bcv, Δ)
    
    invΔx, invΔy = 1 / Δ.x, 1 / Δ.y

    # BC
    SetBCVx1!(Vx, type.x, bcv.x, Δ)
    SetBCVy1!(Vy, type.y, bcv.y, Δ)

    # Velocity gradient
    Dxx = (Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1]) * invΔx             # Static Arrays ???
    Dyy = (Vy[:,2:end] - Vy[:,1:end-1]) * invΔy             
    Dxy = (Vx[2:end-1,2:end] - Vx[2:end-1,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,:] - Vy[1:end-1,:]) * invΔx 

    # Strain rate
    ε̇kk = Dxx + Dyy
    ε̇xx = Dxx - 1/3*ε̇kk
    ε̇yy = Dyy - 1/3*ε̇kk
    ε̇xy = 1/2 * ( Dxy + Dyx )

    # Average vertex to centroid
    ε̇̄xy = 0.25*(ε̇xy[1:end-1,1:end-1] .+ ε̇xy[1:end-1,2:end-0] .+ ε̇xy[2:end-0,1:end-1] .+ ε̇xy[2:end,2:end])
    # Average centroid to vertex
    ε̇̄xx = 0.25*(ε̇xx[1:end-1,1:end-1] .+ ε̇xx[1:end-1,2:end-0] .+ ε̇xx[2:end-0,1:end-1] .+ ε̇xx[2:end,2:end])
    ε̇̄yy = 0.25*(ε̇yy[1:end-1,1:end-1] .+ ε̇yy[1:end-1,2:end-0] .+ ε̇yy[2:end-0,1:end-1] .+ ε̇yy[2:end,2:end])

    # Stress
    τyy = @MMatrix zeros(1,2)
    τxy = @MMatrix zeros(2,1)
    for i=1:2
        τyy[1,i] = 𝐷.c[i][2,1] .* ε̇xx[2,i] + 𝐷.c[i][2,2] .* ε̇yy[2,i] + 𝐷.c[i][2,3] .* ε̇̄xy[1,i]
        τxy[i,1] = 𝐷.v[i][3,1] .* ε̇̄xx[i,1] + 𝐷.v[i][3,2] .* ε̇̄yy[i,1] + 𝐷.v[i][3,3] .* ε̇xy[i,2]
    end

    # Residual
    fy  = (τyy[1,2] - τyy[1,1]) * invΔy
    fy += (τxy[2,1] - τxy[1,1]) * invΔx
    fy -= ( Pt[2,2] -  Pt[2,1]) * invΔy
    fy *= -1 * Δ.x * Δ.y
    
    return fy
end


function TangentOperator!(𝐷, 𝐷_ctl, ε̇, V, Pt, type, BC, materials, phases, Δ)

    # Loop over centroids
    for j=2:size(ε̇.xx,2)-1, i=2:size(ε̇.xx,1)-1
        Vx     = MMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1,   jj in j:j+2)
        Vy     = MMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2,   jj in j:j+1)
        bcx    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        bcy    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
        typex  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        typey  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)

        SetBCVx1!(Vx, typex, bcx, Δ)
        SetBCVy1!(Vy, typey, bcy, Δ)

        Dxx = (Vx[2:end,:] - Vx[1:end-1,:]) / Δ.x
        Dyy = (Vy[:,2:end] - Vy[:,1:end-1]) / Δ.y
        Dxy = (Vx[:,2:end] - Vx[:,1:end-1]) / Δ.y
        Dyx = (Vy[2:end,:] - Vy[1:end-1,:]) / Δ.x

        Dkk   = Dxx[:,2:end-1] + Dyy[2:end-1,:]
        ε̇xx   = Dxx[:,2:end-1] - 1/3*Dkk
        ε̇yy   = Dyy[2:end-1,:] - 1/3*Dkk
        ε̇xy   = 1/2*(Dxy + Dyx)  
        ε̇̄xy   = 0.25*(ε̇xy[1:end-1,1:end-1] .+ ε̇xy[1:end-1,2:end-0] .+ ε̇xy[2:end-0,1:end-1] .+ ε̇xy[2:end,2:end])
        ε̇vec  = @SVector([ε̇xx[1], ε̇yy[1], ε̇̄xy[1]])
        
        # Tangent operator used for Newton Linearisation
        jac   = forwarddiff_jacobian(StressVector!, ε̇vec, Const(materials), Const(phases.c[i,j]))
        
        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        𝐷_ctl.c[i,j][:,1] .= jac.derivs[1][1][1]
        𝐷_ctl.c[i,j][:,2] .= jac.derivs[1][2][1]
        𝐷_ctl.c[i,j][:,3] .= jac.derivs[1][3][1]

        # Tangent operator used for Picard Linearisation
        𝐷.c[i,j] .= diagm(2*jac.val[2]*ones(3))
    end

    # Loop over vertices
    for j=1:size(ε̇.xy,2), i=1:size(ε̇.xy,1)
        Vx     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        Vy     = MMatrix{2,3}(      V.y[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        bcx    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        bcy    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        typex  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        typey  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        
        SetBCVx1!(Vx, typex, bcx, Δ)
        SetBCVy1!(Vy, typey, bcy, Δ)
        
        Dxx = (Vx[2:end,:] - Vx[1:end-1,:]) / Δ.x
        Dyy = (Vy[:,2:end] - Vy[:,1:end-1]) / Δ.y
        Dxy = (Vx[:,2:end] - Vx[:,1:end-1]) / Δ.y
        Dyx = (Vy[2:end,:] - Vy[1:end-1,:]) / Δ.x

        Dkk   = Dxx + Dyy
        ε̇xx   = Dxx - 1/3*Dkk
        ε̇yy   = Dyy - 1/3*Dkk
        ε̇xy   = 1/2*(Dxy[2:end-1,:] + Dyx[:,2:end-1]) 
        ε̇̄xx   = 0.25*(ε̇xx[1:end-1,1:end-1] .+ ε̇xx[1:end-1,2:end-0] .+ ε̇xx[2:end-0,1:end-1] .+ ε̇xx[2:end,2:end])
        ε̇̄yy   = 0.25*(ε̇yy[1:end-1,1:end-1] .+ ε̇yy[1:end-1,2:end-0] .+ ε̇yy[2:end-0,1:end-1] .+ ε̇yy[2:end,2:end])
        ε̇vec = @SVector([ε̇̄xx[1], ε̇̄yy[1], ε̇xy[1]])
        
        # Tangent operator used for Newton Linearisation
        jac   = forwarddiff_jacobian(StressVector!, ε̇vec, Const(materials), Const(phases.v[i,j]))

        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        𝐷_ctl.v[i,j][:,1] .= jac.derivs[1][1][1]
        𝐷_ctl.v[i,j][:,2] .= jac.derivs[1][2][1]
        𝐷_ctl.v[i,j][:,3] .= jac.derivs[1][3][1]

        # Tangent operator used for Picard Linearisation
        𝐷.v[i,j] .= diagm(2*jac.val[2]*ones(3))
    end
end

function Continuity(Vx, Vy, Pt, D, type_loc, bcv_loc, Δ)
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    return ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy)
end

function ResidualMomentum2D_x!(R, V, P, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-1:i-1, jj in j-2:j-1)
        P_loc      = SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        Dc         = SMatrix{2,1}(𝐷.c[ii,j-1] for ii in i-1:i,   jj in j-1:j-1)
        Dv         = SMatrix{1,2}(𝐷.v[i-1,jj] for ii in i-1:i-1, jj in j-2:j-1)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)
        D          = (c=Dc, v=Dv)
        if type.Vx[i,j] == :in
            R.x[i,j]   = Momentum_x_Generic(Vx_loc, Vy_loc, P_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,3)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        
        if type.Vx[i,j] == :in

            bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-1:i-1, jj in j-2:j-1) 
            Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            P_loc      = MMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            Dc         = SVector{2}(𝐷.c[ii,j-1] for ii in i-1:i  )
            Dv         = SVector{2}(𝐷.v[i-1,jj] for jj in j-2:j-1)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            forwarddiff_gradients!(Momentum_x_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
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
        end
    end
    return nothing
end

function ResidualMomentum2D_y!(R, V, P, 𝐷, phases, materials, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-2:i-1, jj in j-1:j-1) 
        P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Dc         = SVector{2}(𝐷.c[i-1,jj] for jj in j-1:j  )
        Dv         = SVector{2}(𝐷.v[ii,j-1] for ii in i-2:i-1)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)
        D          = (c=Dc, v=Dv)

        if type.Vy[i,j] == :in
            R.y[i,j]   = Momentum_y_Generic(Vx_loc, Vy_loc, P_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(3,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x

        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-2:i-1, jj in j-1:j-1) 
        P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Dc         = SVector{2}(𝐷.c[i-1,jj] for jj in j-1:j  )
        Dv         = SVector{2}(𝐷.v[ii,j-1] for ii in i-2:i-1)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)
        D          = (c=Dc, v=Dv)

        if type.Vy[i,j] == :in
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            forwarddiff_gradients!(Momentum_y_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
            # Vy --- Vx
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
        end
    end
    return nothing
end

function ResidualContinuity2D!(R, V, P, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    for j in 2:size(R.p,2)-1, i in 2:size(R.p,1)-1
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (;)
        type_loc   = (;)
        D          = (;)
        R.p[i,j]   = Continuity(Vx_loc, Vy_loc, P[i,j], D, type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
                
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)

    for j in 2:size(P, 2)-1, i in 2:size(P, 1)-1
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (;)
        type_loc   = (;)
        D          = (;)
        
        ∂R∂Vx .= 0.
        ∂R∂Vy .= 0.
        forwarddiff_gradients!(Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Const(P[i,j]), Const(D), Const(type_loc), Const(bcv_loc), Const(Δ))

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
    end
    return nothing
end

function SetBCVx1!(Vx, typex, bcx, Δ)
    for ii in axes(typex, 1)
        if typex[ii,1] == :Dirichlet
            Vx[ii,1] = fma(2, bcx[ii,1], -Vx[ii,2])
        elseif typex[ii,1] == :Neumann
            Vx[ii,1] = fma(Δ.y, bcx[ii,1], Vx[ii,2])
        end

        if typex[ii,end] == :Dirichlet
            Vx[ii,end] = fma(2, bcx[ii,end], -Vx[ii,end-1])
        elseif typex[ii,end] == :Neumann
            Vx[ii,end] = fma(Δ.y, bcx[ii,end], Vx[ii,end-1])
        end
    end
end

function SetBCVy1!(Vy, typey, bcy, Δ)
    for jj in axes(typey, 2)
        if typey[1,jj] == :Dirichlet
            Vy[1,jj] = fma(2, bcy[1,jj], -Vy[2,jj])
        elseif typey[1,jj] == :Neumann
            Vy[1,jj] = fma(Δ.y, bcy[1,jj], Vy[2,jj])
        end

        if typey[end,jj] == :Dirichlet
            Vy[end,jj] = fma(2, bcy[end,jj], -Vy[end-1,jj])
        elseif typey[end,jj] == :Neumann
            Vy[end,jj] = fma(Δ.y, bcy[end,jj], Vy[end-1,jj])
        end
    end
end

function LocalRheology(ε̇, materials, phases)
    ε̇II = sqrt.(1/2*(ε̇[1].^2 .+ ε̇[2].^2) + ε̇[3].^2)
    n   = materials.n[phases]
    η0  = materials.η0[phases]
    η   =  (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
    return η
end

function StressVector!(ε̇, materials, phases) 
    η   =  LocalRheology(ε̇, materials, phases)
    τ   = @SVector([2 * η * ε̇[1],
                    2 * η * ε̇[2],
                    2 * η * ε̇[3]])
    return τ, η
end



@views function main(nc)
    #--------------------------------------------#
    # Resolution

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    BC = Fields(
        fill(0., (nc.x+3, nc.y+4)),
        fill(0., (nc.x+4, nc.y+3)),
        fill(0., (nc.x+2, nc.y+2)),
    )
    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx]  .= :in       
    type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
    type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
    type.Vx[inx_Vx,2]       .= :Dirichlet
    type.Vx[inx_Vx,end-1]   .= :Dirichlet
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy]  .= :in       
    type.Vy[2,iny_Vy]       .= :Dirichlet
    type.Vy[end-1,iny_Vy]   .= :Dirichlet
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in

    #--------------------------------------------#
    # Equation numbering
    number = Fields(
        fill(0, size_x),
        fill(0, size_y),
        fill(0, size_c),
    )
    Numbering!(number, type, nc)

    #--------------------------------------------#
    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([0 1 0; 0 1 0])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([0 0; 1 1; 0 0])), 
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )

    #--------------------------------------------#
    # Intialise field
    L  = (x=1.0, y=1.0)
    Δ  = (x=L.x/nc.x, y=L.y/nc.y)
    R  = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V  = (x=zeros(size_x...), y=zeros(size_y...))
    η  = (c=ones(size_c...), v=ones(size_v...) )
    ε̇  = (xx=ones(size_c...), yy=ones(size_c...), xy=ones(size_v...) )
    τ  = (xx=ones(size_c...), yy=ones(size_c...), xy=ones(size_v...) )

    Dc  =  [@MMatrix(zeros(3,3)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv  =  [@MMatrix(zeros(3,3)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    𝐷   = (c = Dc, v = Dv)

    D_ctl_c =  [@MMatrix(zeros(3,3)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(3,3)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    𝐷_ctl = (c = D_ctl_c, v = D_ctl_v)

    Pt = zeros(size_c...)
    xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    materials = ( 
        n  = [3.0 1.0],
        η0 = [1e2 1e-1] 
    )

    # Initial configuration
    D_BC = [-1  0;
             0  1]
    # D_BC = [1e-15  1;
    #        0  1e-15]
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'

    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_conf) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_conf) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     2 ] .== :Neumann_conf) .* D_BC[1,1]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy, end-1 ] .== :Neumann_conf) .* D_BC[1,1]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)

    phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    phases.v[(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2

    p1 = heatmap(xc, yc, phases.c[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc))
    p2 = heatmap(xv, yv, phases.v', aspect_ratio=1, xlim=extrema(xc))
    display(plot(p1, p2))
    #--------------------------------------------#
    # Newton solver
    niter = 10
    err   = (x = zeros(niter), y = zeros(niter), p = zeros(niter))

    TangentOperator!(𝐷, 𝐷_ctl, ε̇, V, Pt, type, BC, materials, phases, Δ)
    
    #--------------------------------------------#

    for iter=1:niter

        #--------------------------------------------#
        # Residual check
        TangentOperator!(𝐷, 𝐷_ctl, ε̇, V, Pt, type, BC, materials, phases, Δ)
        ResidualContinuity2D!(R, V, Pt, 𝐷, phases, materials, number, type, BC, nc, Δ) 
        ResidualMomentum2D_x!(R, V, Pt, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, Pt, 𝐷, phases, materials, number, type, BC, nc, Δ)

        err.x[iter] = norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
        err.y[iter] = norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
        err.p[iter] = norm(R.p[inx_c,iny_c])/sqrt(nPt)

        #--------------------------------------------#
        # Set global residual vector
        r = zeros(nVx + nVy + nPt)
        SetRHS!(r, R, number, type, nc)

        #--------------------------------------------#
        # Assembly
        AssembleContinuity2D!(M, V, Pt, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
        AssembleMomentum2D_x!(M, V, Pt, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
        AssembleMomentum2D_y!(M, V, Pt, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)

        # Stokes operator as block matrices
        𝐊  = [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
        𝐐  = [M.Vx.Pt; M.Vy.Pt]
        𝐐ᵀ = [M.Pt.Vx M.Pt.Vy]
        𝐌 = [𝐊 𝐐; 𝐐ᵀ M.Pt.Pt]
        
        #--------------------------------------------#
        # Direct solver (TODO: need a better solver)
        dx = - 𝐌 \ r

        α     = LinRange(0.05, 1.0, 10)
        Vi    = (x=zeros(size_x...), y=zeros(size_y...))
        Pti   = zeros(size_c...)
        Vi.x .= V.x 
        Vi.y .= V.y 
        Pti  .= Pt
        rvec = zeros(length(α))
        for i in eachindex(α)
            V.x .= Vi.x 
            V.y .= Vi.y
            Pt  .= Pti
            UpdateSolution!(V, Pt, α[i].*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, ε̇, V, Pt, type, BC, materials, phases, Δ)
            ResidualContinuity2D!(R, V, Pt, 𝐷, phases, materials, number, type, BC, nc, Δ) 
            ResidualMomentum2D_x!(R, V, Pt, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(R, V, Pt, 𝐷, phases, materials, number, type, BC, nc, Δ)
            rvec[i] = norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx) + norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy) + norm(R.p[inx_c,iny_c])/sqrt(nPt)   
        end
        _, imin = findmin(rvec)
        V.x .= Vi.x 
        V.y .= Vi.y
        Pt  .= Pti

        #--------------------------------------------#
        # Update solutions
        UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
    end

    #--------------------------------------------#
    p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vx")
    p2 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc), title="Vy")
    p3 = heatmap(xc, yc,  Pt[inx_c,iny_c]' .- mean(Pt[inx_c,iny_c]), aspect_ratio=1, xlim=extrema(xc), title="Pt")
    p4 = plot(xlabel="Iterations", ylabel="log₁₀ error")
    p4 = plot!(1:niter, log10.(err.x[1:niter]), label="Vx")
    p4 = plot!(1:niter, log10.(err.y[1:niter]), label="Vy")
    p4 = plot!(1:niter, log10.(err.p[1:niter]), label="Pt")
    display(plot(p1, p2, p3, p4, layout=(3,2)))
    
end

main((x = 30, y = 30))