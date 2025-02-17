using StagFDTools, StagFDTools.Stokes, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use
using TimerOutputs

function PowerLaw(ε̇, materials, phases, Δ)
    ε̇II  = sqrt.(1/2*(ε̇[1].^2 .+ ε̇[2].^2) + ε̇[3].^2)
    P    = ε̇[4]
    n    = materials.n[phases]
    η0   = materials.η0[phases]
    G    = materials.G[phases]
    C    = materials.C[phases]
    ϕ    = materials.ϕ[phases]
    ηvp  = materials.ηvp[phases]
    η    =  (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
    τy   = C*cosd(ϕ) + P*sind(ϕ) 
    ηvep = inv(1/η + 1/(G*Δ.t))

    τII  = 2*ηvep*ε̇II
    λ̇    = 0.0
    F    = τII - τy - λ̇*ηvp

    if F > 0
        λ̇    = F / (ηvep + ηvp) 
        τII -= λ̇*ηvep
        # τII = C*cosd(ϕ) + P*sind(ϕ) + ηvp*λ̇
        ηvep = τII/(2*ε̇II)
    end
    return ηvep, λ̇
end

function Rheology!(ε̇, materials, phases, Δ) 
    η, λ̇ = PowerLaw(ε̇, materials, phases, Δ)
    τ    = @SVector([2 * η * ε̇[1],
                     2 * η * ε̇[2],
                     2 * η * ε̇[3],
                             ε̇[4]])
    return τ, η, λ̇
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, Pt, type, BC, materials, phases, Δ)

    # Loop over centroids
    for j=2:size(ε̇.xx,2)-1, i=2:size(ε̇.xx,1)-1
        Vx     = MMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1,   jj in j:j+2)
        Vy     = MMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2,   jj in j:j+1)
        bcx    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        bcy    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
        typex  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        typey  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
        τxy0   = MMatrix{2,2}(    τ0.xy[ii,jj] for ii in i-1:i,   jj in j-1:j)

        SetBCVx1!(Vx, typex, bcx, Δ)
        SetBCVy1!(Vy, typey, bcy, Δ)

        Dxx = SVector{1}( (Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1]) / Δ.x )
        Dyy = SVector{1}( (Vy[2:end-1,2:end] - Vy[2:end-1,1:end-1]) / Δ.y )
        Dxy = SMatrix{2,2}( (Vx[:,2:end] - Vx[:,1:end-1]) / Δ.y )
        Dyx = SMatrix{2,2}( (Vy[2:end,:] - Vy[1:end-1,:]) / Δ.x )

        Dkk   = SVector{1}( Dxx + Dyy     )
        ε̇xx   = SVector{1}( Dxx - 1/3*Dkk )
        ε̇yy   = SVector{1}( Dyy - 1/3*Dkk )
        ε̇xy   = SMatrix{2,2}( 1/2*(Dxy + Dyx) )  
        ε̇̄xy   = SVector{1}( 0.25*(ε̇xy[1:end-1,1:end-1] .+ ε̇xy[1:end-1,2:end-0] .+ ε̇xy[2:end-0,1:end-1] .+ ε̇xy[2:end,2:end]) )
       
        # Visco-elasticity
        G     = SVector{1}( materials.G[phases.c[i,j]] )
        τ̄xy0  = SVector{1}( 0.25*(τxy0[1:end-1,1:end-1] .+ τxy0[1:end-1,2:end-0] .+ τxy0[2:end-0,1:end-1] .+ τxy0[2:end,2:end]) )
        ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), Pt[i,j]])
        
        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, Rheology!, ε̇vec, Const(materials), Const(phases.c[i,j]), Const(Δ))
        
        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        𝐷_ctl.c[i,j][:,1] .= jac.derivs[1][1][1]
        𝐷_ctl.c[i,j][:,2] .= jac.derivs[1][2][1]
        𝐷_ctl.c[i,j][:,3] .= jac.derivs[1][3][1]
        𝐷_ctl.c[i,j][:,4] .= jac.derivs[1][4][1]

        # Tangent operator used for Picard Linearisation
        𝐷.c[i,j] .= diagm(2*jac.val[2]*ones(4))

        # Update stress
        τ.xx[i,j] = jac.val[1][1]
        τ.yy[i,j] = jac.val[1][2]
        λ̇.c[i,j]  = jac.val[3]
        η.c[i,j]  = jac.val[2]
    end

    # Loop over vertices
    for j=1:size(ε̇.xy,2), i=1:size(ε̇.xy,1)
        Vx     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        Vy     = MMatrix{2,3}(      V.y[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        bcx    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        bcy    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        typex  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        typey  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        τxx0   = MMatrix{2,2}(    τ0.xx[ii,jj] for ii in i:i+1,   jj in j:j+1)
        τyy0   = MMatrix{2,2}(    τ0.yy[ii,jj] for ii in i:i+1,   jj in j:j+1)
        P      = MMatrix{2,2}(       Pt[ii,jj] for ii in i:i+1,   jj in j:j+1)

        SetBCVx1!(Vx, typex, bcx, Δ)
        SetBCVy1!(Vy, typey, bcy, Δ)
    
        Dxx = SMatrix{2,2}( (Vx[2:end,:] - Vx[1:end-1,:]) / Δ.x ) 
        Dyy = SMatrix{2,2}( (Vy[:,2:end] - Vy[:,1:end-1]) / Δ.y )
        Dxy = SVector{1}( (Vx[2:end-1,2:end] - Vx[2:end-1,1:end-1]) / Δ.y )
        Dyx = SVector{1}( (Vy[2:end,2:end-1] - Vy[1:end-1,2:end-1]) / Δ.x )

        Dkk   = SMatrix{2,2}( Dxx + Dyy )
        ε̇xx   = SMatrix{2,2}( Dxx - 1/3*Dkk )
        ε̇yy   = SMatrix{2,2}( Dyy - 1/3*Dkk )
        ε̇xy   = SVector{1}( 1/2*(Dxy + Dyx) )
        ε̇̄xx   = SVector{1}( 0.25*(ε̇xx[1:end-1,1:end-1] .+ ε̇xx[1:end-1,2:end-0] .+ ε̇xx[2:end-0,1:end-1] .+ ε̇xx[2:end,2:end]) )
        ε̇̄yy   = SVector{1}( 0.25*(ε̇yy[1:end-1,1:end-1] .+ ε̇yy[1:end-1,2:end-0] .+ ε̇yy[2:end-0,1:end-1] .+ ε̇yy[2:end,2:end]) )
        
        # Visco-elasticity
        G     = SVector{1}( materials.G[phases.v[i,j]] )
        τ̄xx0  = SVector{1}( 0.25*(τxx0[1:end-1,1:end-1] .+ τxx0[1:end-1,2:end-0] .+ τxx0[2:end-0,1:end-1] .+ τxx0[2:end,2:end]) )
        τ̄yy0  = SVector{1}( 0.25*(τyy0[1:end-1,1:end-1] .+ τyy0[1:end-1,2:end-0] .+ τyy0[2:end-0,1:end-1] .+ τyy0[2:end,2:end]) )        
        P̄     = SVector{1}( 0.25*(   P[1:end-1,1:end-1] .+    P[1:end-1,2:end-0] .+    P[2:end-0,1:end-1] .+    P[2:end,2:end]) ) 
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t), P̄[1]])
        
        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, Rheology!, ε̇vec, Const(materials), Const(phases.v[i,j]), Const(Δ))

        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        𝐷_ctl.v[i,j][:,1] .= jac.derivs[1][1][1]
        𝐷_ctl.v[i,j][:,2] .= jac.derivs[1][2][1]
        𝐷_ctl.v[i,j][:,3] .= jac.derivs[1][3][1]
        𝐷_ctl.v[i,j][:,4] .= jac.derivs[1][4][1]

        # Tangent operator used for Picard Linearisation
        𝐷.v[i,j] .= diagm(2*jac.val[2]*ones(4))

        # Update stress
        τ.xy[i,j] = jac.val[1][3]
        λ̇.v[i,j]  = jac.val[3]
        η.v[i,j]  = jac.val[2]
    end
end

function Momentum_x_Generic(Vx, Vy, Pt, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
    invΔx, invΔy = 1 / Δ.x, 1 / Δ.y

    # BC
    SetBCVx1!(Vx, type.x, bcv.x, Δ)
    SetBCVy1!(Vy, type.y, bcv.y, Δ)

    # Velocity gradient
    Dxx = SMatrix{2,3}( (Vx[2:end,:]       - Vx[1:end-1,:]      ) * invΔx )           
    Dyy = SMatrix{2,3}( (Vy[2:end-1,2:end] - Vy[2:end-1,1:end-1]) * invΔy )            
    Dxy = SMatrix{3,2}( (Vx[:,2:end]       - Vx[:,1:end-1]      ) * invΔy ) 
    Dyx = SMatrix{3,2}( (Vy[2:end,2:end-1] - Vy[1:end-1,2:end-1]) * invΔx ) 

    # Strain rate
    ε̇kk = SMatrix{2,3}( Dxx + Dyy           ) 
    ε̇xx = SMatrix{2,3}( Dxx - 1/3*ε̇kk       ) 
    ε̇yy = SMatrix{2,3}( Dyy - 1/3*ε̇kk       )
    ε̇xy = SMatrix{3,2}( 1/2 * ( Dxy + Dyx ) )

    # Average vertex to centroid
    ε̇̄xy  = SVector{2}( 0.25*(ε̇xy[1:end-1,1:end-1] .+ ε̇xy[1:end-1,2:end-0] .+ ε̇xy[2:end-0,1:end-1] .+ ε̇xy[2:end,2:end]) )
    # Average centroid to vertex
    ε̇̄xx  = SVector{2}( 0.25*(ε̇xx[1:end-1,1:end-1] .+ ε̇xx[1:end-1,2:end-0] .+ ε̇xx[2:end-0,1:end-1] .+ ε̇xx[2:end,2:end]) )
    ε̇̄yy  = SVector{2}( 0.25*(ε̇yy[1:end-1,1:end-1] .+ ε̇yy[1:end-1,2:end-0] .+ ε̇yy[2:end-0,1:end-1] .+ ε̇yy[2:end,2:end]) )
    P̄t   = SVector{2}( 0.25*( Pt[1:end-1,1:end-1] .+  Pt[1:end-1,2:end-0] .+  Pt[2:end-0,1:end-1] .+  Pt[2:end,2:end]) )
    τ̄0xx = SVector{2}( 0.25*(τ0.xx[1:end-1,1:end-1] .+ τ0.xx[1:end-1,2:end-0] .+ τ0.xx[2:end-0,1:end-1] .+ τ0.xx[2:end,2:end]) )
    τ̄0yy = SVector{2}( 0.25*(τ0.yy[1:end-1,1:end-1] .+ τ0.yy[1:end-1,2:end-0] .+ τ0.yy[2:end-0,1:end-1] .+ τ0.yy[2:end,2:end]) )
    τ̄0xy = SVector{2}( 0.25*(τ0.xy[1:end-1,1:end-1] .+ τ0.xy[1:end-1,2:end-0] .+ τ0.xy[2:end-0,1:end-1] .+ τ0.xy[2:end,2:end]) )

    # Effective strain rate
    Gc  = SVector{2}( materials.G[phases.c] )
    Gv  = SVector{2}( materials.G[phases.v] )
    ϵ̇xx = SVector{2}( ε̇xx[:,2] .+ τ0.xx[:,2] ./ (2*Gc.*Δ.t) )
    ϵ̇yy = SVector{2}( ε̇yy[:,2] .+ τ0.yy[:,2] ./ (2*Gc.*Δ.t) )
    ϵ̇̄xy = SVector{2}( ε̇̄xy[:]   .+ τ̄0xy[:]    ./ (2*Gc.*Δ.t) )
    ϵ̇̄xx = SVector{2}( ε̇̄xx[:]   .+ τ̄0xx[:]    ./ (2*Gv.*Δ.t) )
    ϵ̇̄yy = SVector{2}( ε̇̄yy[:]   .+ τ̄0yy[:]    ./ (2*Gv.*Δ.t) )
    ϵ̇xy = SVector{2}( ε̇xy[2,:] .+ τ0.xy[2,:] ./ (2*Gv.*Δ.t) )

    # Stress
    τxx = @MVector zeros(2)
    τxy = @MVector zeros(2)
    for i=1:2
        τxx[i] = 𝐷.c[i][1,1] .* ϵ̇xx[i] + 𝐷.c[i][1,2] .* ϵ̇yy[i] + 𝐷.c[i][1,3] .* ϵ̇̄xy[i] + 𝐷.c[i][1,4] * Pt[i,2]
        τxy[i] = 𝐷.v[i][3,1] .* ϵ̇̄xx[i] + 𝐷.v[i][3,2] .* ϵ̇̄yy[i] + 𝐷.v[i][3,3] .* ϵ̇xy[i] + 𝐷.v[i][3,4] * P̄t[i]
    end

    # Residual
    fx  = ( τxx[2]  - τxx[1] ) * invΔx
    fx += ( τxy[2]  - τxy[1] ) * invΔy
    fx -= ( Pt[2,2] - Pt[1,2]) * invΔx
    # fx *= -1 * Δ.x * Δ.y

    return fx
end

function Momentum_y_Generic(Vx, Vy, Pt, τ0, 𝐷, phases, materials, type, bcv, Δ)
    
    invΔx, invΔy = 1 / Δ.x, 1 / Δ.y

    # BC
    SetBCVx1!(Vx, type.x, bcv.x, Δ)
    SetBCVy1!(Vy, type.y, bcv.y, Δ)

    # Velocity gradient
    Dxx = SMatrix{3,2}( (Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1]) * invΔx )             # Static Arrays ???
    Dyy = SMatrix{3,2}( (Vy[:,2:end]       - Vy[:,1:end-1]      ) * invΔy )             
    Dxy = SMatrix{2,3}( (Vx[2:end-1,2:end] - Vx[2:end-1,1:end-1]) * invΔy ) 
    Dyx = SMatrix{2,3}( (Vy[2:end,:]       - Vy[1:end-1,:]      ) * invΔx ) 

    # Strain rate
    ε̇kk = SMatrix{3,2}( Dxx + Dyy           )
    ε̇xx = SMatrix{3,2}( Dxx - 1/3*ε̇kk       )
    ε̇yy = SMatrix{3,2}( Dyy - 1/3*ε̇kk       )
    ε̇xy = SMatrix{2,3}( 1/2 * ( Dxy + Dyx ) )

    # Average vertex to centroid
    ε̇̄xy  = SVector{2}( 0.25*(ε̇xy[1:end-1,1:end-1] .+ ε̇xy[1:end-1,2:end-0] .+ ε̇xy[2:end-0,1:end-1] .+ ε̇xy[2:end,2:end])         )
    # Average centroid to vertex
    ε̇̄xx  = SVector{2}( 0.25*(ε̇xx[1:end-1,1:end-1] .+ ε̇xx[1:end-1,2:end-0] .+ ε̇xx[2:end-0,1:end-1] .+ ε̇xx[2:end,2:end])         )
    ε̇̄yy  = SVector{2}( 0.25*(ε̇yy[1:end-1,1:end-1] .+ ε̇yy[1:end-1,2:end-0] .+ ε̇yy[2:end-0,1:end-1] .+ ε̇yy[2:end,2:end])         )
    P̄t   = SVector{2}( 0.25*( Pt[1:end-1,1:end-1] .+  Pt[1:end-1,2:end-0] .+  Pt[2:end-0,1:end-1] .+  Pt[2:end,2:end])         )
    τ̄0xx = SVector{2}( 0.25*(τ0.xx[1:end-1,1:end-1] .+ τ0.xx[1:end-1,2:end-0] .+ τ0.xx[2:end-0,1:end-1] .+ τ0.xx[2:end,2:end]) )
    τ̄0yy = SVector{2}( 0.25*(τ0.yy[1:end-1,1:end-1] .+ τ0.yy[1:end-1,2:end-0] .+ τ0.yy[2:end-0,1:end-1] .+ τ0.yy[2:end,2:end]) )
    τ̄0xy = SVector{2}( 0.25*(τ0.xy[1:end-1,1:end-1] .+ τ0.xy[1:end-1,2:end-0] .+ τ0.xy[2:end-0,1:end-1] .+ τ0.xy[2:end,2:end]) )
    
    # Effective strain rate
    Gc  = SVector{2}( materials.G[phases.c] )
    Gv  = SVector{2}( materials.G[phases.v] )
    ϵ̇xx = SVector{2}( ε̇xx[2,:] .+ τ0.xx[2,:] ./ (2*Gc.*Δ.t) )
    ϵ̇yy = SVector{2}( ε̇yy[2,:] .+ τ0.yy[2,:] ./ (2*Gc.*Δ.t) )
    ϵ̇̄xy = SVector{2}( ε̇̄xy[:]   .+ τ̄0xy[:]    ./ (2*Gc.*Δ.t) )
    ϵ̇̄xx = SVector{2}( ε̇̄xx[:]   .+ τ̄0xx[:]    ./ (2*Gv.*Δ.t) )
    ϵ̇̄yy = SVector{2}( ε̇̄yy[:]   .+ τ̄0yy[:]    ./ (2*Gv.*Δ.t) )
    ϵ̇xy = SVector{2}( ε̇xy[:,2] .+ τ0.xy[:,2] ./ (2*Gv.*Δ.t) )

    # Stress
    τyy = @MVector zeros(2)
    τxy = @MVector zeros(2)
    for i=1:2
        τyy[i] = 𝐷.c[i][2,1] .* ϵ̇xx[i] + 𝐷.c[i][2,2] .* ϵ̇yy[i] + 𝐷.c[i][2,3] .* ϵ̇̄xy[i] + 𝐷.c[i][2,4] .* Pt[2,i]
        τxy[i] = 𝐷.v[i][3,1] .* ϵ̇̄xx[i] + 𝐷.v[i][3,2] .* ϵ̇̄yy[i] + 𝐷.v[i][3,3] .* ϵ̇xy[i] + 𝐷.v[i][3,4] .* P̄t[i]
    end

    # Residual
    fy  = ( τyy[2]  -  τyy[1] ) * invΔy
    fy += ( τxy[2]  -  τxy[1] ) * invΔx
    fy -= ( Pt[2,2] -  Pt[2,1]) * invΔy
    # fy *= -1 * Δ.x * Δ.y
    
    return fy
end

function Continuity(Vx, Vy, Pt, D, type_loc, bcv_loc, Δ)
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    return ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy)
end

function ResidualMomentum2D_x!(R, V, P, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        if type.Vx[i,j] == :in
            Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-1:i-1, jj in j-2:j-1)
            P_loc      = SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-2:i,   jj in j-2:j-1)

            Dc         = SMatrix{2,1}(𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            Dv         = SMatrix{1,2}(𝐷.v[ii,jj] for ii in i-1:i-1, jj in j-2:j-1)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)
    
            R.x[i,j]   = Momentum_x_Generic(Vx_loc, Vy_loc, P_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, τ0, λ̇, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 

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
            τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-2:i,   jj in j-2:j-1)
            
            Dc         = SMatrix{2,1}(𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            Dv         = SMatrix{1,2}(𝐷.v[ii,jj] for ii in i-1:i-1, jj in j-2:j-1)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)
            
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            autodiff(Enzyme.Reverse, Momentum_x_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
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

            # for i1=0:1
            #     if λ̇.v[i-1,j-2+i1] > 0
            #         @show "x"
            #         display(λ̇.v[i-1:i-1,j-2:j-1])
            #         printxy(num.Pt[i-1:i,j-2:j])
            #         printxy(∂R∂Pt)
            #         display(Dv[1+i1])
            #     end
            # end

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

function ResidualMomentum2D_y!(R, V, P, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        if type.Vy[i,j] == :in
            Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-2:i-1, jj in j-1:j-1) 
            P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τxx0       = SMatrix{3,2}(    τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τyy0       = SMatrix{3,2}(    τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τxy0       = SMatrix{2,3}(    τ0.xy[ii,jj] for ii in i-2:i-1,   jj in j-2:j)

            Dc         = SMatrix{1,2}(𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
            Dv         = SMatrix{2,1}(𝐷.v[ii,jj] for ii in i-2:i-1,   jj in j-1:j-1)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

            R.y[i,j]   = Momentum_y_Generic(Vx_loc, Vy_loc, P_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, τ0, λ̇, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(3,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x

        if type.Vy[i,j] == :in

            Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-2:i-1, jj in j-1:j-1) 
            P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τxx0       = SMatrix{3,2}(    τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τyy0       = SMatrix{3,2}(    τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τxy0       = SMatrix{2,3}(    τ0.xy[ii,jj] for ii in i-2:i-1,   jj in j-2:j)


            Dc         = SMatrix{1,2}(𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
            Dv         = SMatrix{2,1}(𝐷.v[ii,jj] for ii in i-2:i-1,   jj in j-1:j-1)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            autodiff(Enzyme.Reverse, Momentum_y_Generic, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
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

            # for i1=0:1
            #     if λ̇.v[i-2+i1,j-1] > 0
            #         @show "y"
            #         display(λ̇.v[i-2:i-1,j-1]')
            #         printxy(num.Pt[i-2:i,j-1:j])
            #         printxy(∂R∂Pt)
            #         display(Dv[1+i1])
            #     end
            # end

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

function ResidualContinuity2D!(R, V, P, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
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

function AssembleContinuity2D!(K, V, P, τ0, λ̇, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
                
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
        autodiff(Enzyme.Reverse, Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Const(P[i,j]), Const(D), Const(type_loc), Const(bcv_loc), Const(Δ))

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


@views function main(nc)
    #--------------------------------------------#
    # Resolution

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_Pt, iny_Pt, size_x, size_y, size_c, size_v = Ranges(nc)

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
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([1 1; 1 1; 1 1])), 
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
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = 0.5)

    # Allocations
    R       = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V       = (x=zeros(size_x...), y=zeros(size_y...))
    Vi      = (x=zeros(size_x...), y=zeros(size_y...))
    η       = (c=ones(size_c...), v=ones(size_v...) )
    λ̇       = (c=zeros(size_c...), v=zeros(size_v...) )
    ε̇       = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...) )
    τ0      = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...) )
    τ       = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_v...) )
    Pt      = zeros(size_c...)
    Pti     = zeros(size_c...)
    Dc      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)

    xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    materials = ( 
        n   = [1.0 1.0],
        η0  = [1e2 1e-1], 
        G   = [1e1 1e1],
        C   = [150 150],
        ϕ   = [30. 30.],
        ηvp = [1e0 1e0],
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

    phases.c[inx_Pt, iny_Pt][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    phases.v[(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2

    # p1 = heatmap(xc, yc, phases.c[inx_Pt,iny_Pt]', aspect_ratio=1, xlim=extrema(xc))
    # p2 = heatmap(xv, yv, phases.v', aspect_ratio=1, xlim=extrema(xc))
    # display(plot(p1, p2))
    #--------------------------------------------#

    # Time steps
    nt    = 17

    # Newton solver
    niter = 20
    ϵ_nl  = 1e-8

    # Line search
    α    = LinRange(0.05, 1.0, 10)
    rvec = zeros(length(α))

    to = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        @printf("Step %04d\n", it)
        
        err    = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy

        for iter=1:niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, type, BC, materials, phases, Δ)
                @show extrema(λ̇.c)
                @show extrema(λ̇.v)
                ResidualContinuity2D!(R, V, Pt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                ResidualMomentum2D_x!(R, V, Pt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = norm(R.p[inx_Pt,iny_Pt])/sqrt(nPt)
            max(err.x[iter], err.y[iter]) < ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            r = zeros(nVx + nVy + nPt)
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                AssembleContinuity2D!(M, V, Pt, τ0, λ̇, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, τ0, λ̇, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, τ0, λ̇, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            end

            # Stokes operator as block matrices
            𝐊  = [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  = [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ = [M.Pt.Vx M.Pt.Vy]
            𝐏  = [M.Pt.Pt;] 
            
            #--------------------------------------------#
            # Direct-iterative solver
            fu    = -r[1:size(𝐊,1)]
            fp    = -r[size(𝐊,1)+1:end]
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
            dx       = zeros(nVx + nVy + nPt)
            dx[1:size(𝐊,1)]     .= u
            dx[size(𝐊,1)+1:end] .= p

            @timeit to "Line search" begin
                Vi.x .= V.x 
                Vi.y .= V.y 
                Pti  .= Pt
                for i in eachindex(α)
                    V.x .= Vi.x 
                    V.y .= Vi.y
                    Pt  .= Pti
                    UpdateSolution!(V, Pt, α[i].*dx, number, type, nc)
                    TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, type, BC, materials, phases, Δ)
                    ResidualContinuity2D!(R, V, Pt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                    ResidualMomentum2D_x!(R, V, Pt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                    ResidualMomentum2D_y!(R, V, Pt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                    rvec[i] = norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx) + norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy) + norm(R.p[inx_Pt,iny_Pt])/sqrt(nPt)   
                end
                _, imin = findmin(rvec)
                V.x .= Vi.x 
                V.y .= Vi.y
                Pt  .= Pti
            end

            #--------------------------------------------#
            # Update solutions
            UpdateSolution!(V, Pt, α[end]*dx, number, type, nc)

        end

        # TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, type, BC, materials, phases, Δ)
        # ResidualContinuity2D!(R, V, Pt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
        # ResidualMomentum2D_x!(R, V, Pt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        # ResidualMomentum2D_y!(R, V, Pt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        # rVx = zeros(nc.x-1, nc.y)
        # rVy = zeros(nc.x, nc.y-1)
        # rVx .= diff(τ.xx[2:end-1,2:end-1], dims=1)/Δ.x + diff(τ.xy[2:end-1,:], dims=2)/Δ.y - diff(Pt[2:end-1,2:end-1], dims=1)/Δ.x
        # rVy .= diff(τ.yy[2:end-1,2:end-1], dims=2)/Δ.y + diff(τ.xy[:,2:end-1], dims=1)/Δ.x - diff(Pt[2:end-1,2:end-1], dims=2)/Δ.y
        #--------------------------------------------#

        τxyc = 0.25 .* (τ.xy[1:end-1,1:end-1] .+ τ.xy[2:end-0,1:end-1].+ τ.xy[1:end-1,2:end-0] .+ τ.xy[2:end-0,2:end-0])
        τII = sqrt.( 0.5.*(τ.xx[2:end-1,2:end-1].^2 + τ.yy[2:end-1,2:end-1].^2) .+ τxyc.^2 )
        # p1 = heatmap(xc, yv, abs.(R.y[inx_Vy,iny_Vy])', aspect_ratio=1, xlim=extrema(xc), title="Vy")
        p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vx")
        p2 = heatmap(xc, yc,  Pt[inx_Pt,iny_Pt]' .- mean(Pt[inx_Pt,iny_Pt]), aspect_ratio=1, xlim=extrema(xc), title="Pt")
        p3 = heatmap(xc, yc,  τII', aspect_ratio=1, xlim=extrema(xc), title="τII")
        p4 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright)
        p4 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        p4 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        p4 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        p5 = heatmap(xc, yc,  (λ̇.c[inx_Pt,iny_Pt] .> 0.)', aspect_ratio=1, xlim=extrema(xc), title="ηc")
        p6 = heatmap(xv, yv,  (λ̇.v .> 0.)', aspect_ratio=1, xlim=extrema(xv), title="ηv")
        display(plot(p1, p2, p3, p4, layout=(3,2)))

        # p2 = spy(M.Vx.Pt, title="x $(nnz(M.Vx.Pt))" )
        # p1 = spy(M.Vy.Pt, title="y $(nnz(M.Vy.Pt))" )
        # display(plot(p1, p2) )

    end

    display(to)
    
end

main((x = 30, y = 30))