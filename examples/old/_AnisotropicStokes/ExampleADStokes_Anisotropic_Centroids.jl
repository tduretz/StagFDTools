using StagFDTools.Stokes, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays
import Statistics:mean
using DifferentiationInterface
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian
#################################################
# Rheology only on centroids and interpolations #
#################################################
struct BoundaryConditions{Tx,Ty,Tp,Txy}
    Vx::Tx
    Vy::Ty
    Pt::Tp
    xy::Txy
end

function Base.getindex(x::BoundaryConditions, i::Int64)
    @assert 0 < i < 4 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
    i == 4 && return x.xy
end

function Momentum_x(Vx, Vy, Pt, η, type, bcv, Δ)
    
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y

    in_center = type.p == :in
    
    SetBCVx!(Vx, type.x, bcv, Δ)

    Dxx = (Vx[2:end,:] - Vx[1:end-1,:]) * invΔx             # Static Arrays ???
    Dyy = (Vy[:,2:end] - Vy[:,1:end-1]) * invΔy             
    Dkk = Dxx[:,2:end-1] + Dyy[2:end-1,:]

    Dxy = (Vx[:,2:end] - Vx[:,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,:] - Vy[1:end-1,:]) * invΔx 

    ε̇xx = Dxx[:,2:end-1] - 1/3*Dkk
    ε̇yy = Dyy[2:end-1,:] - 1/3*Dkk

    Dx̄ȳ =              1/4*(Dxy[1:end-1,1:end-1] + Dxy[2:end-0,1:end-1] + Dxy[1:end-1,2:end-0] + Dxy[2:end-0,2:end-0])
    Dȳx̄ = in_center .* 1/4*(Dyx[1:end-1,1:end-1] + Dyx[2:end-0,1:end-1] + Dyx[1:end-1,2:end-0] + Dyx[2:end-0,2:end-0])
    ε̇x̄ȳ = 1/2*(Dx̄ȳ + Dȳx̄)

    τxx = 2 * η.c .* ε̇xx 
    τyy = 2 * η.c .* ε̇yy     
    τx̄ȳ = 2 * η.c .* ε̇x̄ȳ 
    τxy = 1/4*(τx̄ȳ[1:end-1,1:end-1] + τx̄ȳ[2:end-0,1:end-1] + τx̄ȳ[1:end-1,2:end-0] + τx̄ȳ[2:end-0,2:end-0])

    # Regular stencil
    # τxy = 2 * η.xy .* ε̇xy[2:2,2:end-1] # dodgy broadcast

    fx  = (τxx[2,2] - τxx[1,2]) * invΔx 
    fx += (τxy[1,2] - τxy[1,1]) * invΔy
    fx -= ( Pt[2,2] -  Pt[1,2]) * invΔx
    fx *= -1 * Δ.x * Δ.y

    return fx
end

function Momentum_y(Vx, Vy, Pt, η, type, bcv, Δ)
    
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y

    in_center = type.p == :in
    
    SetBCVy!(Vy, type.y, bcv, Δ)

    Dxx = (Vx[2:end,:] - Vx[1:end-1,:]) * invΔx  # Static Arrays ???
    Dyy = (Vy[:,2:end] - Vy[:,1:end-1]) * invΔy             
    Dkk = Dxx[:,2:end-1] + Dyy[2:end-1,:]

    Dxy = (Vx[:,2:end] - Vx[:,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,:] - Vy[1:end-1,:]) * invΔx 

    ε̇xx = Dxx[:,2:end-1] - 1/3*Dkk
    ε̇yy = Dyy[2:end-1,:] - 1/3*Dkk

    Dx̄ȳ = in_center .* 1/4*(Dxy[1:end-1,1:end-1] + Dxy[2:end-0,1:end-1] + Dxy[1:end-1,2:end-0] + Dxy[2:end-0,2:end-0])
    Dȳx̄ =              1/4*(Dyx[1:end-1,1:end-1] + Dyx[2:end-0,1:end-1] + Dyx[1:end-1,2:end-0] + Dyx[2:end-0,2:end-0])
    ε̇x̄ȳ = 1/2*(Dx̄ȳ + Dȳx̄)

    τxx = 2 * η.c .* ε̇xx
    τyy = 2 * η.c .* ε̇yy
    τx̄ȳ = 2 * η.c .* ε̇x̄ȳ 

    τxy = 1/4*(τx̄ȳ[1:end-1,1:end-1] + τx̄ȳ[2:end-0,1:end-1] + τx̄ȳ[1:end-1,2:end-0] + τx̄ȳ[2:end-0,2:end-0])
    
    # Regular stencil
    # τxy = 2 * η.xy .* ε̇xy[2:end-1,2:2]

    fy  =  (τyy[2,2] - τyy[2,1]) * invΔy
    fy += (τxy[2,1] - τxy[1,1]) * invΔx
    fy -= ( Pt[2,2] -  Pt[2,1]) * invΔy
    fy *= -1 * Δ.x * Δ.y
    
    return fy
end

function Continuity(Vx, Vy, Pt, η_loc, type_loc, bcv_loc, Δ)
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    fp = ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy) 
    return fp
end

function ResidualMomentum2D_x!(R, V, P, η, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = MMatrix{3,5}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-2:j+2)
        Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)

        typex_loc  = SMatrix{3,5}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-2:j+2)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        ηx_loc     = SMatrix{3,5}(      η.x[ii,jj] for ii in i-1:i+1, jj in j-2:j+2)
        ηy_loc     = SMatrix{4,4}(      η.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        ηc_loc     = SMatrix{2,3}(      η.p[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        ηv_loc     = SMatrix{1,2}(     η.xy[ii,jj] for ii in i-1:i-1, jj in j-2:j-1)

        tp         = SMatrix{2,3}(  type.Pt[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        P_loc      = SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        txy        = SMatrix{1,2}(  type.xy[ii,jj] for ii in i-1:i-1, jj in j-2:j-1)

        Vx_BC      = SMatrix{3,2}(BC.Vx[i-1:i+1,:])
        ∂Vx∂x_BC   = SMatrix{2,5}(BC.∂Vx∂x[:,j-2:j+2])
        ∂Vx∂y_BC   = SMatrix{3,2}(BC.∂Vx∂y[i-1:i+1,:])
        bcv_loc    = (Vx_BC=Vx_BC, ∂Vx∂x_BC=∂Vx∂x_BC, ∂Vx∂y_BC=∂Vx∂y_BC)

        η_loc      = (x=ηx_loc, y=ηy_loc, c=ηc_loc, xy=ηv_loc)
        type_loc   = (x=typex_loc, y=typey_loc, xy=txy, p=tp)
        if type.Vx[i,j] == :in
            R.x[i,j]   = Momentum_x(Vx_loc, Vy_loc, P_loc, η_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, η, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,5)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        
        if type.Vx[i,j] == :in

            typex_loc  = SMatrix{3,5}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-2:j+2)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            ηx_loc     = SMatrix{3,5}(      η.x[ii,jj] for ii in i-1:i+1, jj in j-2:j+2)
            ηy_loc     = SMatrix{4,4}(      η.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            ηc_loc     = SMatrix{2,3}(      η.p[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            ηv_loc     = SMatrix{1,2}(     η.xy[ii,jj] for ii in i-1:i-1, jj in j-2:j-1)

 
            Vx_loc     = MMatrix{3,5}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-2:j+2)
            Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            P_loc      = MMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            tp         = SMatrix{2,3}(  type.Pt[ii,jj] for ii in i-1:i,   jj in j-2:j  )

            txy        = SMatrix{1,2}(  type.xy[ii,jj] for ii in i-1:i-1, jj in j-2:j-1)
            η_loc      = (x=ηx_loc, y=ηy_loc, c=ηc_loc, xy=ηv_loc)
            type_loc   = (x=typex_loc, y=typey_loc, xy=txy, p=tp)
            

            Vx_BC      = SMatrix{3,2}(BC.Vx[i-1:i+1,:])
            ∂Vx∂x_BC   = SMatrix{2,5}(BC.∂Vx∂x[:,j-2:j+2])
            ∂Vx∂y_BC   = SMatrix{3,2}(BC.∂Vx∂y[i-1:i+1,:])
            bcv_loc    = (Vx_BC=Vx_BC, ∂Vx∂x_BC=∂Vx∂x_BC, ∂Vx∂y_BC=∂Vx∂y_BC)
    
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            forwarddiff_gradients!(Momentum_x, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(η_loc), Const(type_loc), Const(bcv_loc), Const(Δ))
            # Vx --- Vx
            Local = num.Vx[i-1:i+1,j-2:j+2] .* pattern[1][1]
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

function ResidualMomentum2D_y!(R, V, P, η, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{5,3}(      V.y[ii,jj] for ii in i-2:i+2, jj in j-1:j+1)
    
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{5,3}(  type.Vy[ii,jj] for ii in i-2:i+2, jj in j-1:j+1)
        ηx_loc     = SMatrix{4,4}(      η.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        ηy_loc     = SMatrix{5,3}(      η.y[ii,jj] for ii in i-2:i+2, jj in j-1:j+1)
        P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        tp         = SMatrix{3,2}(  type.Pt[ii,jj] for ii in i-2:i,   jj in j-1:j  )

        ηc_loc     = SMatrix{3,2}(      η.p[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        ηv_loc     = SMatrix{2,1}(     η.xy[ii,jj] for ii in i-2:i-1, jj in j-1:j-1)

        txy        = SMatrix{2,1}(  type.xy[ii,jj] for ii in i-2:i-1, jj in j-1:j-1)
        η_loc      = (x=ηx_loc, y=ηy_loc, c=ηc_loc, xy=ηv_loc)
        type_loc   = (x=typex_loc, y=typey_loc, xy=txy, p=tp)

        Vy_BC      = SMatrix{2,3}(BC.Vy[:,j-1:j+1])
        ∂Vy∂y_BC   = SMatrix{5,2}(BC.∂Vy∂y[i-2:i+2,:])
        ∂Vy∂x_BC   = SMatrix{2,3}(BC.∂Vy∂x[:,j-1:j+1])
        bcv_loc    = (Vy_BC=Vy_BC, ∂Vy∂y_BC=∂Vy∂y_BC, ∂Vy∂x_BC=∂Vy∂x_BC)

        if type.Vy[i,j] == :in
            R.y[i,j]   = Momentum_y(Vx_loc, Vy_loc, P_loc, η_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, η, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(5,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x

        # ηx_loc     = SMatrix{3,3}(      η.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        # ηy_loc     = SMatrix{4,4}(      η.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        # P_loc      = MMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )

        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{5,3}(      V.y[ii,jj] for ii in i-2:i+2, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{5,3}(  type.Vy[ii,jj] for ii in i-2:i+2, jj in j-1:j+1)
        ηx_loc     = SMatrix{4,4}(      η.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        ηy_loc     = SMatrix{5,3}(      η.y[ii,jj] for ii in i-2:i+2, jj in j-1:j+1)
        P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        tp         = SMatrix{3,2}(  type.Pt[ii,jj] for ii in i-2:i,   jj in j-1:j  )

        ηc_loc     = SMatrix{3,2}(      η.p[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        ηv_loc     = SMatrix{2,1}(     η.xy[ii,jj] for ii in i-2:i-1, jj in j-1:j-1)

        txy        = SMatrix{2,1}(  type.xy[ii,jj] for ii in i-2:i-1, jj in j-1:j-1)
        η_loc      = (x=ηx_loc, y=ηy_loc, c=ηc_loc, xy=ηv_loc)
        type_loc   = (x=typex_loc, y=typey_loc, xy=txy, p=tp)

        Vy_BC      = SMatrix{2,3}(BC.Vy[:,j-1:j+1])
        ∂Vy∂y_BC   = SMatrix{5,2}(BC.∂Vy∂y[i-2:i+2,:])
        ∂Vy∂x_BC   = SMatrix{2,3}(BC.∂Vy∂x[:,j-1:j+1])
        bcv_loc    = (Vy_BC=Vy_BC, ∂Vy∂y_BC=∂Vy∂y_BC, ∂Vy∂x_BC=∂Vy∂x_BC)


        if type.Vy[i,j] == :in
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            forwarddiff_gradients!(Momentum_y, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(η_loc), Const(type_loc), Const(bcv_loc), Const(Δ))
            # Vy --- Vx
            Local = num.Vx[i-2:i+1,j-1:j+2] .* pattern[2][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][1][num.Vy[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vy --- Vy
            Local = num.Vy[i-2:i+2,j-1:j+1] .* pattern[2][2]
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

function ResidualContinuity2D!(R, V, P, η, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    # (; bc_val, type, pattern, num) = numbering
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        typex_loc  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        typey_loc  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        η_loc      =   SVector{4}([η.y[i+1,j] η.x[i,j+1] η.x[i+1,j+1] η.y[i+1,j+1]] )
        bcv_loc    = (;)
        type_loc   = (x=typex_loc, y=typey_loc)
        R.p[i,j]     = Continuity(Vx_loc, Vy_loc, P[i,j], η_loc, type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, η, num, pattern, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    # (; bc_val, type, pattern, num) = numbering
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        
        typex_loc  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        typey_loc  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        η_loc      =   SVector{4}([η.y[i+1,j] η.x[i,j+1] η.x[i+1,j+1] η.y[i+1,j+1]] )
        bcv_loc    = (;)
        type_loc   = (x=typex_loc, y=typey_loc)
        
        ∂R∂Vx .= 0.
        ∂R∂Vy .= 0.
        forwarddiff_gradients!(Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Const(P[i,j]), Const(η_loc), Const(type_loc), Const(bcv_loc), Const(Δ))

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

let  
    #--------------------------------------------#
    # Resolution
    nc = (x = 40, y = 40)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, size_x, size_y, size_c = Ranges(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = BoundaryConditions(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
        fill(:out, (nc.x+1, nc.y+1)),
    )

    type.xy                  .= :τxy 
    type.xy[2:end-1,2:end-1] .= :in 

    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx] .= :in       
    type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
    type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
    type.Vx[inx_Vx,2]       .= :Neumann
    type.Vx[inx_Vx,end-1]   .= :Neumann
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy] .= :in       
    type.Vy[2,iny_Vy]       .= :Neumann
    type.Vy[end-1,iny_Vy]   .= :Neumann
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
        Fields(@SMatrix([1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1]),     @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([0 1 0; 0 1 0])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 0; 1 1; 0 0])), 
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
    xmin, xmax = -1/2, 1/2
    ymin, ymax = -1/2, 1/2
    L  = (x=xmax-xmin, y=ymax-ymin)
    Δ  = (x=L.x/nc.x, y=L.y/nc.y)
    R  = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V  = (x=zeros(size_x...), y=zeros(size_y...))
    η  = (x= ones(size_x...), y= ones(size_y...), p=ones(size_c...), xy=ones(nc.x+1, nc.y+1)  )
    Rp = zeros(size_c...)
    Pt = zeros(size_c...)
    xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)
    xvy = LinRange(-L.x/2-3Δ.x/2, L.x/2+3Δ.x/2, nc.x+4)
    yvy = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    yvx = LinRange(-L.y/2-3Δ.y/2, L.y/2+3Δ.y/2, nc.y+4)

    # Initial configuration
    # Pure Shear
    D_BC = [-1  0;
             0  1]
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    BC = (
        Vx    = zeros(size_x[1], 2),
        Vy    = zeros(2, size_y[2]),
        ∂Vx∂x = zeros(2, size_x[2]),
        ∂Vy∂y = zeros(size_y[1], 2),
        ∂Vx∂y = zeros(size_x[1], 2),
        ∂Vy∂x = zeros(2, size_y[2]),
    )
    BC.Vx[inx_Vx,1] .= xv .* D_BC[1,1] .+ ymin .* D_BC[1,2]
    BC.Vx[inx_Vx,2] .= xv .* D_BC[1,1] .+ ymax .* D_BC[1,2]
    BC.Vy[1,iny_Vy] .= yv .* D_BC[2,2] .+ xmin .* D_BC[2,1]
    BC.Vy[2,iny_Vy] .= yv .* D_BC[2,2] .+ xmax .* D_BC[2,1]
    BC.∂Vx∂x[1,:] .= D_BC[1,1]
    BC.∂Vx∂x[2,:] .= D_BC[1,1]
    BC.∂Vx∂y[:,1] .= D_BC[1,2]
    BC.∂Vx∂y[:,2] .= D_BC[1,2]
    BC.∂Vy∂x[1,:] .= D_BC[2,1]
    BC.∂Vy∂x[2,:] .= D_BC[2,1]
    BC.∂Vy∂y[:,1] .= D_BC[2,2]
    BC.∂Vy∂y[:,2] .= D_BC[2,2]

    η.x .= 1e0
    η.y .= 1e0
    η.x[(xvx.^2 .+ (yvx').^2) .<= 0.1^2] .= 1e2 
    η.y[(xvy.^2 .+ (yvy').^2) .<= 0.1^2] .= 1e2
    η.p  .= 0.25.*(η.x[1:end-1,2:end-1].+η.x[2:end-0,2:end-1].+η.y[2:end-1,1:end-1].+η.y[2:end-1,2:end-0])
    η.xy .= 0.25.*(η.p[1:end-1,1:end-1] .+ η.p[1:end-1,2:end-0] + η.p[2:end-0,1:end-1] .+ η.p[2:end-0,2:end-0] )

    # #--------------------------------------------#

    # for it=1:2

    # Residual check
    ResidualContinuity2D!(R,  V, Pt, η, number, type, BC, nc, Δ) 
    ResidualMomentum2D_x!(R,  V, Pt, η, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R,  V, Pt, η, number, type, BC, nc, Δ)

    @info "Residuals"
    @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
    @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
    @show norm(Rp[inx_c,iny_c])/sqrt(nPt)

    # # printxy(type.Vx)
    # # printxy(type.Pt)
    # # printxy(number.Vx)
    # # printxy(number.Vy)

    # Set global residual vector
    r = zeros(nVx + nVy + nPt)
    SetRHS!(r, R, number, type, nc)

    # #--------------------------------------------#
    # Assembly
    @info "Assembly, ndof  = $(nVx + nVy + nPt)"
    AssembleContinuity2D!(M, V, Pt, η, number, pattern, type, BC, nc, Δ)
    AssembleMomentum2D_x!(M, V, Pt, η, number, pattern, type, BC, nc, Δ)
    AssembleMomentum2D_y!(M, V, Pt, η, number, pattern, type, BC, nc, Δ)

    # Stokes operator as block matrices
    K  = [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
    Q  = [M.Vx.Pt; M.Vy.Pt]
    Qᵀ = [M.Pt.Vx M.Pt.Vy]
    𝑀 = [K Q; Qᵀ M.Pt.Pt]

    @info "Velocity block symmetry"
    Kdiff = K - K'
    dropzeros!(Kdiff)
    @show norm(Kdiff)
    @show extrema(Kdiff)
    
    #--------------------------------------------#
    # Direct solver
    dx = - 𝑀 \ r


    #--------------------------------------------#
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # end

    #--------------------------------------------#
    # Residual check
    ResidualContinuity2D!(R,  V, Pt, η, number, type, BC, nc, Δ) 
    ResidualMomentum2D_x!(R,  V, Pt, η, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R,  V, Pt, η, number, type, BC, nc, Δ)

    #--------------------------------------------#

    # p1 = heatmap(xv, yc, R.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc))
    # p2 = heatmap(xc, yv, R.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc))
    # p3 = heatmap(xc, yc, R.p[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc))
    # display(plot(p1, p2, p3))
    
    p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc))
    p2 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc))
    p3 = heatmap(xc, yc,  Pt[inx_c,iny_c]' .- mean(Pt[inx_c,iny_c]), aspect_ratio=1, xlim=extrema(xc))
    display(plot(p1, p2, p3))
    
    #--------------------------------------------#
    # Kdiff = K - K'
    # dropzeros!(Kdiff)
    # f = GLMakie.spy(rotr90(Kdiff))
    # GLMakie.DataInspector(f)
    # display(f)
end
