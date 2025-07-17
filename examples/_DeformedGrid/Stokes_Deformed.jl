###################################################################################
###################################################################################
###################################################################################

function Continuity_Def(Vx_loc, Vy_loc, Pt, Pt0, D, J, phase, materials, type_loc, bcv_loc, Δ)
    invΔx = 1 / Δ.ξ
    invΔy = 1 / Δ.η
    invΔt = 1 / Δ.t
    # BC
    Vx    = SetBCVx1(Vx_loc, type_loc.x, bcv_loc.x, Δ)
    Vy    = SetBCVy1(Vy_loc, type_loc.y, bcv_loc.y, Δ)
    V̄x    = av(Vx)
    V̄y    = av(Vy)
    β     = materials.β[phase]
    η     = materials.β[phase]
    comp  = materials.compressible
    ∂Vx∂x = (Vx[2,2] - Vx[1,2]) * invΔx * J[1,1][1,1] + (V̄x[1,2] - V̄x[1,1]) * invΔy * J[1,1][1,2]
    ∂Vy∂y = (V̄y[2,1] - V̄y[1,1]) * invΔx * J[1,1][2,1] + (Vy[2,2] - Vy[2,1]) * invΔy * J[1,1][2,2] 
    f     =  (∂Vx∂x + ∂Vy∂y) + comp * β * (Pt[1] - Pt0) * invΔt #+ 1/(1000*η)*Pt[1]
    f    *= max(invΔx, invΔy)
    return f
end

function ResidualContinuity2D_Def!(R, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ) 
                
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
            R.p[i,j]   = Continuity_Def(Vx_loc, Vy_loc, P[i,j], P0[i,j], D, Jinv_c, phases.c[i,j], materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleContinuity2D_Def!(K, V, P, Pt0, ΔP, τ0, 𝐷, Jinv, phases, materials, num, pattern, type, BC, nc, Δ) 
                
    ∂R∂Vx = @MMatrix zeros(2,3)
    ∂R∂Vy = @MMatrix zeros(3,2)
    ∂R∂P  = @MMatrix zeros(1,1)
    
    Vx_loc= @MMatrix zeros(2,3)
    Vy_loc= @MMatrix zeros(3,2)
    P_loc = @MMatrix zeros(1,1)

    for j in 2:size(P, 2)-1, i in 2:size(P, 1)-1
        Vx_loc     = MMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc     = MMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
        P_loc     .= SMatrix{1,1}(        P[ii,jj] for ii in i:i,   jj in j:j  )
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
        autodiff(Enzyme.Reverse, Continuity_Def, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂P), Const(Pt0[i,j]), Const(D), Const(Jinv_c), Const(phases.c[i,j]), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))

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
end

###################################################################################
###################################################################################
###################################################################################