import ForwardDiff


function residual_two_phase(x, ε̇II_eff, divVs, divqD, Pt0, Pf0, Φ0,  G, Kϕ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, ηv, ηΦ, Δt)
    eps   = -1e-13
    ηe    = G*Δt 
    ηve = inv(1/ηv + 1/ηe)
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]
    f       = τII - C*cosϕ - (Pt - Pf)*sinϕ
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = 1/Kϕ * (dPfdt - dPtdt) + 1/ηΦ * (Pf - Pt) + λ̇*sinψ*(f>=eps)
    Φ       = Φ0 + dΦdt*Δt
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks
    return @SVector([ 
        ε̇II_eff   -  τII/2/ηve - λ̇/2*(f>=eps),
        dlnρsdt   - dΦdt/(1-Φ) +   divVs,
        Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD,
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps)
    ])
end

# function DruckerPrager(τII, Pt, Pf, ηve, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δt)
#     λ̇    = 0.0
#     F    = τII - C*cosϕ - (Pt-Pf)*sinϕ - λ̇*ηvp
#     if F > -1e-10
#         λ̇    = F / (ηve + ηvp) 
#         τII -= λ̇ * ηve
#         # Pt   += comp * λ̇*sinψ*Δt/β
#         # Pf   += comp * λ̇*sinψ*Δt/β
#         F    = τII - C*cosϕ - (Pt-Pf)*sinϕ - λ̇*ηvp
#         (F>1e-10) && error("Failed return mapping")
#         # (τII<0.0) && error("Plasticity without condom")
#     end
#     return τII, Pt, Pf, λ̇
# end

function StrainRateTrial(τII, Pt, Pf, ηve, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δt)
    ε̇II_trial = τII/2/ηve
    return ε̇II_trial
end

function LocalRheology(ε̇, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ)

    # Effective strain rate & pressure
    ε̇II  = sqrt.( (ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1]-ε̇[2])^2)/2 + ε̇[3]^2 ) #+ 1e-14
    Pt   = ε̇[4]
    Pf   = ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[phases]
    η0   = materials.ηs0[phases]
    # B    = materials.B[phases]
    G    = materials.G[phases]
    C    = materials.C[phases]
    ηΦ   = materials.ηϕ[phases]
    KΦ   = materials.Kϕ[phases]
    Ks   = materials.Ks[phases]
    Kf   = materials.Kf[phases]

    ηvp  = materials.ηvp[phases]
    sinψ = materials.sinψ[phases]    
    sinϕ = materials.sinϕ[phases] 
    cosϕ = materials.cosϕ[phases]    

    # β    = materials.β[phases]
    # comp = materials.compressible

    # Initial guess
    η    = (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
    ηvep = inv(1/η + 1/(G*Δ.t))
    # ηvep = G*Δ.t

    τII  = 2*ηvep*ε̇II

    # # Visco-elastic powerlaw
    # for iter=1:20
    #     r      = ε̇II - StrainRateTrial(τII, Pt, Pf, ηvep, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δ.t)
    #     (abs(r)<ϵ) && break
    #     ∂ε̇II∂τII = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial, τII, Pt, Pf, ηvep, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δ.t)
    #     ∂τII∂ε̇II = inv(∂ε̇II∂τII[1])
    #     τII     += ∂τII∂ε̇II*r
    # end
    # isnan(τII) && error()
#  
    # Viscoplastic return mapping
    λ̇ = 0.

    # if materials.plasticity === :DruckerPrager
        # τII, Pt, Pf, λ̇ = DruckerPrager(τII, Pt, Pf, ηvep, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δ.t)
    # # elseif materials.plasticity === :tensile
    # #     τII, P, λ̇ = Tensile(τII, P, ηvep, comp, β, Δ.t, materials.σT[phases], ηvp)
    # # elseif materials.plasticity === :Kiss2023
    # #     τII, P, λ̇ = Kiss2023(τII, P, ηvep, comp, β, Δ.t, C, ϕ, ψ, ηvp, materials.σT[phases], materials.δσT[phases], materials.P1[phases], materials.τ1[phases], materials.P2[phases], materials.τ2[phases])
    # end

    x = @MVector( [τII, Pt, Pf, λ̇] )
    J = @MMatrix( zeros(4,4))

    eps = 1e-8

    f0 = residual_two_phase(x, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)


    for iter=1:10
        # p =  @SVector([eps*τII, eps*Pt, eps*Pf, eps*λ̇+eps])
                p =  @SVector([eps, eps, eps, eps])

        f = residual_two_phase(x, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        
        x1 = @SVector( [τII-p[1], Pt, Pf, λ̇] )
        J1m = residual_two_phase(x1, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        x1 = @SVector( [τII+p[1], Pt, Pf, λ̇] )
        J1p = residual_two_phase(x1, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        x2 = @SVector( [τII, Pt-p[2], Pf, λ̇] )
        J2m = residual_two_phase(x2, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        x2 = @SVector( [τII, Pt+p[2], Pf, λ̇] )
        J2p = residual_two_phase(x2, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        x3 = @SVector( [τII, Pt, Pf-p[3], λ̇] )
        J3m = residual_two_phase(x3, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        x3 = @SVector( [τII, Pt, Pf+p[3], λ̇] )
        J3p = residual_two_phase(x3, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        x4 = @SVector( [τII, Pt, Pf, λ̇-p[4]] )
        J4m = residual_two_phase(x4, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        x4 = @SVector( [τII, Pt, Pf, λ̇+p[4]] )
        J4p = residual_two_phase(x4, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        J[1,:] .= (J1p .- J1m)/2/p[1]
        J[2,:] .= (J2p .- J2m)/2/p[2]
        J[3,:] .= (J3p .- J3m)/2/p[3]
        J[4,:] .= (J4p .- J4m)/2/p[4]
        # J[1,:] .= (J1p .- f)/p[1]
        # J[2,:] .= (J2p .- f)/p[2]
        # J[3,:] .= (J3p .- f)/p[3]
        # J[4,:] .= (J4p .- f)/p[4]
        # J = Enzyme.jacobian(Enzyme.Forward, residual_two_phase, x, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)

        # # f_closed = (x) -> residual_two_phase(x, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        # # J = ForwardDiff.jacobian(f_closed, x)
        x .= x .- inv(J')*f
        
        # @show iter, norm(f)#, x# J

        if norm(f)<1e-10
            break
        end
    end

    τII, λ̇ = x[1], x[4]

    # Effective viscosity
    ηvep = τII/(2*ε̇II)

    return ηvep, λ̇, Pt, Pf
end

function StressVector!(ε̇, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ) 
    η, λ̇, Pt, Pf = LocalRheology(ε̇, divVs, divqD, Pt0, Pf0, Φ0, materials, phases, Δ)
    τ            = @SVector([2 * η * ε̇[1],
                             2 * η * ε̇[2],
                             2 * η * ε̇[3],
                                       Pt,
                                       Pf,])
    return τ, η, λ̇
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, P, ΔP, P0, Φ0, type, BC, materials, phases, Δ)

    _ones = @SVector ones(5)

    # Loop over centroids
    for j=2:size(ε̇.xx,2)-1, i=2:size(ε̇.xx,1)-1
 
        Vx     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1,   jj in j:j+2)
        Vy     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2,   jj in j:j+1)
        bcx    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        bcy    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
        typex  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        typey  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
        τxy0   = SMatrix{2,2}(    τ0.xy[ii,jj] for ii in i:i+1,   jj in j:j+1)

        Vx = SetBCVx1(Vx, typex, bcx, Δ)
        Vy = SetBCVy1(Vy, typey, bcy, Δ)

        Dxx = ∂x_inn(Vx) / Δ.x 
        Dyy = ∂y_inn(Vy) / Δ.y 
        Dxy = ∂y(Vx) / Δ.y
        Dyx = ∂x(Vy) / Δ.x
        
        Dkk = Dxx .+ Dyy
        ε̇xx = @. Dxx - Dkk ./ 3
        ε̇yy = @. Dyy - Dkk ./ 3
        ε̇xy = @. (Dxy + Dyx) ./ 2
        ε̇̄xy = av(ε̇xy)

        divqD = 0.0
    
        # Visco-elasticity
        G     = materials.G[phases.c[i,j]]
        τ̄xy0  = av(τxy0)
        ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), P.t[i,j], P.f[i,j]])

        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(Dkk[1]), Const(divqD), Const(P0.t[i,j]), Const(P0.f[i,j]), Const(Φ0.c[i,j]), Const(materials), Const(phases.c[i,j]), Const(Δ))
        
        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        @views 𝐷_ctl.c[i,j][:,1] .= jac.derivs[1][1][1]
        @views 𝐷_ctl.c[i,j][:,2] .= jac.derivs[1][2][1]
        @views 𝐷_ctl.c[i,j][:,3] .= jac.derivs[1][3][1]
        @views 𝐷_ctl.c[i,j][:,4] .= jac.derivs[1][4][1]
        @views 𝐷_ctl.c[i,j][:,5] .= jac.derivs[1][5][1]

        # Tangent operator used for Picard Linearisation
        𝐷.c[i,j] .= diagm(2*jac.val[2] * _ones)
        𝐷.c[i,j][4,4] = 1
        𝐷.c[i,j][5,5] = 1

        # Update stress
        τ.xx[i,j] = jac.val[1][1]
        τ.yy[i,j] = jac.val[1][2]
        ε̇.xx[i,j] = ε̇xx[1]
        ε̇.yy[i,j] = ε̇yy[1]
        λ̇.c[i,j]  = jac.val[3]
        η.c[i,j]  = jac.val[2]
        ΔP.t[i,j] = (jac.val[1][4] - P.t[i,j])
        ΔP.t[i,j] = (jac.val[1][5] - P.f[i,j])
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
        Pt     = SMatrix{2,2}(      P.t[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Pf     = SMatrix{2,2}(      P.f[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Φ0_loc = SMatrix{2,2}(     Φ0.c[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Pt0_loc = SMatrix{2,2}(     P0.t[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Pf0_loc = SMatrix{2,2}(     P0.f[ii,jj] for ii in i:i+1,   jj in j:j+1)
 
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

        divqD = 0.0
        
        # Visco-elasticity
        G     = materials.G[phases.v[i+1,j+1]]
        τ̄xx0  = av(τxx0)
        τ̄yy0  = av(τyy0)
        P̄t    = av(  Pt)
        P̄f    = av(  Pf)
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t), P̄t[1], P̄f[1]])
        D̄kk   = av( Dkk)
        ϕ̄0    = av(Φ0_loc)
        P̄t0   = av(Pt0_loc)
        P̄f0   = av(Pf0_loc)

        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(D̄kk[1]), Const(divqD), Const(P̄t0[1]), Const(P̄f0[1]), Const(ϕ̄0[1]), Const(materials), Const(phases.v[i+1,j+1]), Const(Δ))

        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        @views 𝐷_ctl.v[i+1,j+1][:,1] .= jac.derivs[1][1][1]
        @views 𝐷_ctl.v[i+1,j+1][:,2] .= jac.derivs[1][2][1]
        @views 𝐷_ctl.v[i+1,j+1][:,3] .= jac.derivs[1][3][1]
        @views 𝐷_ctl.v[i+1,j+1][:,4] .= jac.derivs[1][4][1]
        @views 𝐷_ctl.v[i+1,j+1][:,5] .= jac.derivs[1][5][1]

        # Tangent operator used for Picard Linearisation
        𝐷.v[i+1,j+1] .= diagm(2*jac.val[2] * _ones)
        𝐷.v[i+1,j+1][4,4] = 1
        𝐷.v[i+1,j+1][5,5] = 1

        # Update stress
        τ.xy[i+1,j+1] = jac.val[1][3]
        ε̇.xy[i+1,j+1] = ε̇xy[1]
        λ̇.v[i+1,j+1]  = jac.val[3]
        η.v[i+1,j+1]  = jac.val[2]
    end
end