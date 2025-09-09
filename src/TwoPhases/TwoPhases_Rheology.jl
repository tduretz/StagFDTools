import ForwardDiff


function residual_two_phase(x, ε̇II_eff, divVs, divqD, Pt0, Pf0, Φ0,  G, Kϕ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, ηv, ηΦ, Δt)
    eps   = -1e-13
    ηe    = G*Δt 
    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]
    f       = τII - (1-Φ)*C*cosϕ - (Pt - Pf)*sinϕ
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = (dPfdt - dPtdt)/Kϕ + (Pf - Pt)/ηΦ + λ̇*sinψ*(f>=eps)
    dlnρfdt = dPfdt / Kf
    dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks
    ηve     = (1-Φ)*inv(1/ηv + 1/ηe)
    return @SVector([ 
        ε̇II_eff   -  τII/2/ηve - λ̇/2*(f>=eps),
        dlnρsdt   - dΦdt/(1-Φ) +   divVs,
        Φ*dlnρfdt + dΦdt       + Φ*divVs + divqD,
        (f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps),
        Φ  - (Φ0 + dΦdt*Δt)
    ])
end

function StrainRateTrial(τII, Pt, Pf, ηve, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δt)
    ε̇II_trial = τII/2/ηve
    return ε̇II_trial
end

function LocalRheology(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)

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
    # η    = (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
    # ηvep = inv(1/η + 1/(G*Δ.t))
    # τII  = 2*ηvep*ε̇II

    # To be removed?
    τII = sqrt.( (τ0[1]^2 + τ0[2]^2 + (-τ0[1]-τ0[2])^2)/2 + τ0[3]^2 )

    # # Visco-elastic powerlaw
    # for iter=1:20
    #     r      = ε̇II - StrainRateTrial(τII, Pt, Pf, ηvep, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δ.t)
    #     (abs(r)<ϵ) && break
    #     ∂ε̇II∂τII = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial, τII, Pt, Pf, ηvep, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, Δ.t)
    #     ∂τII∂ε̇II = inv(∂ε̇II∂τII[1])
    #     τII     += ∂τII∂ε̇II*r
    # end
    # isnan(τII) && error()

    # Viscoplastic return mapping
    λ̇ = 0.

    x = @MVector( [τII, Pt, Pf, λ̇, Φ0] )

    ri  = residual_two_phase( x, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
    r0  = norm(ri)
    tol = 1e-7

    for iter=1:10
        J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, residual_two_phase, x, ε̇II, divVs, divqD, Pt0, Pf0, Φ0,  G, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, η0, ηΦ, Δ.t)
        f = J.val
        x .= x .- inv(J.derivs[1])*f
        # @show iter, f
        if norm(f)<tol || norm(f)/r0<tol
            break
        end
    end

    # if ri[4]>tol
    #     @show x
    #     error()
    # end

    τII, Pt, Pf, λ̇, Φ = x[1], x[2], x[3], x[4], x[5]

    # Effective viscosity
    ηvep = τII/(2*ε̇II)

    return ηvep, λ̇, Pt, Pf, Φ
end

function StressVector!(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ) 
    η, λ̇, Pt, Pf = LocalRheology(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)
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
        Pf_loc = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typepf = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcpf   = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        # BCs
        Vx  = SetBCVx1(Vx, typex, bcx, Δ)
        Vy  = SetBCVy1(Vy, typey, bcy, Δ)
        Pf  = SetBCPf1(Pf_loc, typepf, bcpf, Δ)

        # Kinematics
        Dxx = ∂x_inn(Vx) / Δ.x 
        Dyy = ∂y_inn(Vy) / Δ.y 
        Dxy = ∂y(Vx) / Δ.y
        Dyx = ∂x(Vy) / Δ.x
        
        Dkk = Dxx .+ Dyy
        ε̇xx = @. Dxx - Dkk ./ 3
        ε̇yy = @. Dyy - Dkk ./ 3
        ε̇xy = @. (Dxy + Dyx) ./ 2
        ε̇̄xy = av(ε̇xy)

        qDx   = materials.k_ηf0[1] .*  ∂x_inn(Pf) / Δ.x 
        qDy   = materials.k_ηf0[1] .*  ∂y_inn(Pf) / Δ.y
        divqD = 0.0 #∂x(qDx) + ∂y(qDy)
       
        # Visco-elasticity
        G     = materials.G[phases.c[i,j]]
        τ̄xy0  = av(τxy0)
        ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), P.t[i,j], P.f[i,j]])
        τ0_loc  = @SVector([τ0.xx[i,j], τ0.yy[i,j], τ̄xy0[1]])

        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(Dkk[1]), Const(divqD), Const(P0.t[i,j]), Const(P0.f[i,j]), Const(Φ0.c[i,j]), Const(τ0_loc), Const(materials), Const(phases.c[i,j]), Const(Δ))
        
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
        ΔP.f[i,j] = (jac.val[1][5] - P.f[i,j])
    end

    # Loop over vertices
    for j=1:size(ε̇.xy,2)-2, i=1:size(ε̇.xy,1)-2
        Vx      = SMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        Vy      = SMatrix{2,3}(      V.y[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        bcx     = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        bcy     = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        typex   = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        typey   = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        τxx0    = SMatrix{2,2}(    τ0.xx[ii,jj] for ii in i:i+1,   jj in j:j+1)
        τyy0    = SMatrix{2,2}(    τ0.yy[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Pt      = SMatrix{2,2}(      P.t[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Pf      = SMatrix{2,2}(      P.f[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Φ0_loc  = SMatrix{2,2}(     Φ0.c[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Pt0_loc = SMatrix{2,2}(     P0.t[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Pf0_loc = SMatrix{2,2}(     P0.f[ii,jj] for ii in i:i+1,   jj in j:j+1)

        # Pfex    = SMatrix{4,4}(      P.f[ii,jj] for ii in i-1:i+2,   jj in j-1:j+2)
 
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
        τ0_loc  = @SVector([τ̄xx0[1], τ̄yy0[1], τ0.xy[i+1,j+1]])

        D̄kk   = av( Dkk)
        ϕ̄0    = av(Φ0_loc)
        P̄t0   = av(Pt0_loc)
        P̄f0   = av(Pf0_loc)

        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(D̄kk[1]), Const(divqD), Const(P̄t0[1]), Const(P̄f0[1]), Const(ϕ̄0[1]), Const(τ0_loc), Const(materials), Const(phases.v[i+1,j+1]), Const(Δ))

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