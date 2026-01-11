@inline mynorm(x) = sum(xi^2 for xi in x)

function PorosityRate(Pt, Pf, Pt0, Pf0, KΦ, ηΦ, λ̇, sinψ, Δt)    
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dΦdt    = (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + λ̇*sinψ
    return dΦdt
end

function ΔP_Trial(x, Pt_trial, Pf_trial, Φ, divVs, divqD, λ̇, Pt0, Pf0, Φ0, ηΦ, KΦ, Ks, Kf, sinψ, Δt )

    Pt, Pf = x[1], x[2]

    # Porosity rate
    dΦdt    = PorosityRate(Pt, Pf, Pt0, Pf0, KΦ, ηΦ, λ̇, sinψ, Δt) 
    dPtdt   = (Pt - Pt0) / Δt
    dPfdt   = (Pf - Pf0) / Δt
    dlnρfdt = dPfdt / Kf
    # dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

    Φ     = Φ0  + dΦdt * Δt
    dPsdt = ((Pt - Φ*Pf)/(1-Φ) - (Pt0 - Φ0*Pf0)/(1-Φ0))/Δt
    # dPsdt = dΦdt*(Pt - Pf*Φ)/(1-Φ)^2 + (dPtdt - Φ*dPfdt - Pf*dΦdt) / (1 - Φ)
    dlnρsdt = 1/Ks * ( dPsdt ) 


    # Ps     = (Pt - phi*Pf)/(1-phi) 
    # dPsdt = (dPtdt - phi*dPfdt) /(1-phi)
    # # dPsdt = ((Pt - phi*Pf)/(1-phi) - (Pt0 - phi0*Pf0)/(1-phi0))/dt
    # # dPsdt = dphidt*(Pt - Pf*phi)/(1-phi)**2 + (dPtdt - phi*dPfdt - 0*Pf*dphidt) / (1 - phi)
    # dlnrhosdt = elastic * 1/K_s * ( dPsdt ) 

    return @SVector [ 
        dlnρsdt   - dΦdt/(1-Φ),
        (Φ*dlnρfdt + dΦdt     )/ηΦ,
    ]
end

function ΔP(Pt_trial, Pf_trial, Φ_trial, divVs, divqD, λ̇, Pt0, Pf0, Φ0, ηΦ,  KΦ, Ks, Kf, sinψ, Δt)

    x   = @SVector[0.0, 0.0]
    r0  = 1.0
    tol = 1e-13

    for iter=1:10
        J  = Enzyme.jacobian(Enzyme.ForwardWithPrimal,  ΔP_Trial, x, Const(Pt_trial), Const(Pf_trial), Const(Φ_trial), Const(0*divVs), Const(0*divqD), Const(λ̇), Const(0*Pt0), Const(0*Pf0), Const(Φ0), Const(ηΦ),  Const(KΦ), Const(Ks), Const(Kf), Const(sinψ), Const(Δt))
        x  = x .- J.derivs[1]\J.val
        nr = mynorm(J.val)
        if iter==1 && nr>1e-17
            r0 = nr
        end
        r = nr/r0
        if r<tol
            break
        end
    end
    return x[1], x[2]
end


function residual_two_phase_P(x, ηve, Δt, ε̇II_eff, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, single_phase )
     
    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]
    α1 = single_phase ? 0.0 : 1.0 

    # Pressure corrections
    # ΔPt = KΦ .* sinψ .* Δt .* Φ_trial .* ηΦ .* λ̇ .* (-Kf + Ks) ./ (-Kf .* KΦ .* Δt .* Φ_trial + Kf .* KΦ .* Δt - Kf .* Φ_trial .* ηΦ + Kf .* ηΦ + Ks .* KΦ .* Δt .* Φ_trial + Ks .* Φ_trial .* ηΦ + KΦ .* Φ_trial .* ηΦ)
    # ΔPf = Kf .* KΦ .* sinψ .* Δt .* ηΦ .* λ̇ ./ (Kf .* KΦ .* Δt .* Φ_trial - Kf .* KΦ .* Δt + Kf .* Φ_trial .* ηΦ - Kf .* ηΦ - Ks .* KΦ .* Δt .* Φ_trial - Ks .* Φ_trial .* ηΦ - KΦ .* Φ_trial .* ηΦ)
    
    # Pressure corrections
    ΔPt, ΔPf = ΔP(Pt_trial, Pf_trial, Φ_trial, divVs, divqD, λ̇, Pt0, Pf0, Φ0, ηΦ,  KΦ, Ks, Kf, sinψ, Δt)

    # Porosity rate
    dΦdt = PorosityRate(Pt, Pf, Pt0, Pf0, KΦ, ηΦ, λ̇, sinψ, Δt)
    Φ    = Φ0  + dΦdt * Δt

    # Check yield
    f       = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    if single_phase
        f   = τII - C*cosϕ - Pt*sinϕ  
        ΔPt = Ks .* sinψ .* Δt .* λ̇
    end

    return @SVector [ 
        ε̇II_eff   -  τII/(2*ηve) - λ̇/2,
        Pt - (Pt_trial + ΔPt),
        Pf - (Pf_trial + ΔPf),
        f, 
    ]
end

function LocalRheology_P(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)

    # Effective strain rate & pressure
    ε̇II_eff  = invII(ε̇)
    Pt       = ε̇[4]
    Pf       = ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[phases]
    η0   = materials.ηs0[phases]
    # B    = materials.B[phases]
    G    = materials.G[phases]
    C    = materials.C[phases]
    ηΦ   = materials.ηΦ[phases]
    KΦ   = materials.KΦ[phases]
    Ks   = materials.Ks[phases]
    Kf   = materials.Kf[phases]

    ηvp  = materials.ηvp[phases]
    sinψ = materials.sinψ[phases]    
    sinϕ = materials.sinϕ[phases] 
    cosϕ = materials.cosϕ[phases]  
    
    α1 = materials.single_phase ? 0.0 : 1.0 

    # Initial guess
    η    = (η0 .* ε̇II_eff.^(1 ./ n .- 1.0 ))[1]
    ηve  = inv(1/η + 1/(G*Δ.t))
    τII  = 2*ηve*ε̇II_eff

    # Trial porosity
    Φ = (KΦ .* Δ.t .* (Pf - Pt) + KΦ .* Φ0 .* ηΦ + ηΦ .* (Pf - Pf0 - Pt + Pt0)) ./ (KΦ .* ηΦ)

    # Check yield
    λ̇  = 0.0

    # f       = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, 0.0)
    # if f>0
    #     λ̇ = f / (KΦ .* Δ.t * sinϕ * sinψ + ηve + ηvp)
    #     f  = τII - λ̇*ηve - C*cosϕ - (Pt + KΦ .* Δ.t * sinψ * λ̇)*sinϕ
    #     # @show f, λ̇
    #     # error()

    #     τII = τII - λ̇*ηve
    #     Pt  = Pt + KΦ .* Δ.t * sinψ * λ̇
    # end

    #############################

    f_trial  = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    x = @SVector ([τII, Pt, Pf, 0.0])

    # Return mapping
    if f_trial>-1e-13 

        # This is the proper return mapping with plasticity
        r0  = 1.0
        tol = 1e-10

        for iter=1:10
            J = Enzyme.jacobian(Enzyme.ForwardWithPrimal, residual_two_phase_P, x, Const(ηve), Const(Δ.t), Const(ε̇II_eff), Const(Pt), Const(Pf), Const(divVs), Const(divqD), Const(Φ), Const(Pt0), Const(Pf0), Const(Φ0), Const(ηΦ), Const(KΦ), Const(Ks), Const(Kf), Const(C), Const(cosϕ), Const(sinϕ), Const(sinψ), Const(ηvp), Const(materials.single_phase) )
            # display(J.derivs[1])
            x = x .- J.derivs[1]\J.val
            nr = mynorm(J.val)
            if iter==1 
                r0 = nr
            end
            r = nr/r0

            if r<tol
                break
            end
        end

    end

    τII, Pt, Pf, λ̇ = x[1], x[2], x[3], x[4]

    dΦdt = PorosityRate(Pt, Pf, Pt0, Pf0, KΦ, ηΦ, λ̇, sinψ, Δ.t) 
    Φ    = Φ0 + dΦdt * Δ.t

    #############################

    # Effective viscosity
    ηvep = τII/(2*ε̇II_eff)

    if materials.single_phase
        Φ = 0.0
    end

    f       = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)

    return ηvep, λ̇, Pt, Pf, τII, Φ, f
end


function StressVector_P!(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ) 
    η, λ̇, Pt, Pf, τII, Φ, f = LocalRheology_P(ε̇, divVs, divqD, Pt0, Pf0, Φ0, τ0, materials, phases, Δ)
    τ  = @SVector([2 * η * ε̇[1],
                   2 * η * ε̇[2],
                   2 * η * ε̇[3],
                             Pt,
                             Pf,])
    return τ, η, λ̇, τII, Φ, f
end


function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)

    _ones = @SVector ones(5)
    s = 1

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

        # Density for Darcy flux
        ρfS = materials.ρf[phases.c[i,j-1]]
        ρfC = materials.ρf[phases.c[i,j]]
        ρfg = 1/2*(ρfS + ρfC) * materials.g[2]

        # BCs
        Vx  = SetBCVx1(Vx, typex, bcx, Δ)
        Vy  = SetBCVy1(Vy, typey, bcy, Δ)
        Pf  = SetBCPf1(Pf_loc, typepf, bcpf, Δ, ρfg)

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
        divqD = (∂x(qDx) + ∂y(qDy))[1]
       
        # Visco-elasticity
        G      = materials.G[phases.c[i,j]]
        τ̄xy0   = av(τxy0)
        ε̇vec   = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), P.t[i,j], P.f[i,j]])
        τ0_loc = @SVector([τ0.xx[i,j], τ0.yy[i,j], τ̄xy0[1]])

        ##################################

        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector_P!, ε̇vec, Const(Dkk[1]), Const(divqD), Const(P0.t[i,j]), Const(P0.f[i,j]), Const(Φ0.c[i,j]), Const(τ0_loc), Const(materials), Const(phases.c[i,j]), Const(Δ))
        
        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        @views 𝐷_ctl.c[i,j][:,1] .= jac.derivs[1][1][1]
        @views 𝐷_ctl.c[i,j][:,2] .= jac.derivs[1][2][1]
        @views 𝐷_ctl.c[i,j][:,3] .= jac.derivs[1][3][1]
        @views 𝐷_ctl.c[i,j][:,4] .= jac.derivs[1][4][1]
        @views 𝐷_ctl.c[i,j][:,5] .= jac.derivs[1][5][1]

        ##################################

        # Tangent operator used for Picard Linearisation
        𝐷.c[i,j] .= diagm(2*jac.val[2] * _ones)
        𝐷.c[i,j][4,4] = 1
        𝐷.c[i,j][5,5] = 1

        ##################################

        # Update stress
        τ.xx[i,j] = jac.val[1][1]
        τ.yy[i,j] = jac.val[1][2]
        τ.II[i,j] = jac.val[4]
        τ.f[i,j]  = jac.val[6]
        ε̇.xx[i,j] = ε̇xx[1]
        ε̇.yy[i,j] = ε̇yy[1]
        ε̇.II[i,j] = invII( @SVector([ε̇xx[1], ε̇yy[1], ε̇̄xy[1]]) )
        λ̇.c[i,j]  = jac.val[3]
        Φ.c[i,j]  = jac.val[5]
        η.c[i,j]  = jac.val[2]
        if  λ̇.c[i,j] > 0
            ΔP.t[i,j] =  (jac.val[1][4] - P.t[i,j])
            ΔP.f[i,j] =  (jac.val[1][5] - P.f[i,j])
        end
    end

    # Loop over vertices
    for j=3:size(ε̇.xy,2)-2, i=3:size(ε̇.xy,1)-2
        Vx      = SMatrix{3,2}(      V.x[ii,jj] for ii in i-1:i+1,   jj in j-1+1:j+1)
        Vy      = SMatrix{2,3}(      V.y[ii,jj] for ii in i-1+1:i+1, jj in j-1:j+1  )
        bcx     = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i-1:i+1,   jj in j-1+1:j+1)
        bcy     = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i-1+1:i+1, jj in j-1:j+1  )
        typex   = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i-1:i+1,   jj in j-1+1:j+1)
        typey   = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i-1+1:i+1, jj in j-1:j+1  )
        τxx0    = SMatrix{2,2}(    τ0.xx[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        τyy0    = SMatrix{2,2}(    τ0.yy[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Pt      = SMatrix{2,2}(      P.t[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Pf      = SMatrix{2,2}(      P.f[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Φ0_loc  = SMatrix{2,2}(     Φ0.c[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Pt0_loc = SMatrix{2,2}(     P0.t[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)
        Pf0_loc = SMatrix{2,2}(     P0.f[ii,jj] for ii in i-1:i+0,   jj in j-1:j+0)

        Pfex    = SMatrix{4,4}(      P.f[ii,jj] for ii in i-2:i+1,   jj in j-2:j+1)
        typepf  = SMatrix{4,4}(  type.Pf[ii,jj] for ii in i-2:i+1,   jj in j-2:j+1)
        bcpf    = SMatrix{4,4}(    BC.Pf[ii,jj] for ii in i-2:i+1,   jj in j-2:j+1)

        Vx     = SetBCVx1(Vx, typex, bcx, Δ)
        Vy     = SetBCVy1(Vy, typey, bcy, Δ)

        ρgf    = materials.ρf[phases.v[i,j]] * materials.g[2]
        
        Pf     = SetBCPf1(Pfex, typepf, bcpf, Δ, ρgf)

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

        qDx   = materials.k_ηf0[1] .*  ∂x_inn(Pf) / Δ.x 
        qDy   = materials.k_ηf0[1] .*  ∂y_inn(Pf) / Δ.y
        divqD = (∂x(qDx) + ∂y(qDy))
        
        divqD̄ = av(divqD)[1]
        
        # Visco-elasticity
        G     = materials.G[phases.v[i,j]]
        τ̄xx0  = av(τxx0)
        τ̄yy0  = av(τyy0)
        P̄t    = av(  Pt)
        P̄f    = av(  Pf)
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t), P̄t[1], P̄f[1]])
        τ0_loc  = @SVector([τ̄xx0[1], τ̄yy0[1], τ0.xy[i,j]])

        D̄kk   = av(Dkk)
        ϕ̄0    = av(Φ0_loc)
        P̄t0   = av(Pt0_loc)
        P̄f0   = av(Pf0_loc)

        ##################################

        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector_P!, ε̇vec, Const(D̄kk[1]), Const(divqD̄), Const(P̄t0[1]), Const(P̄f0[1]), Const(ϕ̄0[1]), Const(τ0_loc), Const(materials), Const(phases.v[i,j]), Const(Δ))

        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        @views 𝐷_ctl.v[i,j][:,1] .= jac.derivs[1][1][1]
        @views 𝐷_ctl.v[i,j][:,2] .= jac.derivs[1][2][1]
        @views 𝐷_ctl.v[i,j][:,3] .= jac.derivs[1][3][1]
        @views 𝐷_ctl.v[i,j][:,4] .= jac.derivs[1][4][1]
        @views 𝐷_ctl.v[i,j][:,5] .= jac.derivs[1][5][1]

        ##################################

        # Tangent operator used for Picard Linearisation
        𝐷.v[i,j] .= diagm(2*jac.val[2] * _ones)
        𝐷.v[i,j][4,4] = 1
        𝐷.v[i,j][5,5] = 1

        # Update stress
        τ.xy[i,j] = jac.val[1][3]
        ε̇.xy[i,j] = ε̇xy[1]
        λ̇.v[i,j]  = jac.val[3]
        η.v[i,j]  = jac.val[2]
    end

    # Cheap copy edges
    for j=2:size(ε̇.xy,2)-1 
        i = 2
        @views 𝐷_ctl.v[i,j] .= 𝐷_ctl.v[3,j]
        @views 𝐷.v[i,j]     .= 𝐷.v[3,j]
        i = size(ε̇.xy,1)-1
        @views 𝐷_ctl.v[i,j] .= 𝐷_ctl.v[end-2,j]
        @views 𝐷.v[i,j]     .= 𝐷.v[end-2,j]
    end

    for i=2:size(ε̇.xy,1)-1 
        j = 2
        @views 𝐷_ctl.v[i,j] .= 𝐷_ctl.v[i,3]
        @views 𝐷.v[i,j]     .= 𝐷.v[i,3]
        j = size(ε̇.xy,2)-1
        @views 𝐷_ctl.v[i,j] .= 𝐷_ctl.v[i,end-2]
        @views 𝐷.v[i,j]     .= 𝐷.v[i,end-2]
    end


end

# function LocalRheology(ε̇, materials, phases, Δ)

#     # Effective strain rate & pressure
#     ε̇II  = sqrt.( (ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1]-ε̇[2])^2)/2 + ε̇[3]^2 )
#     P    = ε̇[4]

#     # Parameters
#     ϵ    = 1e-10 # tolerance
#     n    = materials.n[phases]
#     η0   = materials.ηs0[phases]
#     B    = materials.B[phases]
#     G    = materials.G[phases]
#     C    = materials.C[phases]

#     ϕ    = materials.ϕ[phases]
#     ψ    = materials.ψ[phases]

#     ηvp  = materials.ηvp[phases]
#     sinψ = materials.sinψ[phases]    
#     sinϕ = materials.sinϕ[phases] 
#     cosϕ = materials.cosϕ[phases]    

#     β    = materials.β[phases]
#     comp = materials.compressible

#     # Initial guess
#     η    = (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
#     ηvep = inv(1/η + 1/(G*Δ.t))
#     τII  = 2*ηvep*ε̇II

#     # # Visco-elastic powerlaw
#     # for it=1:20
#     #     r      = ε̇II - StrainRateTrial(τII, G, Δ.t, B, n)
#     #     # @show abs(r)
#     #     (abs(r)<ϵ) && break
#     #     ∂ε̇II∂τII = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial, τII, G, Δ.t, B, n)
#     #     ∂τII∂ε̇II = inv(∂ε̇II∂τII[1])
#     #     τII     += ∂τII∂ε̇II*r
#     # end
#     # isnan(τII) && error()
 
#     # # Viscoplastic return mapping
#     λ̇ = 0.
#     # if materials.plasticity === :DruckerPrager
#     #     τII, P, λ̇ = DruckerPrager(τII, P, ηvep, comp, β, Δ.t, C, cosϕ, sinϕ, sinψ, ηvp)
#     # elseif materials.plasticity === :tensile
#     #     τII, P, λ̇ = Tensile(τII, P, ηvep, comp, β, Δ.t, materials.σT[phases], ηvp)
#     # elseif materials.plasticity === :Kiss2023
#     #     τII, P, λ̇ = Kiss2023(τII, P, ηvep, comp, β, Δ.t, C, ϕ, ψ, ηvp, materials.σT[phases], materials.δσT[phases], materials.P1[phases], materials.τ1[phases], materials.P2[phases], materials.τ2[phases])
#     # end

#     # Effective viscosity
#     ηvep = τII/(2*ε̇II)

#     return ηvep, λ̇, P
# end

# function StressVector!(ε̇, materials, phases, Δ) 
#     η, λ̇, P = LocalRheology(ε̇, materials, phases, Δ)
#     τ       = @SVector([2 * η * ε̇[1],
#                         2 * η * ε̇[2],
#                         2 * η * ε̇[3],
#                                   P])
#     return τ, η, λ̇
# end

# function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔPt, P0, Φ, Φ0, type, BC, materials, phases, Δ)

#     _ones = @SVector ones(5)

#     # Loop over centroids
#     for j=2:size(ε̇.xx,2)-1, i=2:size(ε̇.xx,1)-1

#             Vx     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1,   jj in j:j+2)
#             Vy     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2,   jj in j:j+1)
#             bcx    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
#             bcy    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
#             typex  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
#             typey  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
#             τxy0   = SMatrix{2,2}(    τ0.xy[ii,jj] for ii in i:i+1,   jj in j:j+1)

#             Vx = SetBCVx1(Vx, typex, bcx, Δ)
#             Vy = SetBCVy1(Vy, typey, bcy, Δ)

#             Dxx = ∂x_inn(Vx) / Δ.x 
#             Dyy = ∂y_inn(Vy) / Δ.y 
#             Dxy = ∂y(Vx) / Δ.y
#             Dyx = ∂x(Vy) / Δ.x
            
#             Dkk = Dxx .+ Dyy
#             ε̇xx = @. Dxx - Dkk ./ 3
#             ε̇yy = @. Dyy - Dkk ./ 3
#             ε̇xy = @. (Dxy + Dyx) ./ 2
#             ε̇̄xy = av(ε̇xy)
        
#             # Visco-elasticity
#             G     = materials.G[phases.c[i,j]]
#             τ̄xy0  = av(τxy0)
#             ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), P.t[i,j]])

#             # Tangent operator used for Newton Linearisation
#             jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.c[i,j]), Const(Δ))
            
#             # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
#             @views 𝐷_ctl.c[i,j][1:4,1] .= jac.derivs[1][1][1]
#             @views 𝐷_ctl.c[i,j][1:4,2] .= jac.derivs[1][2][1]
#             @views 𝐷_ctl.c[i,j][1:4,3] .= jac.derivs[1][3][1]
#             @views 𝐷_ctl.c[i,j][1:4,4] .= jac.derivs[1][4][1]

#             # Tangent operator used for Picard Linearisation
#             𝐷.c[i,j] .= diagm(2*jac.val[2] * _ones)
#             𝐷.c[i,j][4,4] = 1

#             # Update stress
#             τ.xx[i,j]  = jac.val[1][1]
#             τ.yy[i,j]  = jac.val[1][2]
#             ε̇.xx[i,j]  = ε̇xx[1]
#             ε̇.yy[i,j]  = ε̇yy[1]
#             λ̇.c[i,j]   = jac.val[3]
#             η.c[i,j]   = jac.val[2]
#             # ΔP.t[i,j] = (jac.val[1][4] - P.t[i,j])
#     end

#     # Loop over vertices
#     for j=2:size(ε̇.xy,2)-1, i=2:size(ε̇.xy,1)-1
#         Vx     = SMatrix{3,2}(      V.x[ii,jj] for ii in i-1:i+1, jj in j:j+1)
#         Vy     = SMatrix{2,3}(      V.y[ii,jj] for ii in i+0:i+1, jj in j-1:j+1)
#         bcx    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j:j+1)
#         bcy    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i+0:i+1, jj in j-1:j+1)
#         typex  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j:j+1)
#         typey  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i+0:i+1, jj in j-1:j+1)
#         τxx0   = SMatrix{2,2}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-1:j)
#         τyy0   = SMatrix{2,2}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-1:j)
#         Pt     = SMatrix{2,2}(      P.t[ii,jj] for ii in i-1:i,   jj in j-1:j)

#         Vx     = SetBCVx1(Vx, typex, bcx, Δ)
#         Vy     = SetBCVy1(Vy, typey, bcy, Δ)
    
#         Dxx    = ∂x(Vx) / Δ.x
#         Dyy    = ∂y(Vy) / Δ.y
#         Dxy    = ∂y_inn(Vx) / Δ.y
#         Dyx    = ∂x_inn(Vy) / Δ.x

#         Dkk   = @. Dxx + Dyy
#         ε̇xx   = @. Dxx - Dkk / 3
#         ε̇yy   = @. Dyy - Dkk / 3
#         ε̇xy   = @. (Dxy + Dyx) /2
#         ε̇̄xx   = av(ε̇xx)
#         ε̇̄yy   = av(ε̇yy)
        
#         # Visco-elasticity
#         G     = materials.G[phases.v[i,j]]
#         τ̄xx0  = av(τxx0)
#         τ̄yy0  = av(τyy0)
#         P̄     = av(  Pt)
#         ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t), P̄[1]])
        
#         # Tangent operator used for Newton Linearisation
#         jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.v[i,j]), Const(Δ))

#         # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
#         @views 𝐷_ctl.v[i,j][1:4,1] .= jac.derivs[1][1][1]
#         @views 𝐷_ctl.v[i,j][1:4,2] .= jac.derivs[1][2][1]
#         @views 𝐷_ctl.v[i,j][1:4,3] .= jac.derivs[1][3][1]
#         @views 𝐷_ctl.v[i,j][1:4,4] .= jac.derivs[1][4][1]

#         # Tangent operator used for Picard Linearisation
#         𝐷.v[i,j] .= diagm(2*jac.val[2] * _ones)
#         𝐷.v[i,j][4,4] = 1

#         # Update stress
#         τ.xy[i,j] = jac.val[1][3]
#         ε̇.xy[i,j] = ε̇xy[1]
#         λ̇.v[i,j]  = jac.val[3]
#         η.v[i,j]  = jac.val[2]
#         # τ.xy[i,j] = 2*jac.val[2]*(ε̇xy[1]+τ0.xy[i,j]/(2*G[1]*Δ.t))
#     end
# end


# function residual_two_phase_P(x, ηve, Δt, ε̇II_eff, Pt_trial, Pf_trial, divVs, divqD, Φ_trial, Pt0, Pf0, Φ0, ηΦ, KΦ, Ks, Kf, C, cosϕ, sinϕ, sinψ, ηvp, single_phase )
     
#     # eps   = -1e-20
#     τII, Pt, Pf, λ̇, Φ, ΔPt, ΔPf = x[1], x[2], x[3], x[4], x[5], x[6], x[7]
#     single_phase ? α1 = 0.0 : α1 = 1.0 

#     # Check yield
#     f       = F(τII, Pt, Pf, 0.0, C, cosϕ, sinϕ, λ̇, ηvp, α1)

#     # Porosity rate
#     dPtdt   = (Pt - Pt0) / Δt
#     dPfdt   = (Pf - Pf0) / Δt
#     dΦdt    = (dPfdt - dPtdt)/KΦ + (Pf - Pt)/ηΦ + λ̇*sinψ#*(f>=eps)
#     f_poro  = Φ    - (Φ0 + dΦdt*Δt)

#     if single_phase
#         f   = τII - C*cosϕ - Pt*sinϕ  
#         ΔPt = Ks .* sinψ .* Δt .* λ̇
#     end

#     # Create a residual for the pressure increments... 
#     # Porosity rate
#     dPtdt   = (ΔPt) / Δt
#     dPfdt   = (ΔPf) / Δt
#     dΦdt    = (dPfdt - dPtdt)/KΦ + (ΔPf - ΔPt)/ηΦ + λ̇*sinψ
#     dlnρfdt = dPfdt / Kf
#     dlnρsdt = 1/(1-Φ) *(dPtdt - Φ*dPfdt) / Ks

#     return @SVector [ 
#         ε̇II_eff   -  τII/2/ηve - λ̇/2,#*(f>=eps),
#         Pt - (Pt_trial + ΔPt),
#         Pf - (Pf_trial + ΔPf),
#         f, #(f - ηvp*λ̇)*(f>=eps) +  λ̇*1*(f<eps),
#         f_poro,
#         dlnρsdt   - dΦdt/(1-Φ),
#         (Φ*dlnρfdt + dΦdt     )/ηΦ,
#     ]
# end