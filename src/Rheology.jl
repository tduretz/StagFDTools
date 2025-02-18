function line(p, K, dt, η_ve, ψ, p1, t1)
    p2 = p1 + K*dt*sind(ψ)  # introduce sinϕ ?
    t2 = t1 - η_ve  
    a  = (t2-t1)/(p2-p1)
    b  = t2 - a*p2
    return a*p + b
end


function Kiss2023(τ, P, η_ve, comp, β, Δt, C, φ, ψ, ηvp, σ_T, δσ_T, pc1, τc1, pc2, τc2)

    K         = 1/β
    λ̇         = 0.
    domain_pl = 0.0
    Pc        = P
    τc        = τ

    l1    = line(P, K, Δt, η_ve, π/2, pc1, τc1)
    l2    = line(P, K, Δt, η_ve, π/2, pc2, τc2)
    l3    = line(P, K, Δt, η_ve,   ψ, pc2, τc2)

    if max(τ - P*sind(φ) - C*cosd(φ) , τ - P - σ_T , - P - (σ_T - δσ_T) ) > 0.0                                                         # check if F_tr > 0
        if τ <= τc1 
            # pressure limiter 
            dqdp = -1.0
            f    = - P - (σ_T - δσ_T) 
            λ̇    = f / (K*Δt)                                                                                                                          # tensile pressure cutoff
            τc   = τ 
            Pc   = P - K*Δt*λ̇*dqdp
            f    = - Pc - (σ_T - δσ_T) 
            domain_pl = 1.0
        elseif τc1 < τ <= l1    
            # corner 1 
            τc = τ - η_ve*(τ - τc1)/(η_ve)
            Pc = P - K*Δt*(P - pc1)/(K*Δt)
            domain_pl = 2.0
        # elseif l1 < τ <= l2            # mode-1
        # # if τ - P - σ_T > 1e-10
        #     # tension
        #     dqdp = -1.0
        #     dqdτ =  1.0
        #     f    = τ - P - σ_T 
        #     λ̇    = f / (K*Δt + η_ve + ηvp) 
        #     τc   = τ - η_ve*λ̇*dqdτ
        #     Pc   = P - K*Δt*λ̇*dqdp
        #     domain_pl = 3.0 
        #     # @show τc - Pc - σ_T - ηvp*λ̇
        # elseif l2< τ <= l3 # 2nd corner
        #     # corner 2
        #     τc = τ - η_ve*(τ - τc2)/(η_ve)
        #     Pc = P - K*Δt*(P - pc2)/(K*Δt)
        #     domain_pl = 4.0
        # elseif l3 < τ                  # mode-2
        # # if τ - P*sind(φ) - C*cosd(φ) > 1e-10
        #     # Drucker-Prager                                                             
        #     dqdp = -sind(ψ)
        #     dqdτ =  1.0
        #     f    = τ - P*sind(φ) - C*cosd(φ) 
        #     λ̇    = f / (K*Δt*sind(φ)*sind(ψ) + η_ve + ηvp) 
        #     τc   = τ - η_ve*λ̇*dqdτ
        #     Pc   = P - K*Δt*λ̇*dqdp
        end
    end

    return τc, Pc, λ̇
end


function DruckerPrager(τII, P, ηve, comp, β, Δt, C, ϕ, ψ, ηvp)
    λ̇    = 0.0
    F    = τII - C*cosd(ϕ) - P*sind(ϕ )- λ̇*ηvp
    if F > 1e-10
        λ̇    = F / (ηve + ηvp + comp*Δt/β*sind(ϕ)*sind(ψ)) 
        τII -= λ̇ * ηve
        P   += comp * λ̇*sind(ψ)*Δt/β
        F    = τII - C*cosd(ϕ) - P*sind(ϕ )- λ̇*ηvp
        (F>1e-10) && error("Failed return mapping")
        (τII<0.0) && error("Plasticity without condom")
    end
    return τII, P, λ̇
end

function StrainRateTrial(τII, G, Δt, B, n)
    ε̇II_vis   = B.*τII.^n 
    ε̇II_trial = ε̇II_vis + τII/(2*G*Δt)
    return ε̇II_trial
end

function LocalRheology(ε̇, materials, phases, Δ)

    # Effective strain rate & pressure
    ε̇II  = sqrt.(1/2*(ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1]-ε̇[2])^2) + ε̇[3]^2)
    P    = ε̇[4]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[phases]
    η0   = materials.η0[phases]
    B    = materials.B[phases]
    G    = materials.G[phases]
    C    = materials.C[phases]
    ϕ    = materials.ϕ[phases]
    ηvp  = materials.ηvp[phases]
    ψ    = materials.ψ[phases]    
    β    = materials.β[phases]
    comp = materials.compressible

    # Initial guess
    η    =  (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
    ηvep = inv(1/η + 1/(G*Δ.t))
    τII  = 2*ηvep*ε̇II

    # Visco-elastic powerlaw
    for it=1:20
        r      = ε̇II - StrainRateTrial(τII, G, Δ.t, B, n)
        # @show abs(r)
        (abs(r)<ϵ) && break
        ∂ε̇II∂τII = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial_Invariant, τII, G, Δ.t, B, n)
        ∂τII∂ε̇II = inv(∂ε̇II∂τII[1])
        τII     += ∂τII∂ε̇II*r
    end
    isnan(τII) && error()
 
    # # Viscoplastic return mapping
    λ̇ = 0.
    if materials.plasticity === :DruckerPrager
        τII, P, λ̇ = DruckerPrager(τII, P, ηvep, comp, β, Δ.t, C, ϕ, ψ, ηvp)
    elseif materials.plasticity === :Kiss2023
        τII, P, λ̇ = Kiss2023(τII, P, ηvep, comp, β, Δ.t, C, ϕ, ψ, ηvp, materials.σT[phases], materials.δσT[phases], materials.P1[phases], materials.τ1[phases], materials.P2[phases], materials.τ2[phases])
    end

    # Effective viscosity
    ηvep = τII/(2*ε̇II)

    return ηvep, λ̇, P
end

function StressVector!(ε̇, materials, phases, Δ) 
    η, λ̇, P = LocalRheology(ε̇, materials, phases, Δ)
    τ       = @SVector([2 * η * ε̇[1],
                        2 * η * ε̇[2],
                        2 * η * ε̇[3],
                                  P])
    return τ, η, λ̇
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, Pt, ΔPt, type, BC, materials, phases, Δ)

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
        
            # Visco-elasticity
            G     = materials.G[phases.c[i,j]]
            τ̄xy0  = av(τxy0)
            ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), Pt[i,j]])

            # Tangent operator used for Newton Linearisation
            jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.c[i,j]), Const(Δ))
            
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
        
        # Visco-elasticity
        G     = materials.G[phases.v[i,j]]
        τ̄xx0  = av(τxx0)
        τ̄yy0  = av(τyy0)
        P̄     = av(   P)
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t), P̄[1]])
        
        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.v[i+1,j+1]), Const(Δ))

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
    end
end

function LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x 
    Vi.y .= V.y 
    Pti  .= Pt
    for i in eachindex(α)
        V.x .= Vi.x 
        V.y .= Vi.y
        Pt  .= Pti
        UpdateSolution!(V, Pt, α[i].*dx, number, type, nc)
        TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, type, BC, materials, phases, Δ)
        ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
        ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        rvec[i] = @views norm(R.x[inx_Vx,iny_Vx])/length(R.x[inx_Vx,iny_Vx]) + norm(R.y[inx_Vy,iny_Vy])/length(R.y[inx_Vy,iny_Vy]) + norm(R.p[inx_c,iny_c])/length(R.p[inx_c,iny_c])  
    end
    imin = argmin(rvec)
    V.x .= Vi.x 
    V.y .= Vi.y
    Pt  .= Pti
    return imin
end
