using StagFDTools, StagFDTools.TwoPhases, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf, JLD2
import Statistics:mean
using DifferentiationInterface
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian
function LocalRheology(ε̇, materials, phases, Δ)

    # Effective strain rate & pressure
    ε̇II  = sqrt.( (ε̇[1]^2 + ε̇[2]^2 + (-ε̇[1]-ε̇[2])^2)/2 + ε̇[3]^2 ) + 1e-14
    Pt   = ε̇[4]
    Pf   = ε̇[5]

    # Parameters
    ϵ    = 1e-10 # tolerance
    n    = materials.n[phases]
    η0   = materials.ηs0[phases]
    # B    = materials.B[phases]
    G    = materials.G[phases]
    # C    = materials.C[phases]

    # ϕ    = materials.ϕ[phases]
    # ψ    = materials.ψ[phases]

    # ηvp  = materials.ηvp[phases]
    # sinψ = materials.sinψ[phases]    
    # sinϕ = materials.sinϕ[phases] 
    # cosϕ = materials.cosϕ[phases]    

    # β    = materials.β[phases]
    # comp = materials.compressible

    # Initial guess
    η    = (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
    ηvep = inv(1/η + 1/(G*Δ.t))
    # ηvep = G*Δ.t

    τII  = 2*ηvep*ε̇II

    # # Visco-elastic powerlaw
    # for it=1:20
    #     r      = ε̇II - StrainRateTrial(τII, G, Δ.t, B, n)
    #     # @show abs(r)
    #     (abs(r)<ϵ) && break
    #     ∂ε̇II∂τII = forwarddiff_jacobian(StrainRateTrial, τII, G, Δ.t, B, n)
    #     ∂τII∂ε̇II = inv(∂ε̇II∂τII[1])
    #     τII     += ∂τII∂ε̇II*r
    # end
    # isnan(τII) && error()
 
    # # Viscoplastic return mapping
    λ̇ = 0.
    # if materials.plasticity === :DruckerPrager
    #     τII, P, λ̇ = DruckerPrager(τII, P, ηvep, comp, β, Δ.t, C, cosϕ, sinϕ, sinψ, ηvp)
    # elseif materials.plasticity === :tensile
    #     τII, P, λ̇ = Tensile(τII, P, ηvep, comp, β, Δ.t, materials.σT[phases], ηvp)
    # elseif materials.plasticity === :Kiss2023
    #     τII, P, λ̇ = Kiss2023(τII, P, ηvep, comp, β, Δ.t, C, ϕ, ψ, ηvp, materials.σT[phases], materials.δσT[phases], materials.P1[phases], materials.τ1[phases], materials.P2[phases], materials.τ2[phases])
    # end

    # Effective viscosity
    ηvep = τII/(2*ε̇II)

    return ηvep, λ̇, Pt, Pf
end

function StressVector!(ε̇, materials, phases, Δ) 
    η, λ̇, Pt, Pf = LocalRheology(ε̇, materials, phases, Δ)
    τ            = @SVector([2 * η * ε̇[1],
                             2 * η * ε̇[2],
                             2 * η * ε̇[3],
                                       Pt,
                                       Pf,])
    return τ, η, λ̇
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, P, ΔP, type, BC, materials, phases, Δ)

    _ones = @SVector ones(5)

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
            ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), P.t[i,j], P.f[i,j]])

            # Tangent operator used for Newton Linearisation
            jac   = forwarddiff_jacobian(StressVector!, ε̇vec, Const(materials), Const(phases.c[i,j]), Const(Δ))
            
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
        Pt     = SMatrix{2,2}(      P.t[ii,jj] for ii in i:i+1,   jj in j:j+1)
        Pf     = SMatrix{2,2}(      P.f[ii,jj] for ii in i:i+1,   jj in j:j+1)

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
        G     = materials.G[phases.v[i+1,j+1]]
        τ̄xx0  = av(τxx0)
        τ̄yy0  = av(τyy0)
        P̄t    = av(   Pt)
        P̄f    = av(   Pf)
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t), P̄t[1], P̄f[1]])
        
        # Tangent operator used for Newton Linearisation
        jac   = forwarddiff_jacobian(StressVector!, ε̇vec, Const(materials), Const(phases.v[i+1,j+1]), Const(Δ))

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

function Rozhko2008(rho, phi, r1, rc, P0, dPf, m, G, ν)
    eta   = (1.0 - 2.0*ν)/(1.0-ν)/2.0
    kappa = 3. - 4. * ν
    if rho < r1
        Pf   = dPf
        Ux   = 0.
        Uy   = 0.
        Ur   = 0.
        Ut   = 0.
        Pt   = 0.
        Sxx  = 0.  
        Syy  = 0.  
        Sxy  = 0.  
    else
        Srr = (eta*rho^2*dPf*m^3*cos(2*phi)*log(1/(rho^32))+eta*rho^2*dPf*m^4*log(1/(rc^8))+eta*dPf*m^4*log(rho^8*rc^8)+eta*rho^6*dPf*log(rc^8)+eta*m^2*rho^4*P0*log(1/(rho^32))+eta*m^2*rho^4*dPf*log(rho^32)+eta*m*rho^6*P0*cos(2*phi)*log(rho^32)+eta*m^2*rho^2*dPf*log(1/(rc^8))+eta*rho^2*P0*m^4*log(rc^8)+eta*rho^8*dPf*log(1/rc^8*rho^8)+eta*rho^8*P0*log(1/rho^8*rc^8)+8*eta*P0*m^4+24*eta*m^2*rho^2*dPf+16*eta*m^2*rho^4*P0-16*eta*m^2*rho^4*dPf-8*eta*rho^6*dPf*m^2+8*eta*rho^6*P0*m^2-8*eta*dPf*m^4-24*eta*m^2*rho^2*P0+eta*rho^6*dPf*m^2*log(rc^8)+8*eta*rho^2*dPf*m^4-8*eta*rho^2*P0*m^4+8*eta*rho^2*dPf*m^3*cos(2*phi)-8*eta*m^3*dPf*cos(2*phi)+8*eta*m^3*P0*cos(2*phi)+eta*P0*m^4*log(1/(rho^8*rc^8))+eta*rho^6*P0*log(1/(rc^8))+24*eta*m*rho^4*P0*cos(2*phi)-8*eta*m^2*rho^2*P0*cos(4*phi)+8*eta*m^2*rho^4*P0*cos(4*phi)-24*eta*m*rho^6*P0*cos(2*phi)+8*eta*m^2*rho^2*dPf*cos(4*phi)-8*eta*m^2*rho^4*dPf*cos(4*phi)-8*eta*rho^2*P0*m^3*cos(2*phi)+eta*m^2*rho^4*dPf*log(rho^16)*cos(4*phi)+eta*m^2*rho^4*P0*log(1/(rho^16))*cos(4*phi)-24*eta*m*rho^4*dPf*cos(2*phi)+eta*rho^2*P0*m^3*cos(2*phi)*log(rho^32)+24*eta*m*rho^6*dPf*cos(2*phi)+eta*m^2*rho^2*P0*log(rc^8)+eta*rho^6*P0*m^2*log(1/(rc^8))+eta*m*rho^6*dPf*cos(2*phi)*log(1/(rho^32)))/(m^2*rho^4*cos(4*phi)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*phi)*log(1/(rc^32))+rho^2*m^3*cos(2*phi)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
        Stt = (eta*dPf*m^4*log(rho^8*rc^8)+eta*m^2*rho^4*P0*log(1/(rho^32))+eta*m^2*rho^4*dPf*log(rho^32)+eta*m^2*rho^2*dPf*log(rc^8)+eta*rho^6*P0*m^2*log(rc^8)+eta*rho^2*dPf*m^4*log(rc^8)+eta*m^2*rho^2*P0*log(1/(rc^8))+eta*rho^8*dPf*log(1/rc^8*rho^8)+eta*rho^8*P0*log(1/rho^8*rc^8)+16*eta*P0*m^4-24*eta*m^2*rho^2*dPf+16*eta*m^2*rho^4*P0-16*eta*m^2*rho^4*dPf+8*eta*rho^6*dPf*m^2-8*eta*rho^6*P0*m^2-16*eta*dPf*m^4+24*eta*m^2*rho^2*P0-8*eta*rho^2*dPf*m^4+8*eta*rho^2*P0*m^4+56*eta*rho^2*dPf*m^3*cos(2*phi)+8*eta*m^3*dPf*cos(2*phi)-8*eta*m^3*P0*cos(2*phi)+eta*P0*m^4*log(1/(rho^8*rc^8))-24*eta*m*rho^4*P0*cos(2*phi)+8*eta*m^2*rho^2*P0*cos(4*phi)+8*eta*m^2*rho^4*P0*cos(4*phi)+24*eta*m*rho^6*P0*cos(2*phi)-8*eta*m^2*rho^2*dPf*cos(4*phi)-8*eta*m^2*rho^4*dPf*cos(4*phi)-56*eta*rho^2*P0*m^3*cos(2*phi)+eta*m^2*rho^4*dPf*log(rho^16)*cos(4*phi)+eta*m^2*rho^4*P0*log(1/(rho^16))*cos(4*phi)+24*eta*m*rho^4*dPf*cos(2*phi)-24*eta*m*rho^6*dPf*cos(2*phi)+eta*rho^6*dPf*log(1/(rc^8))+eta*rho^6*dPf*m^2*log(1/(rc^8))+eta*rho^2*P0*m^3*cos(2*phi)*log(rho^32*rc^32)+eta*rho^6*P0*log(rc^8)+eta*rho^2*P0*m^4*log(1/(rc^8))+eta*m*rho^6*P0*cos(2*phi)*log(1/rc^32*rho^32)+8*eta*rho^8*dPf-8*eta*rho^8*P0+eta*m*rho^6*dPf*cos(2*phi)*log(1/rho^32*rc^32)+eta*rho^2*dPf*m^3*cos(2*phi)*log(1/(rho^32*rc^32)))/(m^2*rho^4*cos(4*phi)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*phi)*log(1/(rc^32))+rho^2*m^3*cos(2*phi)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
        Srt = eta*m*sin(2*phi)*(-2*rho^6*dPf*log(rc)+2*rho^2*log(rc)*P0*m^2-2*rho^4*log(rc)*P0*m^2+2*rho^4*dPf*log(rc)*m^2+2*m*dPf*rho^2*cos(2*phi)-2*m*P0*rho^2*cos(2*phi)-2*rho^4*dPf*m^2+2*rho^6*log(rc)*P0+2*rho^4*P0*m^2-2*m*rho^4*dPf*cos(2*phi)-m^2*dPf+m^2*P0-3*rho^2*P0*m^2+3*rho^4*P0-3*rho^4*dPf+3*rho^2*dPf*m^2-3*rho^6*P0+3*rho^6*dPf+2*m*rho^4*P0*cos(2*phi)+2*rho^4*dPf*log(rc)-2*rho^2*dPf*log(rc)*m^2-2*rho^4*log(rc)*P0)/log(rc)/(4*m^2*rho^4*cos(2*phi)^2+2*m^2*rho^4-4*rho^6*m*cos(2*phi)-4*rho^2*m^3*cos(2*phi)+rho^8+m^4);
                
        Ux  = -1/8*eta*r1*cos(phi)*(11*m*rho^4*dPf-11*m*rho^4*P0+kappa*rho^6*dPf+4*m^3*log(rho)*P0-4*m^3*log(rho)*dPf+5*rho^2*P0*m^2-kappa*rho^6*P0+4*rho^2*P0*m^3-3*kappa*m^3*dPf+3*kappa*m^3*P0-4*rho^2*dPf*m^3+12*m*P0*rho^2-12*rho^2*m*dPf-4*rho^2*m*log(rc)*P0+2*kappa*log(rc)*rho^6*P0+4*rho^2*dPf*log(rc)*m^3+4*kappa*log(rc)*m^3*dPf-20*rho^4*m*dPf*cos(phi)^2+6*rho^4*m*log(rc)*P0+20*m*P0*rho^4*cos(phi)^2+12*dPf*m^2*cos(phi)^2*rho^2-16*m*P0*cos(phi)^2*rho^2-12*P0*m^2*cos(phi)^2*rho^2+16*m*dPf*cos(phi)^2*rho^2+dPf*m^3+4*P0*m^2-rho^6*P0+rho^6*dPf-5*rho^2*dPf*m^2-8*kappa*log(rc)*m*P0*rho^4*cos(phi)^2+8*kappa*log(rc)*m^2*dPf*rho^2+2*kappa*log(rc)*m*P0*rho^4-2*kappa*log(rc)*m^2*P0*rho^2-4*kappa*rho^4*dPf*m*cos(phi)^2+4*kappa*log(rc)*m*dPf*rho^4+4*kappa*m*P0*rho^4*cos(phi)^2-8*rho^4*m*log(rc)*P0*cos(phi)^2+16*cos(phi)^2*dPf*rho^2*log(rho)*m^2+4*rho^4*dPf*m^2-4*rho^4*P0*m^2-5*kappa*rho^2*dPf*m^2+5*kappa*m^2*P0*rho^2+16*cos(phi)^2*m*log(rho)*dPf*rho^4-16*cos(phi)^2*rho^2*P0*log(rho)*m^2-16*cos(phi)^2*m*log(rho)*P0*rho^4+8*kappa*log(rc)*m^2*P0*cos(phi)^2*rho^2-16*kappa*log(rc)*m^2*dPf*cos(phi)^2*rho^2-12*kappa*m^2*P0*cos(phi)^2*rho^2-16*log(rc)*m^2*dPf*cos(phi)^2*rho^2+4*rho^6*dPf*log(rc)+4*rho^4*log(rc)*P0-4*rho^4*dPf*log(rc)+8*log(rc)*m^2*P0*cos(phi)^2*rho^2+12*kappa*m^2*dPf*cos(phi)^2*rho^2-2*rho^6*log(rc)*P0-4*dPf*m^2-4*rho^4*dPf*log(rc)*m^2+4*rho^4*log(rc)*P0*m^2+12*rho^2*dPf*log(rc)*m^2-6*rho^2*log(rc)*P0*m^2-P0*m^3+kappa*m*P0*rho^4-kappa*rho^4*dPf*m-12*dPf*rho^2*log(rho)*m^2+12*rho^2*P0*log(rho)*m^2-4*rho^2*log(rc)*P0*m^3-12*m*log(rho)*dPf*rho^4+12*m*log(rho)*P0*rho^4-2*kappa*log(rc)*m^3*P0+4*rho^2*m*dPf*log(rc)+4*rho^6*P0*log(rho)-4*dPf*rho^6*log(rho)+2*log(rc)*m^3*P0)/rho/log(rc)/G/(-m^2+4*m*rho^2*cos(phi)^2-2*m*rho^2-rho^4);       
        Uy  = -1/8*eta*r1*sin(phi)*(-9*m*rho^4*dPf+9*m*rho^4*P0-kappa*rho^6*dPf+4*m^3*log(rho)*P0-4*m^3*log(rho)*dPf+7*rho^2*P0*m^2+kappa*rho^6*P0+4*rho^2*P0*m^3-3*kappa*m^3*dPf+3*kappa*m^3*P0-4*rho^2*dPf*m^3-4*m*P0*rho^2+4*rho^2*m*dPf-4*rho^2*m*log(rc)*P0-2*kappa*log(rc)*rho^6*P0+4*rho^2*dPf*log(rc)*m^3+4*kappa*log(rc)*m^3*dPf+20*rho^4*m*dPf*cos(phi)^2-2*rho^4*m*log(rc)*P0-20*m*P0*rho^4*cos(phi)^2+12*dPf*m^2*cos(phi)^2*rho^2+16*m*P0*cos(phi)^2*rho^2-12*P0*m^2*cos(phi)^2*rho^2-16*m*dPf*cos(phi)^2*rho^2+dPf*m^3-4*P0*m^2+rho^6*P0-rho^6*dPf-7*rho^2*dPf*m^2+8*kappa*log(rc)*m*P0*rho^4*cos(phi)^2+8*kappa*log(rc)*m^2*dPf*rho^2-6*kappa*log(rc)*m*P0*rho^4-6*kappa*log(rc)*m^2*P0*rho^2+4*kappa*rho^4*dPf*m*cos(phi)^2+4*kappa*log(rc)*m*dPf*rho^4-4*kappa*m*P0*rho^4*cos(phi)^2+8*rho^4*m*log(rc)*P0*cos(phi)^2+16*cos(phi)^2*dPf*rho^2*log(rho)*m^2-4*rho^4*dPf*m^2+4*rho^4*P0*m^2-7*kappa*rho^2*dPf*m^2+7*kappa*m^2*P0*rho^2-16*cos(phi)^2*m*log(rho)*dPf*rho^4-16*cos(phi)^2*rho^2*P0*log(rho)*m^2+16*cos(phi)^2*m*log(rho)*P0*rho^4+8*kappa*log(rc)*m^2*P0*cos(phi)^2*rho^2-16*kappa*log(rc)*m^2*dPf*cos(phi)^2*rho^2-12*kappa*m^2*P0*cos(phi)^2*rho^2-16*log(rc)*m^2*dPf*cos(phi)^2*rho^2-4*rho^6*dPf*log(rc)-4*rho^4*log(rc)*P0+4*rho^4*dPf*log(rc)+8*log(rc)*m^2*P0*cos(phi)^2*rho^2+12*kappa*m^2*dPf*cos(phi)^2*rho^2+2*rho^6*log(rc)*P0+4*dPf*m^2+4*rho^4*dPf*log(rc)*m^2-4*rho^4*log(rc)*P0*m^2+4*rho^2*dPf*log(rc)*m^2-2*rho^2*log(rc)*P0*m^2-P0*m^3+5*kappa*m*P0*rho^4-5*kappa*rho^4*dPf*m-4*dPf*rho^2*log(rho)*m^2+4*rho^2*P0*log(rho)*m^2-4*rho^2*log(rc)*P0*m^3+4*m*log(rho)*dPf*rho^4-4*m*log(rho)*P0*rho^4-2*kappa*log(rc)*m^3*P0+4*rho^2*m*dPf*log(rc)-4*rho^6*P0*log(rho)+4*dPf*rho^6*log(rho)+2*log(rc)*m^3*P0)/rho/log(rc)/G/(m^2-4*m*rho^2*cos(phi)^2+2*m*rho^2+rho^4);

        Ur  =  1/8*r1*eta*(-4*rho^2*dPf*log(rc)*m^2+4*m^2*log(rho)*dPf-4*m^2*log(rho)*P0-4*rho^2*log(rc)*m*P0*cos(2*phi)-2*rho^4*log(rc)*P0+4*rho^4*dPf*log(rc)+4*rho^2*log(rc)*P0*m^2+4*kappa*log(rc)*m*dPf*rho^2*cos(2*phi)+4*rho^2*log(rc)*m*dPf*cos(2*phi)+2*kappa*log(rc)*m^2*P0-4*kappa*log(rc)*m^2*dPf+2*kappa*log(rc)*rho^4*P0-4*m*P0*cos(2*phi)-4*kappa*rho^2*dPf*m*cos(2*phi)+4*kappa*rho^2*P0*m*cos(2*phi)-2*log(rc)*m^2*P0-3*kappa*P0*m^2+kappa*rho^4*dPf+4*rho^2*log(rc)*P0+3*kappa*dPf*m^2-4*dPf*rho^2*log(rc)-kappa*rho^4*P0+4*m*dPf*cos(2*phi)+P0*m^2-dPf*m^2-8*m*dPf*rho^2*cos(2*phi)+8*m*P0*rho^2*cos(2*phi)-4*rho^4*dPf*log(rho)+4*rho^4*P0*log(rho)-4*kappa*log(rc)*rho^2*P0*m*cos(2*phi)-rho^4*P0+rho^4*dPf-4*rho^2*P0*m^2+4*rho^2*dPf*m^2)/(-2*m*rho^2*cos(2*phi)+rho^4+m^2)^(1/2)/rho/G/log(rc);
        Ut  = -1/4*r1*eta*m*sin(2*phi)*(2*dPf*rho^2*log(rc)-kappa*rho^2*dPf+rho^2*kappa*P0+2*kappa*log(rc)*dPf*rho^2-rho^2*P0+rho^2*dPf-2*dPf+2*P0-4*rho^2*dPf*log(rho)+4*rho^2*P0*log(rho))/(-2*m*rho^2*cos(2*phi)+rho^4+m^2)^(1/2)/rho/G/log(rc);

        Pf  = P0 + dPf - dPf*log(rho)/log(rc);
        Sxx =  1/2*(((-2*rho.^2+1+rho.^4).*Srr+(-2*rho.^2-1-rho.^4).*Stt).*cos(2*phi)+(-2*rho.^2+1+rho.^4).*Srr+(2*rho.^2+1+rho.^4).*Stt+(-2*rho.^4 .*sin(2*phi)+2*sin(2*phi)).*Srt)./(-2*rho.^2 .*cos(2*phi)+rho.^4+1);
        Syy = -1/2*(((2*rho.^2+1+rho.^4).*Srr+(-rho.^4+2*rho.^2-1).*Stt).*cos(2*phi)+(-2*rho.^2-1-rho.^4).*Srr+(-rho.^4+2*rho.^2-1).*Stt+(-2*rho.^4 .*sin(2*phi)+2*sin(2*phi)).*Srt)./(-2*rho.^2 .*cos(2*phi)+rho.^4+1);
        Sxy =  1/2*((2+2*rho.^4).*Srt.*cos(2*phi)+(-sin(2*phi)+rho.^4 .*sin(2*phi)).*Srr+(sin(2*phi)-rho.^4 .*sin(2*phi)).*Stt-4*Srt.*rho.^2)./(-2*rho.^2 .*cos(2*phi)+rho.^4+1);
        Pt  = -1/2*(Sxx + Syy) 
    end   
    return (ux=Ux, uy=Uy, ur=Ur, ut=Ut, pt=Pt, pf=Pf, sxx=Sxx, syy=Syy, sxy=Sxy)
end

@views function main(nc, Ωl, Ωη)

    # Independant
    len      = 20.              # Box size
    ϕ0       = 1e-6
    # Dependant
    r_in     = 1.0        # Inclusion radius 
    r_out    = 10*r_in
    ε̇        = 0.0    # Background strain rate
    
    # Set Rozhko values for fluid pressure
    G_anal = 1.0
    ν_anal = 0.25
    K      = 2/3*G_anal*(1+ν_anal)/(1-2ν_anal) 

    materials = ( 
        oneway       = true,
        compressible = true,
        n     = [1.0 1.0  1.0],
        ηs0   = [1e0  1e0*1e-6  1e0*1e-6], 
        ηb    = [K  K*1e6   K*1e-6]./(1-ϕ0),
        G     = [1e30 1e30 1e30], 
        Kd    = [1e30 1e30 1e30],
        Ks    = [1e30 1e30 1e30],
        KΦ    = [1e30 1e30 1e30],
        Kf    = [1e30 1e30 1e30],
        k_ηf0 = [1e0 1e0 1e0],
    )

    # nondim 
    m      = 0.0   # 0 - circle, 0.5 - ellipse, 1 - cut 
    # dependent scales
    Pf_out = 0.    # Fluid pressure on external boundary, Pa
    dPf   = 1.0   # Fluid pressure on cavity - Po    
    Δt0   = 1e0
    nt    = 1

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )
    
    # Resolution
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)
    
    # Intialise field
    L   = (x=len, y=len)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (c  =  ones(size_c...), v  =  ones(size_v...) )
    ϕ   = (c=ϕ0.*ones(size_c...), v=ϕ0.*ones(size_c...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    Dc      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), x =ones(Int64, size_x...), y=ones(Int64, size_y...) )  # phase on velocity points
    P       = (t=zeros(size_c...), f=zeros(size_c...))
    P0      = (t=zeros(size_c...), f=zeros(size_c...))
    ΔP      = (t=zeros(size_c...), f=zeros(size_c...))

    xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc  = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc  = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    xce = LinRange(-L.x/2-Δ.x/2, L.x/2+Δ.x/2, nc.x+2)
    yce = LinRange(-L.y/2-Δ.y/2, L.y/2+Δ.y/2, nc.y+2)

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx]  .= :in       
    type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
    type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
    type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
    type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy]  .= :in       
    type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
    type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in
    # -------- Pf -------- #
    type.Pf[2:end-1,2:end-1] .= :in
    type.Pf[1,:]             .= :Dirichlet 
    type.Pf[end,:]           .= :Dirichlet 
    type.Pf[:,1]             .= :Dirichlet
    type.Pf[:,end]           .= :Dirichlet

    # Add a constrant pressure within a circular region
    @views type.Pf[inx_c,  iny_c ][(xc.^2 .+ (yc').^2) .<= r_in^2 ] .= :constant
    @views type.Pf[inx_c,  iny_c ][(xc.^2 .+ (yc').^2) .>= r_out^2] .= :constant
    
    @views type.Vx[inx_Vx, iny_Vx][(xv.^2 .+ (yc').^2) .<= r_in^2 ] .= :constant
    @views type.Vx[inx_Vx, iny_Vx][(xv.^2 .+ (yc').^2) .>= r_out^2] .= :constant
    
    @views type.Vy[inx_Vy, iny_Vy][(xc.^2 .+ (yv').^2) .<= r_in^2 ] .= :constant
    @views type.Vy[inx_Vy, iny_Vy][(xc.^2 .+ (yv').^2) .>= r_out^2] .= :constant
    
    # @views type.Pt[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= r_in^2] .= :constant
    # @views type.Pt[inx_c, iny_c][(xc.^2 .+ (yc').^2) .>= r_out^2] .= :constant
    
    #--------------------------------------------#

    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    P.f[inx_c, iny_c][(xc.^2 .+ (yc').^2) .< r_in^2]  .= dPf
    P.f[inx_c, iny_c][(xc.^2 .+ (yc').^2) .> r_out^2] .= Pf_out

    phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .< r_in^2 ] .= 2
    phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .> r_out^2] .= 3
    phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .< r_in^2 ] .= 2
    phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .> r_out^2] .= 3
    
    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)

    Ur_ana = zero(BC.Pf)
    Ux_ana = zero(BC.Pf)
    Ut_ana = zero(BC.Pf)
    Ux_ana = zero(BC.Vx)
    Uy_ana = zero(BC.Vy)
    Pf_ana = zero(BC.Pf)
    Pt_ana = zero(BC.Pf)
    ϵ_Ur   = zero(BC.Pf)
    ϵ_Pf   = zero(BC.Pf)
    ϵ_Ux   = zero(BC.Vx)

    for i=1:size(BC.Pf,1), j=1:size(BC.Pf,2)
        # coordinate transform
        ro  = sqrt(xce[i]^2 + yce[j]^2)
        phi = atan(yce[j], xce[i])
        sol = Rozhko2008(ro, phi, r_in, r_out, Pf_out, dPf, m, G_anal, ν_anal)
        BC.Pf[i,j]  = sol.pf
        Pf_ana[i,j] = sol.pf
        Pt_ana[i,j] = sol.pf
        Ur_ana[i,j] = sol.ur
        Ut_ana[i,j] = sol.ut
    end

    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)# nc.x+3, nc.y+4
    yvx  = LinRange(-L.y/2-3*Δ.y/2, L.y/2+3*Δ.y/2, nc.y+4)
    for i=1:size(BC.Vx,1), j=1:size(BC.Vx,2)
        # coordinate transform
        ro  = sqrt(xvx[i]^2 + yvx[j]^2)
        phi = atan(yvx[j], xvx[i])
        sol = Rozhko2008(ro, phi, r_in, r_out, Pf_out, dPf, m, G_anal, ν_anal)
        BC.Vx[i,j] = sol.ux
        V.x[i,j]   = sol.ux
        Ux_ana[i,j] = sol.ux
    end

    xvy = LinRange(-L.x/2-3*Δ.x/2, L.x/2+3*Δ.x/2, nc.x+4)# nc.x+3, nc.y+4
    yvy  = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    for i=1:size(BC.Vy,1), j=1:size(BC.Vy,2)
        # coordinate transform
        ro  = sqrt(xvy[i]^2 + yvy[j]^2)
        phi = atan(yvy[j], xvy[i])
        sol = Rozhko2008(ro, phi, r_in, r_out, Pf_out, dPf, m, G_anal, ν_anal)
        BC.Vy[i,j] = sol.uy
        V.y[i,j]   = sol.uy
        Uy_ana[i,j] = sol.uy
    end

    # Equation Fields
    number = Fields(
        fill(0, (nc.x+3, nc.y+4)),
        fill(0, (nc.x+4, nc.y+3)),
        fill(0, (nc.x+2, nc.y+2)),
        fill(0, (nc.x+2, nc.y+2)),
    )
    Numbering!(number, type, nc)

    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([0 1 0; 1 1 1; 0 1 0]),                 @SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]), @SMatrix([0 1 0;  0 1 0]),        @SMatrix([0 1 0;  0 1 0])), 
        Fields(@SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]),  @SMatrix([0 1 0; 1 1 1; 0 1 0]),                @SMatrix([0 0; 1 1; 0 0]),        @SMatrix([0 0; 1 1; 0 0])),
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                    @SMatrix([1])),
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                    @SMatrix([1 1 1; 1 1 1; 1 1 1])),
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    nPf   = maximum(number.Pf)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt), ExtendableSparseMatrix(nVx, nPf)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt), ExtendableSparseMatrix(nVy, nPf)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt), ExtendableSparseMatrix(nPt, nPf)),
        Fields(ExtendableSparseMatrix(nPf, nVx), ExtendableSparseMatrix(nPf, nVy), ExtendableSparseMatrix(nPf, nPt), ExtendableSparseMatrix(nPf, nPf)),
    )

    time = 0.0
    
    for it=1:nt

        time += Δ.t
        @printf("Step %04d --- time = %1.3f \n", it, time)

        # Swap old values 
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        P0.t  .= P.t
        P0.f  .= P.f

        # #--------------------------------------------#
        # Residual check
        TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, type, BC, materials, phases, Δ)
        ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                    ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, Φ0, 𝐷, phases, materials, number, type, BC, nc, Δ)

        ResidualContinuity2D!(R, V, P, P0, ϕ, phases, materials, number, type, BC, nc, Δ) 
        ResidualFluidContinuity2D!(R, V, P, P0, ϕ, phases, materials, number, type, BC, nc, Δ) 

        @info "Residuals"
        @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
        @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
        @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
        @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

        # Set global residual vector
        r = zeros(nVx + nVy + nPt + nPf)
        SetRHS!(r, R, number, type, nc)

        # #--------------------------------------------#
        # Assembly
        @info "Assembly, ndof  = $(nVx + nVy + nPt + nPf)"
        AssembleMomentum2D_x!(M, V, P, P0, ΔP, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
        AssembleMomentum2D_y!(M, V, P, P0, ΔP, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
        AssembleContinuity2D!(M, V, P, P0, ϕ, phases, materials, number, pattern, type, BC, nc, Δ)
        AssembleFluidContinuity2D!(M, V, P, P0, ϕ, phases, materials, number, pattern, type, BC, nc, Δ)

    # Two-phases operator as block matrix
    𝑀 = [
        M.Vx.Vx M.Vx.Vy M.Vx.Pt M.Vx.Pf;
        M.Vy.Vx M.Vy.Vy M.Vy.Pt M.Vy.Pf;
        M.Pt.Vx M.Pt.Vy M.Pt.Pt M.Pt.Pf;
        M.Pf.Vx M.Pf.Vy M.Pf.Pt M.Pf.Pf;
    ]

    @info "System symmetry"
    𝑀diff = 𝑀 - 𝑀'
    dropzeros!(𝑀diff)
    @show norm(𝑀diff)

    #--------------------------------------------#
    # Direct solver 
    @time dx = - 𝑀 \ r

    #--------------------------------------------#
    UpdateSolution!(V, P, dx, number, type, nc)

    #--------------------------------------------#

    # Residual check
    TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, type, BC, materials, phases, Δ)
    ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
    ResidualContinuity2D!(R, V, P, P0, ϕ, phases, materials, number, type, BC, nc, Δ) 
    ResidualFluidContinuity2D!(R, V, P, P0, ϕ, phases, materials, number, type, BC, nc, Δ) 

    @info "Residuals"
    @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
    @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
    @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
    @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

    #--------------------------------------------#

    Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])
    Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])
    Vs   = sqrt.( Vxsc.^2 .+ Vysc.^2)
    Vxf  = -materials.k_ηf0[1]*diff(P.f, dims=1)/Δ.x
    Vyf  = -materials.k_ηf0[1]*diff(P.f, dims=2)/Δ.y
    Vyfc = 0.5*(Vyf[1:end-1,:] .+ Vyf[2:end,:])
    Vxfc = 0.5*(Vxf[:,1:end-1] .+ Vxf[:,2:end])
    Vf   = sqrt.( Vxfc.^2 .+ Vyfc.^2)

    Vr_viz  = zero(Vxsc)
    Vt_viz  = zero(Vxsc)
    Pt_viz = copy(P.t)
    Pf_viz = copy(P.f)

    for i in 1:length(xce), j in 1:length(yce)

        r = sqrt.(xce[i].^2 .+ yce[j].^2)
        t = atan.(yce[j], xce[i])

        J = [cos(t) sin(t);    
             -sin(t) cos(t)]
        V_cart = [Vxsc[i,j]; Vysc[i,j]]
        V_pol  =  J*V_cart

        Vr_viz[i,j] = V_pol[1]
        Vt_viz[i,j] = V_pol[2]

        if (xce[i].^2 .+ yce[j].^2) <= r_in^2 ||  (xce[i].^2 .+ yce[j].^2) >= r_out^2
            Vr_viz[i,j] = NaN
            Vt_viz[i,j] = NaN
            Pf_viz[i,j] = NaN
            Pt_viz[i,j] = NaN
            Ur_ana[i,j] = NaN
            Ut_ana[i,j] = NaN
        else
            ϵ_Ur[i,j] = abs(Ur_ana[i,j] - Vr_viz[i,j] )
            ϵ_Pf[i,j] = abs(Pf_ana[i,j] - P.f[i,j])
        end
        
    end

    for i=1:size(BC.Vx,1), j=1:size(BC.Vx,2)
        ro  = sqrt(xvx[i]^2 + yvx[j]^2)
        if ro <= r_in || ro >= r_out
            # Vx[i,j]     = NaN
        else
            ϵ_Ux[i,j] = abs(Ux_ana[i,j] - V.x[i,j])
        end
    end

    @show mean(ϵ_Ur)
    @show mean(ϵ_Ux)
    @show mean(ϵ_Pf)

    p1 = heatmap(xc, yc, Vs[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Vs")
    p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, title="Ux", xlims=(-5,5), ylims=(-5,5))
    p2 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, title="Uy", xlims=(-5,5), ylims=(-5,5))
    p1 = heatmap(xce, yce, Vr_viz', aspect_ratio=1, title="Ur", c=:jet)
    p2 = heatmap(xce, yce, Vt_viz', aspect_ratio=1, title="Ut", c=:jet)
    p3 = heatmap(xc, yc, Pt_viz[inx_c,iny_c]',   aspect_ratio=1, title="Pt", c=:jet)
    p4 = heatmap(xc, yc, Pf_viz[inx_c,iny_c]',   aspect_ratio=1, title="Pf", c=:jet)
    display(plot(p4, p3, p1, p2))

    ymid = Int64(floor(nc.y/2))
    p5 = plot(xlabel="x", ylabel="Pf")
    p5 = scatter!(xc, P.f[2:end-1, ymid], label="numerics")
    p5 = plot!(xc, BC.Pf[2:end-1, ymid], label="analytics")
    p6 = plot(xlabel="x", ylabel="Ur")
    p6 = scatter!(xc, Vr_viz[2:end-1, ymid].*Δ.t, label="numerics")
    p6 = plot!(xc, Ur_ana[2:end-1, ymid], label="analytics")
    # p6 = scatter!(xv, V.x[inx_Vx,iny_Vx][:,ymid].*Δ.t, label="numerics", markershape=:x)
    # p6 = plot!(xv, Ux_ana[inx_Vx,iny_Vx][:,ymid], label="analytics")

    display(plot(p5, p6))

    end

    #--------------------------------------------#

    # return P, Δ, (c=xc, v=xv), (c=yc, v=yv)
end

##################################
function Run()

    nc = (x=100, y=100)

    # Mode 0   
    Ωl = 0.1
    Ωη = 10.
    main(nc,  Ωl, Ωη);

end

Run()
