using StagFDTools, StagFDTools.TwoPhases, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf, JLD2, ExactFieldSolutions
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use

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
    #     ∂ε̇II∂τII = Enzyme.jacobian(Enzyme.Forward, StrainRateTrial, τII, G, Δ.t, B, n)
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
            jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.c[i,j]), Const(Δ))
            
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
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector!, ε̇vec, Const(materials), Const(phases.v[i+1,j+1]), Const(Δ))

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

@views function main(nc, Ωl, Ωη)

    # Independant
    len      = 10.              # Box size
    ϕ0       = 1e-6
    # Dependant
    ε̇        = 1.0    # Background strain rate

    params = (mm = 1.0, mc = 100, rc = 2.0, gr = 0.0, er = ε̇)
    
    materials = ( 
        oneway       = false,
        compressible = true,
        n     = [1.0  1.0],
        ηs0   = [1e0  100], 
        ηb    = [1e30 1e30]./(1-ϕ0),
        G     = [1e30 1e30], 
        Kd    = [1e30 1e30],
        Ks    = [1e30 1e30],
        Kϕ    = [1e30 1e30],
        Kf    = [1e30 1e30],
        k_ηf0 = [1e0 1e0 1e0],
    )

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
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
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

    #--------------------------------------------#

    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'

    phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .< params.rc^2 ] .= 2
    phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .< params.rc^2 ] .= 2
    
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

    Vx_ana = zero(BC.Vx)
    Vy_ana = zero(BC.Vy)
    Pt_ana = zero(BC.Pt)
    ϵ_Pt   = zero(BC.Pt)
    ϵ_Vx   = zero(BC.Vx)

    for i=1:size(BC.Pf,1), j=1:size(BC.Pf,2)
        # coordinate transform
        sol = Stokes2D_Schmid2003( [xce[i], yce[j]]; params )
        Pt_ana[i,j] = sol.p
    end

    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)# nc.x+3, nc.y+4
    yvx  = LinRange(-L.y/2-3*Δ.y/2, L.y/2+3*Δ.y/2, nc.y+4)
    for i=1:size(BC.Vx,1), j=1:size(BC.Vx,2)
        # coordinate transform
        sol = Stokes2D_Schmid2003( [xvx[i], yvx[j]]; params )
        BC.Vx[i,j] = sol.V[1]
        V.x[i,j]   = sol.V[1]
        Vx_ana[i,j] = sol.V[1]
    end

    xvy = LinRange(-L.x/2-3*Δ.x/2, L.x/2+3*Δ.x/2, nc.x+4)# nc.x+3, nc.y+4
    yvy  = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    for i=1:size(BC.Vy,1), j=1:size(BC.Vy,2)
        # coordinate transform
        sol = Stokes2D_Schmid2003( [xvy[i], yvy[j]]; params )
        BC.Vy[i,j] = sol.V[2]
        V.y[i,j]   = sol.V[2]
        Vy_ana[i,j] = sol.V[2]
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
        ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
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

    @show 1e-7/Δ.x/Δ.y
    γ  = 0.0006399999999999999
    PP = spdiagm(γ*ones(nPt))

    𝑀 = [
        M.Vx.Vx M.Vx.Vy M.Vx.Pt 0*M.Vx.Pf;
        M.Vy.Vx M.Vy.Vy M.Vy.Pt 0*M.Vy.Pf;
        M.Pt.Vx M.Pt.Vy PP 0*M.Pt.Pf;
        0*M.Pf.Vx 0*M.Pf.Vy 0*M.Pf.Pt M.Pf.Pf;
    ]

    @time dx = - 𝑀 \ r

    # # M2Di solver
    # fv    = -r[1:(nVx+nVy)]
    # fpt   = -r[(nVx+nVy+1):(nVx+nVy+nPt)]
    # fpf   = -r[(nVx+nVy+nPt+1):end]
    # dv    = zeros(nVx+nVy)
    # dpt   = zeros(nPt)
    # dpf   = zeros(nPf)
    # rv    = zeros(nVx+nVy)
    # rpt   = zeros(nPt)
    # rpf   = zeros(nPf)
    # rv_t  = zeros(nVx+nVy)
    # rpt_t = zeros(nPt)
    # s     = zeros(nPf)
    # ddv   = zeros(nVx+nVy)
    # ddpt  = zeros(nPt)
    # ddpf  = zeros(nPf)


    # Jvv  = [M.Vx.Vx M.Vx.Vy;
    #         M.Vy.Vx M.Vy.Vy]
    # Jvp  = [M.Vx.Pt;
    #         M.Vy.Pt]
    # Jpv  = [M.Pt.Vx M.Pt.Vy]
    # Jpp  = M.Pt.Pt
    # Jppf = M.Pt.Pf
    # Jpfv = [M.Pf.Vx M.Pf.Vy]
    # Jpfp = M.Pf.Pt
    # Jpf  = M.Pf.Pf
    # Kvv  = Jvv

    # @time begin 
    #     # Pre-conditionning (~Jacobi)

    #     γ = 1e-4
    #     Γ = spdiagm(γ*ones(nPt))
    #     ΓV = spdiagm(γ*ones(nVx+nVy))

    #     Jpv_t  = Jpv  - Jppf*spdiagm(1 ./ diag(Jpf  ) )*Jpfv
    #     Jpp_t  = Jpp  - Jppf*spdiagm(1 ./ diag(Jpf  ) )*Jpfp

    #     Jvv_t  = Kvv  - Jvp *spdiagm(1 ./ diag(Jpp_t) )*Jpv 
    #     @show mean(diag(Jpp))
    #     @show mean(diag(Jvv_t))
    #    #  Jpf_h  = cholesky(Hermitian(SparseMatrixCSC(Jpf)), check = false  )        # Cholesky factors
    #    #  Jvv_th = cholesky(Hermitian(SparseMatrixCSC(Jvv_t)), check = false)        # Cholesky factors
    #    Jpf_h  = cholesky(Hermitian(SparseMatrixCSC(Jpf )) )        # Cholesky factors
    #    Jvv_th = cholesky(Hermitian(SparseMatrixCSC(Jvv_t .+  ΓV )))        # Cholesky factors
    #    Jpp_th = spdiagm(1 ./diag(Jpp_t));             # trivial inverse
    #     @views for itPH=1:15
    #         rv    .= -( Jvv*dv  + Jvp*dpt             - fv  )
    #         rpt   .= -( Jpv*dv  + Jpp*dpt  + Jppf*dpf - fpt )
    #         rpf   .= -( Jpfv*dv + Jpfp*dpt + Jpf*dpf  - fpf )


            
    #         s     .= Jpf_h \ rpf
    #         rpt_t .= -( Jppf*s - rpt)
    #         s     .=    Jpp_th*rpt_t
    #         rv_t  .= -( Jvp*s  - rv )
    #         ddv   .= Jvv_th \ rv_t
    #         s     .= -( Jpv_t*ddv - rpt_t )
    #         ddpt  .=    Jpp_th*s 
    #         s     .= -( Jpf*ddpt + Jpfv*ddv - rpf )
    #         ddpf  .= Jpf_h \ s 
    #         dv   .+= ddv
    #         dpt  .+= ddpt
    #         dpf  .+= ddpf
            
            
    #         @printf("  --- iteration %d --- \n",itPH);
    #         @printf("  ||res.v ||=%2.2e\n", norm(rv)/ 1)
    #         @printf("  ||res.pt||=%2.2e\n", norm(rpt)/1)
    #         @printf("  ||res.pf||=%2.2e\n", norm(rpf)/1)
    #     #     if ((norm(rv)/length(rv)) < tol_linv) && ((norm(rpt)/length(rpt)) < tol_linpt) && ((norm(rpf)/length(rpf)) < tol_linpf), break; end
    #     #     if ((norm(rv)/length(rv)) > (norm(rv0)/length(rv0)) && norm(rv)/length(rv) < tol_glob && (norm(rpt)/length(rpt)) > (norm(rpt0)/length(rpt0)) && norm(rpt)/length(rpt) < tol_glob && (norm(rpf)/length(rpf)) > (norm(rpf0)/length(rpf0)) && norm(rpf)/length(rpf) < tol_glob),
    #     #         if noisy>=1, fprintf(' > Linear residuals do no converge further:\n'); break; end
    #     #     end
    #     #     rv0=rv; rpt0=rpt; rpf0=rpf; if (itPH==nPH), nfail=nfail+1; end
    #     end
    # end

    
    # dx = zeros(nVx + nVy + nPt + nPf)
    # dx[1:(nVx+nVy)] .= dv
    # dx[(nVx+nVy+1):(nVx+nVy+nPt)] .= dpt
    # dx[(nVx+nVy+nPt+1):end] .= dpf

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

    P.t .-= mean(P.t) 




    for i in 1:length(xce), j in 1:length(yce)
       
            ϵ_Pt[i,j] = abs(Pt_ana[i,j] - P.t[i,j])
        
    end

    for i=1:size(BC.Vx,1), j=1:size(BC.Vx,2)

            ϵ_Vx[i,j] = abs(Vx_ana[i,j] - V.x[i,j])
    end

    @show mean(ϵ_Vx)
    @show mean(ϵ_Pt)

    P.t[P.t.>maximum(Pt_ana)] .= maximum(Pt_ana)
    P.t[P.t.<minimum(Pt_ana)] .= minimum(Pt_ana)

    p1 = heatmap(xc, yc, Vs[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Vs")
    p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, title="Ux", xlims=(-5,5), ylims=(-5,5))
    p2 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, title="Uy", xlims=(-5,5), ylims=(-5,5))
    p1 = heatmap(xce, yce, Vr_viz', aspect_ratio=1, title="Ur", c=:jet)
    p2 = heatmap(xce, yce, Vt_viz', aspect_ratio=1, title="Ut", c=:jet)
    p3 = heatmap(xc, yc, P.t[inx_c,iny_c]', colorrange=(-3.5, 3.5),   aspect_ratio=1, title="Pt", c=:jet)
    p4 = heatmap(xc, yc, Pt_ana[inx_c,iny_c]',   aspect_ratio=1, title="Pt_ana", c=:jet)
    display(plot(p4, p3, p1, p2))

    end

    #--------------------------------------------#

    return P, Δ, (c=xc, v=xv), (c=yc, v=yv)
end

##################################
function Run()

    nc = (x=100, y=100)

    # Mode 0   
    Ωl = 0.1
    Ωη = 10.
    P, Δ, x, y = main(nc,  Ωl, Ωη);
    # save("/Users/tduretz/PowerFolders/_manuscripts/TwoPhasePressure/benchmark/SchmidTest.jld2", "x", x, "y", y, "P", P )

end

Run()
