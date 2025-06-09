using StagFDTools, StagFDTools.ThermoMechanics, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf, JLD2
import Statistics:mean
using Enzyme 

# This example shows how mechanical loading (compaction) leads to heating
# The temperature is predicted numerically and exactly using the adiabatic relation:
# ΔT = α*T*ΔP / (ρ*cp - α*ΔP) 

@views function main(nc)

    sc = (L=1e-2, t=1e7, σ=1e6, T=1000)
    m  = sc.σ * sc.L * sc.t^2.0
    J  = m * sc.L^2.0 / sc.t^2.0
    W  = J/sc.t

    nt           = 10
    ηi           = 1e18 / (sc.σ*sc.t)
    ηinc         = 1e17 / (sc.σ*sc.t)
    Gi           = 1e10 / sc.σ  
    Ginc         = Gi/5
    Ki           = 1e11 / sc.σ 
    αi           = 1e-5 / (1/sc.T)
    Δt0          = ηi/Gi/4.0/1000
    ki           = 3.0    / (W/sc.L/sc.T)
    ρi           = 3000.0 / (m/sc.L^3)
    ρinc         = 2800.0 / (m/sc.L^3)
    cpi          = 1000.0 / (J/m/sc.T)
    ε̇            = -1e-7    / (1/sc.t)
    L            = 2.0/100    / sc.L
    r            = 0.4/100    / sc.L
    T_ini        = 473.0  / sc.T
    P_ini        = 1e6    / sc.σ

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 ε̇] )

    # Material parameters
    materials = ( 
        oneway       = false,
        compressible = true,
        Dzz          = ε̇,
        n            = [1.0  1.0],
        ηs0          = [ηi  ηinc], 
        G            = [Gi  Ginc], 
        K            = [Ki  Ki  ],
        α            = [αi  αi  ],
        k            = [ki  ki  ],
        cp           = [cpi cpi ],
        ρr           = [ρi  ρinc],
    )
 
    # Resolution
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

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
    # -------- T -------- #
    type.T[2:end-1,2:end-1] .= :in
    type.T[1,:]             .= :Dirichlet 
    type.T[end,:]           .= :Dirichlet 
    type.T[:,1]             .= :Dirichlet
    type.T[:,end]           .= :Dirichlet
    
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
    nT    = maximum(number.T )
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt), ExtendableSparseMatrix(nPt, nT )),
        Fields(ExtendableSparseMatrix(nT , nVx), ExtendableSparseMatrix(nT , nVy), ExtendableSparseMatrix(nT , nPt), ExtendableSparseMatrix(nT , nT )),
    )

    # #--------------------------------------------#
    # Intialise field
    L   = (x=L, y=L)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), T=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (c  =  ones(size_c...), v  =  ones(size_v...) )
    T   = (c  =  T_ini.*ones(size_c...), v  =  T_ini.*ones(size_v...) )
    T0  = (c  =  T_ini.*ones(size_c...), v  =  T_ini.*ones(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), zz = zeros(size_c...), xy = zeros(size_v...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), zz = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), zz = zeros(size_c...), xy = zeros(size_v...) )
    Dc      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), x =ones(Int64, size_x...), y=ones(Int64, size_y...) )  # phase on velocity points
    P       = (t=P_ini*ones(size_c...),)
    P0      = (t=P_ini*ones(size_c...),)
    ΔP      = (t=zeros(size_c...),)

    xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc  = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc  = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)
    xvy = LinRange(-L.x/2-3Δ.x/2, L.x/2+3Δ.x/2, nc.x+4)
    yvy = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    yvx = LinRange(-L.y/2-3Δ.y/2, L.y/2+3Δ.y/2, nc.y+4)

    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'

    Xc = xc .+ 0*yc'
    Yc = 0*xc .+ yc'
    Xv = xv .+ 0*yv'
    Yv = 0*xv .+ yv'
    α  = 30.
    ax = 1
    ay = 1/4
    X_tilt = cosd(α).*Xc .- sind(α).*Yc
    Y_tilt = sind(α).*Xc .+ cosd(α).*Yc
    phases.c[inx_c, iny_c][(X_tilt.^2 ./ax.^2 .+ (Y_tilt).^2 ./ay^2) .< r^2 ] .= 2
    X_tilt = cosd(α).*Xv .- sind(α).*Yv
    Y_tilt = sind(α).*Xv .+ cosd(α).*Yv
    phases.v[inx_v, iny_v][(X_tilt.^2 ./ax.^2 .+ (Y_tilt).^2 ./ay^2) .< r^2 ] .= 2

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), T = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal)  .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal)  .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)

    #--------------------------------------------#

    probes = (
            T   = zeros(nt),
            Pt  = zeros(nt),
            t   = zeros(nt),
    )
    
    for it=1:nt

        T0.c  .= T.c
        P0.t  .= P.t
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.zz .= τ.zz
        τ0.xy .= τ.xy

        # ramp up boundary t
        BC.T .= T_ini .+ 5*it/sc.T

        for iter=1:5

            # Residual check
            TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, T, P, ΔP, type, BC, materials, phases, Δ)
            ResidualMomentum2D_x!(R, V, T, T0, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(R, V, T, T0, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualContinuity2D!(R, V, T, T0, P, P0, phases, materials, number, type, BC, nc, Δ) 
            ResidualHeatDiffusion2D!(R, V, T, T0, P, P0, phases, materials, number, type, BC, nc, Δ) 

            # Set global residual vector
            r = zeros(nVx + nVy + nPt + nT )
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @info "Assembly, ndof  = $(nVx + nVy + nPt + nT )"
            AssembleMomentum2D_x!(M, V, T, T0, P, P0, ΔP, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleMomentum2D_y!(M, V, T, T0, P, P0, ΔP, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleContinuity2D!(M, V, T, T0, P, P0, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleHeatDiffusion2D!(M, V, T, T0, P, P0, phases, materials, number, pattern, type, BC, nc, Δ)

            # Two-phases operator as block matrix
            𝑀 = [
                M.Vx.Vx M.Vx.Vy M.Vx.Pt M.Vx.T;
                M.Vy.Vx M.Vy.Vy M.Vy.Pt M.Vy.T;
                M.Pt.Vx M.Pt.Vy M.Pt.Pt M.Pt.T;
                M.T.Vx  M.T.Vy  M.T.Pt  M.T.T;
            ]

            @info "System symmetry"
            𝑀diff = 𝑀 - 𝑀'
            dropzeros!(𝑀diff)
            @show norm(𝑀diff)

            #--------------------------------------------#
            # Direct solver 
            @time dx = - 𝑀 \ r

            # # M2Di solver
            # fv    = -r[1:(nVx+nVy)]
            # fpt   = -r[(nVx+nVy+1):(nVx+nVy+nPt)]
            # fpf   = -r[(nVx+nVy+nPt+1):end]
            # dv    = zeros(nVx+nVy)
            # dpt   = zeros(nPt)
            # dpf   = zeros(nT )
            # rv    = zeros(nVx+nVy)
            # rpt   = zeros(nPt)
            # rpf   = zeros(nT )
            # rv_t  = zeros(nVx+nVy)
            # rpt_t = zeros(nPt)
            # s     = zeros(nT )
            # ddv   = zeros(nVx+nVy)
            # ddpt  = zeros(nPt)
            # ddpf  = zeros(nT )

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
            #     # γ = 1e-8
            #     # Γ = spdiagm(γ*ones(nPt))
            #     # Pre-conditionning (~Jacobi)
            #     Jpv_t  = Jpv  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfv  
            #     Jpp_t  = Jpp  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfp  #.+ Γ
            #     Jvv_t  = Kvv  - Jvp *spdiagm(1 ./ diag(Jpp_t))*Jpv 
            #     @show typeof(SparseMatrixCSC(Jpf))
            #     Jpf_h  = cholesky(Hermitian(SparseMatrixCSC(Jpf)), check = false  )        # Cholesky factors
            #     Jvv_th = cholesky(Hermitian(SparseMatrixCSC(Jvv_t)), check = false)        # Cholesky factors
            #     Jpp_th = spdiagm(1 ./diag(Jpp_t));             # trivial inverse
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
            #         s     .= -( Jpfp*ddpt + Jpfv*ddv - rpf )
            #         ddpf  .= Jpf_h \ s
            #         dv   .+= ddv
            #         dpt  .+= ddpt
            #         dpf  .+= ddpf
            #         @printf("  --- iteration %d --- \n",itPH);
            #         @printf("  ||res.v ||=%2.2e\n", norm(rv)/ 1)
            #         @printf("  ||res.pt||=%2.2e\n", norm(rpt)/1)
            #         @printf("  ||res.pf||=%2.2e\n", norm(rpf)/1)
            #     #     if ((norm(rv)/length(rv)) < tol_linv) && ((norm(rpt)/length(rpt)) < tol_linpt) && ((norm(rpf)/length(rpf)) < tol_linT ), break; end
            #     #     if ((norm(rv)/length(rv)) > (norm(rv0)/length(rv0)) && norm(rv)/length(rv) < tol_glob && (norm(rpt)/length(rpt)) > (norm(rpt0)/length(rpt0)) && norm(rpt)/length(rpt) < tol_glob && (norm(rpf)/length(rpf)) > (norm(rpf0)/length(rpf0)) && norm(rpf)/length(rpf) < tol_glob),
            #     #         if noisy>=1, fprintf(' > Linear residuals do no converge further:\n'); break; end
            #     #     end
            #     #     rv0=rv; rpt0=rpt; rpf0=rpf; if (itPH==nPH), nfail=nfail+1; end
            #     end
            # end
            
            # dx = zeros(nVx + nVy + nPt + nT )
            # dx[1:(nVx+nVy)] .= dv
            # dx[(nVx+nVy+1):(nVx+nVy+nPt)] .= dpt
            # dx[(nVx+nVy+nPt+1):end] .= dpf

            #--------------------------------------------#
            UpdateSolution!(V, T, P, dx, number, type, nc)

            #--------------------------------------------#
            # Residual check
            TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, T, P, ΔP, type, BC, materials, phases, Δ)
            ResidualMomentum2D_x!(R, V, T, T0, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(R, V, T, T0, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualContinuity2D!(R, V, T, T0, P, P0, phases, materials, number, type, BC, nc, Δ) 
            ResidualHeatDiffusion2D!(R, V, T, T0, P, P0, phases, materials, number, type, BC, nc, Δ) 

            @info "Residuals"
            @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            @show norm(R.T[inx_c,iny_c])/sqrt(nT )

        end

        probes.T[it]    = mean(T.c[inx_c,iny_c])
        probes.Pt[it]   = mean(P.t[inx_c,iny_c])
        probes.t[it]    = it*Δ.t

        #--------------------------------------------#
        # Post process 
  
        Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])
        Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])
        Vs   = sqrt.( Vxsc.^2 .+ Vysc.^2)
        
        # p1 = heatmap(xc, yc, Vs[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Vs")
        p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]'.*sc.L/sc.t, aspect_ratio=1, xlim=extrema(xc), title="Vx")

        p2 = heatmap(xc, yc, T.c[inx_c,iny_c]'.*sc.T, aspect_ratio=1, xlim=extrema(xc), title="T")

        τxyc = av2D(τ.xy)
        τII  = sqrt.( 0.5.*(τ.xx[inx_c,iny_c].^2 + τ.yy[inx_c,iny_c].^2 + (-τ.xx[inx_c,iny_c]-τ.yy[inx_c,iny_c]).^2) .+ τxyc[inx_c,iny_c].^2 )
        p1   = heatmap(xc, yc, τII'.*sc.σ/1e9,   aspect_ratio=1, xlim=extrema(xc), title="τII")
       
        st = 1
        # divV = diff(V.x[2:end-1,3:end-2], dims=1)/Δ.x  + diff(V.y[3:end-2,2:end-1], dims=2)/Δ.y
        ρ  = @. materials.ρr[phases.c] .*  exp(1/materials.K[phases.c].*P.t - materials.α[phases.c].*T.c)
        p3 = heatmap(xc, yc, ρ[inx_c,iny_c]'.*m/sc.L^3,   aspect_ratio=1, xlim=extrema(xc), title="ρ")
        # p3 = heatmap(xc, yc, divV'*(1/sc.t),   aspect_ratio=1, xlim=extrema(xc), title="div(v)")
        p4 = heatmap(xc, yc, P.t[inx_c,iny_c]'*sc.σ/1e9,   aspect_ratio=1, xlim=extrema(xc), title="Pt")
        # p4 = quiver!(Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end], quiver=(Vxsc[1:st:end,1:st:end], Vysc[1:st:end,1:st:end]), c=:black,  aspect_ratio=1)

        display(plot(p1, p2, p3, p4, layout=(2,2)))
        @show sum(phases.c.==1)
        @show sum(phases.c.==2)
        @show it*Δ.t*sc.t
    end

    #--------------------------------------------#

    return nothing
end

function Run()

    nc = (x=200, y=200)

    main(nc)
    
end

Run()
