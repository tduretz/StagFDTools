using StagFDTools, StagFDTools.TwoPhases, ExtendableSparse, StaticArrays, GLMakie, LinearAlgebra, SparseArrays, Printf, JLD2
import Statistics:mean
using Enzyme  # AD backends you want to use

@views function main(nc)

    sc = (σ=1e7, t=1e10, L=1e3)

    nt     = 1
    Δt0    = 1e10/sc.t
    niter  = 10
    ϵ_nl   = 1e-10

    Φ0     = 0.05
    Φi     = Φ0
    Pi     = 1e6/sc.σ
    ε̇      = 2e-15.*sc.t
    rad    = 2e3/sc.L

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )

    # Material parameters
    materials = ( 
        oneway       = false,
        compressible = true,
        plasticity   = :DruckerPrager,
        n     = [1.0    1.0  ],
        ηs0   = [1e22   1e22 ]/sc.σ/sc.t, 
        ηϕ    = [2e22   2e22 ]/sc.σ/sc.t,
        G     = [3e10   1e10 ]./sc.σ, 
        Kd    = [1e30   1e30 ]./sc.σ,  # not needed
        Ks    = [1e11   1e11 ]./sc.σ,
        Kϕ    = [1e9    1e9  ]./sc.σ,
        Kf    = [1e10   1e-10]./sc.σ, 
        k_ηf0 = [1e-15  1e-15]./(sc.L^2/sc.σ/sc.t),
        ψ     = [10.    10.  ],
        ϕ     = [35.    35.  ],
        C     = [1e70   1e7 ]./sc.σ,
        ηvp   = [0.0    0.0  ]./sc.σ/sc.t,
        cosϕ  = [0.0    0.0  ],
        sinϕ  = [0.0    0.0  ],
        sinψ  = [0.0    0.0  ],
    )

    # For plasticity
    @. materials.cosϕ  = cosd(materials.ϕ)
    @. materials.sinϕ  = sind(materials.ϕ)
    @. materials.sinψ  = sind(materials.ψ)

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
    # -------- Pf -------- #
    type.Pf[2:end-1,2:end-1] .= :in
    type.Pf[1,:]             .= :Neumann 
    type.Pf[end,:]           .= :Neumann 
    type.Pf[:,1]             .= :Neumann
    type.Pf[:,end]           .= :Neumann
    
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
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt), ExtendableSparseMatrix(nPt, nPf)),
        Fields(ExtendableSparseMatrix(nPf, nVx), ExtendableSparseMatrix(nPf, nVy), ExtendableSparseMatrix(nPf, nPt), ExtendableSparseMatrix(nPf, nPf)),
    )

    #--------------------------------------------#
    # Intialise field
    L   = (x=40e3/sc.L, y=20e3/sc.L)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (c  =  ones(size_c...), v  =  ones(size_v...) )
    Φ   = (c=Φi.*ones(size_c...), v=Φi.*ones(size_v...) )
    Φ0  = (c=Φi.*ones(size_c...), v=Φi.*ones(size_v...) )

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
    P       = (t=Pi .* ones(size_c...), f=Pi .* ones(size_c...))
    P0      = (t=zeros(size_c...), f=zeros(size_c...))
    ΔP      = (t=zeros(size_c...), f=zeros(size_c...))

    # Generate grid coordinates 
    x = (min=-L.x/2, max=L.x/2)
    y = (min=-L.y/2, max=L.y/2)
    X = GenerateGrid(x, y, Δ, nc)

    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.c.y' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*X.c.x .+ D_BC[2,2]*X.v.y'

    # for I in CartesianIndices(Φ.c)
    #     i, j = I[1], I[2]
    #     if i>1 && i<size(Φ.c,1) && j>1 && j<size(Φ.c,2)
    #         if (X.c.x[i-1]^2 + X.c.y[j-1]^2) < rad^2
    #             Φ.c[i,j] = 1.1*Φi
    #         end
    #     end 
    # end

    # Set material geometry 
    @views phases.c[inx_c, iny_c][(X.c.x.^2 .+ (X.c.y').^2) .<= rad^2] .= 2
    @views phases.v[inx_v, iny_v][(X.v.x.^2 .+ (X.v.y').^2) .<= rad^2] .= 2


    # Xc = xc .+ 0*yc'
    # Yc = 0*xc .+ yc'
    # Xv = xv .+ 0*yv'
    # Yv = 0*xv .+ yv'
    # α  = 30.
    # # ax = 2
    # # ay = 1/2
    # ax = 1
    # ay = 1
    # X_tilt = cosd(α).*Xc .- sind(α).*Yc
    # Y_tilt = sind(α).*Xc .+ cosd(α).*Yc
    # phases.c[inx_c, iny_c][(X_tilt.^2 ./ax.^2 .+ (Y_tilt).^2 ./ay^2) .< r^2 ] .= 2
    # X_tilt = cosd(α).*Xv .- sind(α).*Yv
    # Y_tilt = sind(α).*Xv .+ cosd(α).*Yv
    # phases.v[inx_v, iny_v][(X_tilt.^2 ./ax.^2 .+ (Y_tilt).^2 ./ay^2) .< r^2 ] .= 2

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), Pf = zeros(size_c...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.v.y[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[1]   .+ D_BC[2,2]*X.v.y)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*X.v.x[end] .+ D_BC[2,2]*X.v.y)
    
    #--------------------------------------------#

    probes = (
        Pe  = zeros(nt),
        Pt  = zeros(nt),
        Pf  = zeros(nt),
        τ   = zeros(nt),
        Φ   = zeros(nt),
        λ̇   = zeros(nt),
        t   = zeros(nt),
    )

    err  = (x = zeros(niter), y = zeros(niter), pt = zeros(niter), pf = zeros(niter))
    
    for it=1:nt

        @printf("\nStep %04d\n", it)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.pt, 0e0)
        fill!(err.pf, 0e0)

        # Swap old values 
        P0.t  .= P.t
        P0.f  .= P.f
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Φ0.c  .= Φ.c 

        for iter=1:niter

            @printf("Iteration %04d\n", iter)

            λ̇.c   .= 0.0
            λ̇.v   .= 0.0

            #--------------------------------------------#
            # Residual check
            TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ0, type, BC, materials, phases, Δ)
            ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualContinuity2D!(R, V, P, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 
            ResidualFluidContinuity2D!(R, V, P, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 

            @show extrema(λ̇.c[inx_c,iny_c])
            @show extrema(λ̇.v[inx_v,iny_v])

            @info "Residuals"
            @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

            err.x[iter]  = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter]  = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.pt[iter] = @views norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            err.pf[iter] = @views norm(R.pf[inx_c,iny_c])/sqrt(nPt)
            max(err.x[iter], err.y[iter], err.pt[iter], err.pf[iter]) < ϵ_nl ? break : nothing

            # Set global residual vector
            r = zeros(nVx + nVy + nPt + nPf)
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @info "Assembly, ndof  = $(nVx + nVy + nPt + nPf)"
            AssembleMomentum2D_x!(M, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleMomentum2D_y!(M, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
            # AssembleMomentum2D_x!(M, V, P, P0, ΔP, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            # AssembleMomentum2D_y!(M, V, P, P0, ΔP, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleContinuity2D!(M, V, P, P0, Φ0, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleFluidContinuity2D!(M, V, P, P0, Φ0, phases, materials, number, pattern, type, BC, nc, Δ)

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
        end

        #--------------------------------------------#

        # Residual check
        TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ0, type, BC, materials, phases, Δ)
        ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualContinuity2D!(R, V, P, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 
        ResidualFluidContinuity2D!(R, V, P, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 

        @info "Residuals - posteriori"
        @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
        @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
        @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
        @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

        #--------------------------------------------#

        # Include plasticity corrections
        P.t .= P.t .+ ΔP.t
        P.f .= P.f .+ ΔP.f
        
        τxyc = av2D(τ.xy)
        τII  = sqrt.( 0.5.*(τ.xx[inx_c,iny_c].^2 + τ.yy[inx_c,iny_c].^2 + (-τ.xx[inx_c,iny_c]-τ.yy[inx_c,iny_c]).^2) .+ τxyc[inx_c,iny_c].^2 )
        ε̇xyc = av2D(ε̇.xy)
        ε̇II  = sqrt.( 0.5.*(ε̇.xx[inx_c,iny_c].^2 + ε̇.yy[inx_c,iny_c].^2 + (-ε̇.xx[inx_c,iny_c]-ε̇.yy[inx_c,iny_c]).^2) .+ ε̇xyc[inx_c,iny_c].^2 )
        
        # Post process 
        @time for i in eachindex(Φ.c)
            Kϕ     = materials.Kϕ[phases.c[i]]
            ηϕ     = materials.ηϕ[phases.c[i]] 
            sinψ   = materials.sinψ[phases.c[i]] 
            dPtdt  = (P.t[i] - P0.t[i]) / Δ.t
            dPfdt  = (P.f[i] - P0.f[i]) / Δ.t
            dΦdt   = 1/Kϕ * (dPfdt - dPtdt) + 1/ηϕ * (P.f[i] - P.t[i]) + λ̇.c[i]*sinψ
            Φ.c[i] = Φ0.c[i] + dΦdt*Δ.t
        end

        Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])
        Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])
        Vs   = sqrt.( Vxsc.^2 .+ Vysc.^2)
        Vxf  = -materials.k_ηf0[1]*diff(P.f, dims=1)/Δ.x
        Vyf  = -materials.k_ηf0[1]*diff(P.f, dims=2)/Δ.y
        Vyfc = 0.5*(Vyf[1:end-1,:] .+ Vyf[2:end,:])
        Vxfc = 0.5*(Vxf[:,1:end-1] .+ Vxf[:,2:end])
        Vf   = sqrt.( Vxfc.^2 .+ Vyfc.^2)

        fig = Figure(fontsize = 20, size = (600, 400) )    
        #-------------------------------------------# 
        ax1 = Axis(fig[1,1], title="τII",  xlabel=L"$x$ [-]",  ylabel=L"$y$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect())
        hm=heatmap!(ax1, X.c.x, X.c.y, τII, colormap=(GLMakie.Reverse(:matter), 1))
        Colorbar(fig[2, 1], hm, label = L"$τII$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )

        # ax1 = Axis(fig[1,1], title="ϕ",  xlabel=L"$x$ [-]",  ylabel=L"$y$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect())
        # hm=heatmap!(ax1, X.c.x, X.c.y, Φ.c[inx_c,iny_c], colormap=(GLMakie.Reverse(:matter), 1))
        # Colorbar(fig[2, 1], hm, label = L"$ϕ$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )

        ax2 = Axis(fig[1,2], title="λ̇.v",  xlabel=L"$x$ [-]",  ylabel=L"$y$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect())
        hm=heatmap!(ax2, X.v.x, X.v.y, λ̇.v[inx_v,iny_v], colormap=(GLMakie.Reverse(:matter), 1))
        Colorbar(fig[2, 2], hm, label = L"$λ̇.v$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
        display(fig)

        # ax2 = Axis(fig[1,2], title="Pt",  xlabel=L"$x$ [-]",  ylabel=L"$y$ [-]", xlabelsize=20, ylabelsize=20, aspect=DataAspect())
        # hm=heatmap!(ax2, X.c.x, X.c.y, P.t[inx_c,iny_c], colormap=(GLMakie.Reverse(:matter), 1))
        # Colorbar(fig[2, 2], hm, label = L"$Pt$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
        # display(fig)

        #--------------------------------------------#
        probes.Pe[it]   = mean(P.t[inx_c,iny_c] .- P.f[inx_c,iny_c])*sc.σ
        probes.Pt[it]   = mean(P.t[inx_c,iny_c])*sc.σ
        probes.Pf[it]   = mean(P.f[inx_c,iny_c])*sc.σ
        probes.τ[it]    = mean(τII)*sc.σ
        probes.Φ[it]    = mean(Φ.c[inx_c,iny_c])
        probes.λ̇[it]    = mean(λ̇.c[inx_c,iny_c])/sc.t
        probes.t[it]    = it*Δ.t*sc.t

    end

    #--------------------------------------------#

    save("./examples/_TwoPhases/TwoPhasesPlasticity/VE_loading_homogeneous.jld2", "probes", probes)

    return 
end

function Run()

    nc = (x=100, y=50)

    # Mode 0   
    main(nc);
    
end

Run()
