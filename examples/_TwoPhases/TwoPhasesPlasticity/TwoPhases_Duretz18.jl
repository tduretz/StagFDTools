using StagFDTools, StagFDTools.TwoPhases, ExtendableSparse, StaticArrays, GLMakie, LinearAlgebra, SparseArrays, Printf, JLD2, MAT
import Statistics:mean
using Enzyme  # AD backends you want to use

@views function main(nc)

    sc = (σ = 3e10, L = 1e3, t = 1e10)

    # Load data
    filepath = joinpath(@__DIR__, "DataM2Di_EP_test01.mat")
    data = matread(filepath)
    @show keys(data)

    homo   = false

    # Time steps
    nt     = 30
    Δt0    = 1e10/sc.t 

    # Newton solver
    niter = 20
    ϵ_nl  = 1e-10
    α     = LinRange(0.05, 1.0, 10)

    rad     = 1e2/sc.L 
    Pt_ini  = 0*1e8/sc.σ
    Pf_ini  = 0*1e6/sc.σ
    ε̇bg     = -5e-15.*sc.t
    τ_ini   = 0*(sind(35)*(Pt_ini-Pf_ini) + 0*1e7/sc.σ*cosd(35))  

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇bg 0; 0 -ε̇bg] )

    τxx_ini = τ_ini*D_BC[1,1]/abs(ε̇bg)
    τyy_ini = τ_ini*D_BC[2,2]/abs(ε̇bg)

    # Material parameters
    materials = ( 
        oneway       = false,
        compressible = true,
        plasticity   = :off,
        linearizeϕ   = false,        
        single_phase = false,
        n     = [1.0    1.0  ],
        ηs0   = [1e20   1e20 ]/sc.σ/sc.t .* 1e6,  # achtung turn of viscous shear
        ηΦ    = [2e22   2e22 ]/sc.σ/sc.t .* 1e6,  # achtung turn of viscous volumetric
        G     = [1e10   0.25e10]./sc.σ, 
        Kd    = [1e30   1e30 ]./sc.σ,  # not needed
        Ks    = [2e10   2e10 ]./sc.σ,
        KΦ    = [5e9    5e9  ]./sc.σ,
        Kf    = [2e9    2e9 ]./sc.σ, 
        k_ηf0 = [1e-15  1e-15]./(sc.L^2/sc.σ/sc.t),
        ϕ     = [30.    30.  ].*1,
        ψ     = [10.    10.  ].*1,
        C     = [3e7    3e7  ]./sc.σ,
        ηvp   = [0.0    0.0  ]./sc.σ/sc.t,
        cosϕ  = [0.0    0.0  ],
        sinϕ  = [0.0    0.0  ],
        sinψ  = [0.0    0.0  ],
    )

    # For plasticity
    @. materials.cosϕ  = cosd(materials.ϕ)
    @. materials.sinϕ  = sind(materials.ϕ)
    @. materials.sinψ  = sind(materials.ψ)

    Φ0      = 1e-3
    # Φ0 = (materials.KΦ[1] .* Δt0 .* (Pf_ini - Pt_ini)) ./ (materials.KΦ[1] .* materials.ηΦ[1])
    @show Φ0
    # error()
    Φ_ini   = Φ0

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
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1;  1 1 1]),        @SMatrix([1 1 1;  1 1 1])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([1 1; 1 1; 1 1]),        @SMatrix([1 1; 1 1; 1 1])),
        Fields(@SMatrix([0 1 0;  0 1 0]),                       @SMatrix([0 0; 1 1; 0 0]),                       @SMatrix([1]),                   @SMatrix([1])),
        Fields(@SMatrix([0 1 0;  0 1 0]),                       @SMatrix([0 0; 1 1; 0 0]),                       @SMatrix([1]),                   @SMatrix([1 1 1; 1 1 1; 1 1 1])),
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
    L   = (x=4e3/sc.L, y=2e3/sc.L)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t=Δt0)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...), Φ=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    Vi  = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (c  =  ones(size_c...), v  =  ones(size_v...) )
    Φ   = (c=Φ_ini.*ones(size_c...), v=Φ_ini.*ones(size_v...) )
    Φ0  = (c=Φ_ini.*ones(size_c...), v=Φ_ini.*ones(size_v...) )

    εp  = zeros(size_c...)

    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    τ0      = (xx = τxx_ini.*ones(size_c...), yy = τyy_ini.*ones(size_c...), xy = zeros(size_v...) )
    τ       = (xx = τxx_ini.*ones(size_c...), yy = τyy_ini.*ones(size_c...), xy = zeros(size_v...), II = zeros(size_c...), f = zeros(size_c...),)
    Dc      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(5,5)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), x =ones(Int64, size_x...), y=ones(Int64, size_y...) )  # phase on velocity points
    P       = (t = Pt_ini.*ones(size_c...), f = Pf_ini.*ones(size_c...))
    Pi      = (t = Pt_ini.*ones(size_c...), f = Pf_ini.*ones(size_c...))
    P0      = (t = zeros(size_c...), f = zeros(size_c...))
    ΔP      = (t = zeros(size_c...), f = zeros(size_c...))

    # Generate grid coordinates 
    x = (min=-L.x/2, max=L.x/2)
    y = (min=-L.y/2, max=L.y/2)
    X = GenerateGrid(x, y, Δ, nc)

    # Initial configuration
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*X.v.x .+ D_BC[1,2]*X.c.y' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*X.c.x .+ D_BC[2,2]*X.v.y'

    if !homo
        # for I in CartesianIndices(Φ.c)
        #     i, j = I[1], I[2]
        #     if i>1 && i<size(Φ.c,1) && j>1 && j<size(Φ.c,2)
        #         if (X.c.x[i-1]^2 + X.c.y[j-1]^2) < rad^2
        #             Φ.c[i,j] = 1.1*Φ_ini
        #         end
        #     end 
        # end

        # Set material geometry 
        @views phases.c[inx_c, iny_c][(X.c.x.^2 .+ (X.c.y').^2) .<= rad^2] .= 2
        @views phases.v[inx_v, iny_v][(X.v.x.^2 .+ (X.v.y').^2) .<= rad^2] .= 2
    end

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

    rvec   = zeros(length(α))
    probes = (
        Pe  = zeros(nt),
        Pt  = zeros(nt),
        Pf  = zeros(nt),
        τ   = zeros(nt),
        Φ   = zeros(nt),
        λ̇   = zeros(nt),
        t   = zeros(nt),
        str = zeros(nt),
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

            @printf("     Step %04d --- Iteration %04d\n", it, iter)

            λ̇.c   .= 0.0
            λ̇.v   .= 0.0

            #--------------------------------------------#
            # Residual check
            TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)
            ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualContinuity2D!(R, V, P, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 
            ResidualFluidContinuity2D!(R, V, P, ΔP, P0, Φ0, phases, materials, number, type, BC, nc, Δ) 

            println("min/max λ̇.c  - ",  extrema(λ̇.c[inx_c,iny_c]))
            println("min/max λ̇.v  - ",  extrema(λ̇.v[3:end-2,3:end-2]))
            println("min/max ΔP.t - ",  extrema(ΔP.t[inx_c,iny_c]))
            println("min/max ΔP.f - ",  extrema(ΔP.f[inx_c,iny_c]))

            @info "Residuals"
            @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            @show norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            @show norm(R.pf[inx_c,iny_c])/sqrt(nPf)

            err.x[iter]  = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter]  = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.pt[iter] = @views norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            err.pf[iter] = @views norm(R.pf[inx_c,iny_c])/sqrt(nPt)
            if max(err.x[iter], err.y[iter], err.pt[iter], err.pf[iter]) < ϵ_nl 
                println("Converged")
                break 
            end

            # Set global residual vector
            r = zeros(nVx + nVy + nPt + nPf)
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @info "Assembly, ndof  = $(nVx + nVy + nPt + nPf)"
            AssembleMomentum2D_x!(M, V, P, P0, ΔP, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleMomentum2D_y!(M, V, P, P0, ΔP, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleContinuity2D!(M, V, P, P0, Φ0, phases, materials, number, pattern, type, BC, nc, Δ)
            AssembleFluidContinuity2D!(M, V, P, ΔP, P0, Φ0, phases, materials, number, pattern, type, BC, nc, Δ)

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
            imin = LineSearch!(rvec, α, dx, R, V, P, ε̇, τ, Vi, Pi, ΔP, P0, Φ, Φ0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            UpdateSolution!(V, P, α[imin]*dx, number, type, nc)
        end

        TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, P0, Φ, Φ0, type, BC, materials, phases, Δ)

        #--------------------------------------------#

        # if norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx) > ϵ_nl || norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy) > ϵ_nl
        #     error("Global convergence failed !")
        # end 

        #--------------------------------------------#

        # Include plasticity corrections
        P.t .+= ΔP.t
        P.f .+= ΔP.f
        εp  .+= ε̇.II*Δ.t
        
        τxyc = av2D(τ.xy)
        ε̇xyc = av2D(ε̇.xy)

        # # Post process 
        # @time for i in eachindex(Φ.c)
        #     KΦ     = materials.KΦ[phases.c[i]]
        #     ηΦ     = materials.ηΦ[phases.c[i]] 
        #     sinψ   = materials.sinψ[phases.c[i]] 
        #     dPtdt  = (P.t[i] - P0.t[i]) / Δ.t
        #     dPfdt  = (P.f[i] - P0.f[i]) / Δ.t
        #     dΦdt   = 1/KΦ * (dPfdt - dPtdt) + 1/ηΦ * (P.f[i] - P.t[i]) + λ̇.c[i]*sinψ
        #     Φ.c[i] = Φ0.c[i] + dΦdt*Δ.t
        # end

        Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])[2:end-1,2:end-1]
        Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])[2:end-1,2:end-1]
        Vs   = sqrt.( Vxsc.^2 .+ Vysc.^2)
        Vxf  = -materials.k_ηf0[1]*diff(P.f, dims=1)/Δ.x
        Vyf  = -materials.k_ηf0[1]*diff(P.f, dims=2)/Δ.y
        Vyfc = 0.5*(Vyf[1:end-1,:] .+ Vyf[2:end,:])
        Vxfc = 0.5*(Vxf[:,1:end-1] .+ Vxf[:,2:end])
        Vf   = sqrt.( Vxfc.^2 .+ Vyfc.^2)

        #--------------------------------------------#
        probes.Pe[it]   = mean(P.t[inx_c,iny_c] .- P.f[inx_c,iny_c])*sc.σ
        probes.Pt[it]   = mean(P.t[inx_c,iny_c])*sc.σ
        probes.Pf[it]   = mean(P.f[inx_c,iny_c])*sc.σ
        probes.τ[it]    = mean(τ.II[inx_c,iny_c])*sc.σ
        probes.Φ[it]    = mean(Φ.c[inx_c,iny_c])
        probes.λ̇[it]    = mean(λ̇.c[inx_c,iny_c])/sc.t
        probes.t[it]    = it*Δ.t*sc.t
        probes.str[it]  = abs(ε̇bg)*it*Δ.t

        #-------------------------------------------# 

        @info τ_ini*sc.σ
        @show τxx_ini*sc.σ, τyy_ini*sc.σ
      
        # Visualise
        function figure()
            fig  = Figure(fontsize = 20, size = (900, 600) )    
            step = 10
            ftsz = 15
            eps  = 1e-10

            ax   = Axis(fig[1,1], aspect=DataAspect(), title=L"$$Strain", xlabel=L"x", ylabel=L"y")
            # field = log10.((λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            field = log10.(εp[inx_c,iny_c])
            hm = heatmap!(ax, X.c.x, X.c.y, field, colormap=:jet, colorrange=(-3, -2.3))
            contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 1], hm, label = L"$\lambda$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            # arrows2d!(ax, X.c.x[1:step:end], X.c.y[1:step:end], Vxsc[1:step:end,1:step:end], Vysc[1:step:end,1:step:end], lengthscale=10000.4, color=:white)

            ax    = Axis(fig[3,1], aspect=DataAspect(), title=L"$$Porosity", xlabel=L"x", ylabel=L"y")
            field = Φ.c[inx_c,iny_c]
            hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[4, 1], hm, label = L"$\dot\lambda$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            ax    = Axis(fig[1,2], aspect=DataAspect(), title=L"$P^t$ [MPa]", xlabel=L"x", ylabel=L"y")
            field = (P.t)[inx_c,iny_c].*sc.σ./1e6 
            hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:jet, colorrange=(-6, 4))
            contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            hidexdecorations!(ax)
            Colorbar(fig[2, 2], hm, label = L"$P^t$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            # arrows2d!(ax, X.c.x[1:step:end], X.c.y[1:step:end], Vxsc[1:step:end,1:step:end], Vysc[1:step:end,1:step:end], lengthscale=10000.4, color=:white)

            #######################
            # ax    = Axis(fig[3,2], aspect=DataAspect(), title=L"$P^e - \tau$", xlabel=L"P^e", ylabel=L"\tau")
                 
            # (materials.single_phase) ? α1 = 0.0 : α1 = 1.0 
            # Pe    = (P.t .- α1*P.f)[inx_c,iny_c].*sc.σ

            # τII       = (τ.II)[inx_c,iny_c].*sc.σ
            # P_ax      = LinRange(-5e6, 5e6, 100)
            # τ_ax_rock = materials.C[1]*sc.σ*materials.cosϕ[1] .+ P_ax.*materials.sinϕ[1]
            # lines!(ax, P_ax/1e6, τ_ax_rock/1e6, color=:black)
            # scatter!(ax, Pe[:]/1e6, τII[:]/1e6, color=:black )
            # F_post = @. τ.II - materials.C[1]*materials.cosϕ[1] - (P.t .- α1*P.f)*materials.sinϕ[1]
            # maxF   =  maximum( F_post[inx_c,iny_c] )
            # @info maxF, maxF .*sc.σ /1e6
            # @show maximum(τ.f[inx_c,iny_c]),  maximum(τ.f[inx_c,iny_c]) .*sc.σ /1e6
            #######################

            # # Previous stress states
            # τxyc0 = av2D(τ0.xy)
            # τII0  = sqrt.( 0.5.*(τ0.xx[inx_c,iny_c].^2 + τ0.yy[inx_c,iny_c].^2 + (-τ0.xx[inx_c,iny_c]-τ0.yy[inx_c,iny_c]).^2) .+ τxyc0[inx_c,iny_c].^2 )
            # Pe    = (P0.t .- α1*P0.f)[inx_c,iny_c].*sc.σ
            # τII   = τII0.*sc.σ
            # scatter!(ax, Pe[:]/1e6, τII[:]/1e6, color=:gray )

            # ax    = Axis(fig[1,3], aspect=DataAspect(), title=L"$\tau_\text{II}$ [MPa]", xlabel=L"x", ylabel=L"y")
            # field = (τ.II)[inx_c,iny_c].*sc.σ./1e6
            # hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # hidexdecorations!(ax)
            # Colorbar(fig[2, 3], hm, label = L"$\tau_\text{II}$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            ax  = Axis(fig[3,2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error")
            scatter!(ax, 1:niter, log10.(err.x[1:niter]./err.x[1]) )
            scatter!(ax, 1:niter, log10.(err.y[1:niter]./err.x[1]) )
            scatter!(ax, 1:niter, log10.(err.pt[1:niter]./err.pt[1]) )
            scatter!(ax, 1:niter, log10.(err.pf[1:niter]./err.pf[1]) )
            ylims!(ax, -10, 1.1)

            ax  = Axis(fig[1,3], xlabel="Strain", ylabel="Mean pressure")
            lines!(  ax, data["strvec"][1:end], data["Pvec"][1:end] )
            scatter!(ax, probes.str[1:2:nt], probes.Pt[1:2:nt] )

            ax  = Axis(fig[3,3], xlabel="Strain", ylabel="Mean stress invariant")
            lines!(  ax, data["strvec"][1:end], data["Tiivec"][1:end] )
            scatter!(ax, probes.str[1:2:nt], probes.τ[1:2:nt] )

            # field = P.f.*sc.σ
            # hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # hidexdecorations!(ax)
            # Colorbar(fig[4, 2], hm, label = L"$P^f$", height=20, width = 200, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            
            display(fig) 
        end
        with_theme(figure, theme_latexfonts())

        #-------------------------------------------# 
    end

    #--------------------------------------------#

    save("./examples/_TwoPhases/TwoPhasesPlasticity/VE_loading_homogeneous.jld2", "probes", probes)

    return 
end

function Run()

    nc = (x=100, y=100)

    # Mode 0   
    @time main(nc);
    
end

Run()
