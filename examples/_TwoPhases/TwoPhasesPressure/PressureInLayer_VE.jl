using StagFDTools.TwoPhases, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf, JLD2
import Statistics:mean
using Enzyme  # AD backends you want to use

@views function main(nc, Ωl, Ωη)

    nt  = 100
    Δt0 = 1e9 * 10
    viscoelastic = true

    # nt  = 1
    # Δt0 = 1e9 * 10
    # viscoelastic = false

    # Adimensionnal numbers
    Ωr     = 0.1*1             # Ratio inclusion radius / len
    Ωηi    = 1e-4            # Ratio (inclusion viscosity) / (matrix viscosity)
    Ωp     = 1.              # Ratio (ε̇bg * ηs) / P0
    # Independant
    ηs0    = 1.              # Shear viscosity
    r      = 1.0              # Box size
    P0     = 1.              # Initial ambiant pressure
    ϕ0     = 1e-1
    # Dependant
    ηb0    = Ωη * ηs0        # Bulk viscosity
    k_ηf0  = (r.^2 * Ωl^2) / (ηb0 + 4/3 * ηs0) # Permeability / fluid viscosity
    len    = r/Ωr            # Inclusion radius
    ηs_inc = 1 ./ Ωηi * ηs0  # Inclusion shear viscosity
    ε̇      = Ωp * P0 / ηs0   # Background strain rate

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )

    if viscoelastic
    # Material parameters
    materials = ( 
        oneway       = false,
        compressible = true,
        n     = [1.0  1.0],
        ηs0   = [ηs0  ηs_inc], 
        ηb    = [ηb0  ηb0 ]./(1-ϕ0),
        G     = [1e-7 1e-7], 
        Kd    = [1e-6 1e-6],
        Ks    = [1e-6 1e-6],
        KΦ    = [1e-6 1e-6],
        Kf    = [1e-5 1e-5],
        k_ηf0 = [k_ηf0 k_ηf0],
    )
    else
    materials = ( 
        oneway       = false,
        compressible = true,
        n     = [1.0  1.0],
        ηs0   = [ηs0  ηs_inc], 
        ηb    = [ηb0  ηb0 ]./(1-ϕ0),
        G     = [1e30 1e30], 
        Kd    = [1e30 1e30],
        Ks    = [1e30 1e30],
        KΦ    = [1e30 1e30],
        Kf    = [1e30 1e30],
        k_ηf0 = [k_ηf0 k_ηf0],
    )
    end

    @show materials
    @show materials.ηs0 ./ materials.G
    @show materials.ηb  ./ materials.G
    @show materials.ηs0 ./ materials.Kd
    @show materials.ηb  ./ materials.Kd
    @show materials.ηs0 ./ materials.KΦ
    @show materials.ηb  ./ materials.KΦ
    @show materials.ηs0 ./ materials.Kf
    @show materials.ηb  ./ materials.Kf
    @show r^2/k_ηf0/materials.Kd[1]

    error()
    
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

    P   = (t=zeros(size_c...), f=zeros(size_c...))
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
    phases.c[inx_c, iny_c][abs.(Yc) .< r] .= 2
    phases.v[inx_v, iny_v][abs.(Yv) .< r] .= 2

    # α  = 30.
    # ax = 2.0
    # ay = 1/2
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
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)
    
    #--------------------------------------------#

    probes = (
            Pti = zeros(nt),
            Pfi = zeros(nt),
            Pei = zeros(nt),
            ΔPt = zeros(nt),
            ΔPf = zeros(nt),
            ΔPe = zeros(nt),
            Pe  = zeros(nt),
            Pt  = zeros(nt),
            Pf  = zeros(nt),
            t   = zeros(nt),
    )
    
    for it=1:nt

        P0.t  .= P.t
        P0.f  .= P.f
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy

        # Residual check
        TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, P, ΔP, type, BC, materials, phases, Δ)
        ResidualMomentum2D_x!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualContinuity2D!(R, V, P, P0, ϕ, phases, materials, number, type, BC, nc, Δ) 
        ResidualFluidContinuity2D!(R, V, P, P0, ϕ, phases, materials, number, type, BC, nc, Δ) 

        # Set global residual vector
        r = zeros(nVx + nVy + nPt + nPf)
        SetRHS!(r, R, number, type, nc)

        #--------------------------------------------#
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
        # @time dx = - 𝑀 \ r

        # M2Di solver
        fv    = -r[1:(nVx+nVy)]
        fpt   = -r[(nVx+nVy+1):(nVx+nVy+nPt)]
        fpf   = -r[(nVx+nVy+nPt+1):end]
        dv    = zeros(nVx+nVy)
        dpt   = zeros(nPt)
        dpf   = zeros(nPf)
        rv    = zeros(nVx+nVy)
        rpt   = zeros(nPt)
        rpf   = zeros(nPf)
        rv_t  = zeros(nVx+nVy)
        rpt_t = zeros(nPt)
        s     = zeros(nPf)
        ddv   = zeros(nVx+nVy)
        ddpt  = zeros(nPt)
        ddpf  = zeros(nPf)

        Jvv  = [M.Vx.Vx M.Vx.Vy;
                M.Vy.Vx M.Vy.Vy]
        Jvp  = [M.Vx.Pt;
                M.Vy.Pt]
        Jpv  = [M.Pt.Vx M.Pt.Vy]
        Jpp  = M.Pt.Pt
        Jppf = M.Pt.Pf
        Jpfv = [M.Pf.Vx M.Pf.Vy]
        Jpfp = M.Pf.Pt
        Jpf  = M.Pf.Pf
        Kvv  = Jvv

        @time begin 
            γ = 1e8
            Γ = spdiagm(γ*ones(nPt))
            # Pre-conditionning (~Jacobi)
            Jpv_t  = Jpv  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfv  
            Jpp_t  = Jpp  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfp  #.+ Γ
            Jvv_t  = Kvv  - Jvp *spdiagm(1 ./ diag(Jpp_t))*Jpv 
            @show typeof(SparseMatrixCSC(Jpf))
            Jpf_h  = cholesky(Hermitian(SparseMatrixCSC(Jpf)), check = false  )        # Cholesky factors
            Jvv_th = cholesky(Hermitian(SparseMatrixCSC(Jvv_t)), check = false)        # Cholesky factors
            Jpp_th = spdiagm(1 ./diag(Jpp_t));             # trivial inverse
            @views for itPH=1:50
                rv    .= -( Jvv*dv  + Jvp*dpt             - fv  )
                rpt   .= -( Jpv*dv  + Jpp*dpt  + Jppf*dpf - fpt )
                rpf   .= -( Jpfv*dv + Jpfp*dpt + Jpf*dpf  - fpf )
                s     .= Jpf_h \ rpf
                rpt_t .= -( Jppf*s - rpt)
                s     .=    Jpp_th*rpt_t
                rv_t  .= -( Jvp*s  - rv )
                ddv   .= Jvv_th \ rv_t
                s     .= -( Jpv_t*ddv - rpt_t )
                ddpt  .=    Jpp_th*s
                s     .= -( Jpfp*ddpt + Jpfv*ddv - rpf )
                ddpf  .= Jpf_h \ s
                dv   .+= ddv
                dpt  .+= ddpt
                dpf  .+= ddpf
                @printf("  --- iteration %d --- \n",itPH);
                @printf("  ||res.v ||=%2.2e\n", norm(rv)/ 1)
                @printf("  ||res.pt||=%2.2e\n", norm(rpt)/1)
                @printf("  ||res.pf||=%2.2e\n", norm(rpf)/1)
            #     if ((norm(rv)/length(rv)) < tol_linv) && ((norm(rpt)/length(rpt)) < tol_linpt) && ((norm(rpf)/length(rpf)) < tol_linpf), break; end
            #     if ((norm(rv)/length(rv)) > (norm(rv0)/length(rv0)) && norm(rv)/length(rv) < tol_glob && (norm(rpt)/length(rpt)) > (norm(rpt0)/length(rpt0)) && norm(rpt)/length(rpt) < tol_glob && (norm(rpf)/length(rpf)) > (norm(rpf0)/length(rpf0)) && norm(rpf)/length(rpf) < tol_glob),
            #         if noisy>=1, fprintf(' > Linear residuals do no converge further:\n'); break; end
            #     end
            #     rv0=rv; rpt0=rpt; rpf0=rpf; if (itPH==nPH), nfail=nfail+1; end
            end
        end
        
        dx = zeros(nVx + nVy + nPt + nPf)
        dx[1:(nVx+nVy)] .= dv
        dx[(nVx+nVy+1):(nVx+nVy+nPt)] .= dpt
        dx[(nVx+nVy+nPt+1):end] .= dpf

        #--------------------------------------------#
        UpdateSolution!(V, P, dx, number, type, nc)

        #--------------------------------------------#
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

        #--------------------------------------------#

        Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])
        Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])
        Vs   = sqrt.( Vxsc.^2 .+ Vysc.^2)
        Vxf  = -k_ηf0*diff(P.f, dims=1)/Δ.x
        Vyf  = -k_ηf0*diff(P.f, dims=2)/Δ.y
        Vyfc = 0.5*(Vyf[1:end-1,:] .+ Vyf[2:end,:])
        Vxfc = 0.5*(Vxf[:,1:end-1] .+ Vxf[:,2:end])
        Vf   = sqrt.( Vxfc.^2 .+ Vyfc.^2)

        p1 = heatmap(xc, yc, Vs[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Vs")
        p2 = heatmap(xv[2:end-1], yv[2:end-1], Vf[2:end-1,2:end-1]', aspect_ratio=1, xlim=extrema(xc), title="Vf")
        p3 = heatmap(xc, yc, P.t[inx_c,iny_c]',   aspect_ratio=1, xlim=extrema(xc), title="Pt")
        # divV = diff(V.x[2:end-1,3:end-2], dims=1)/Δ.x  + diff(V.y[3:end-2,2:end-1], dims=2)/Δ.y
        # p3 = heatmap(xc, yc, divV',   aspect_ratio=1, xlim=extrema(xc), title="Pt")
        p4 = heatmap(xc, yc, P.f[inx_c,iny_c]',   aspect_ratio=1, xlim=extrema(xc), title="Pf")
        display(plot(p1, p2, p3, p4))

        # P.t .-= mean(P.t)
        # P.f .-= mean(P.f)

        @show extrema(P.f[phases.c.==1]), maximum(P.f[phases.c.==1]) - minimum(P.f[phases.c.==1])
        @show extrema(P.t[phases.c.==1]), maximum(P.t[phases.c.==1]) - minimum(P.t[phases.c.==1])

        probes.Pti[it]  = mean(P.t[phases.c.==2])
        probes.Pfi[it]  = mean(P.f[phases.c.==2])
        probes.Pei[it]  = mean(P.t[phases.c.==2] .- P.f[phases.c.==2])
        probes.ΔPt[it]  = maximum(P.t[phases.c.==1]) - minimum(P.t[phases.c.==1])
        probes.ΔPf[it]  = maximum(P.f[phases.c.==1]) - minimum(P.f[phases.c.==1])
        probes.ΔPe[it]  = maximum(P.t[phases.c.==1] .- P.f[phases.c.==1]) - minimum(P.t[phases.c.==1] .- P.f[phases.c.==1]) 
        probes.Pe[it]   = norm(P.t .- P.f)
        probes.Pt[it]   = norm(P.t)
        probes.Pf[it]   = norm(P.f)
        probes.t[it]    = it*Δ.t

        @show mean(P.t[phases.c.==2])
        @show mean(P.f[phases.c.==2])

        @show sum(phases.c.==2)
    end

    #--------------------------------------------#

    if viscoelastic
        save("./examples/_TwoPhases/TwoPhasesPressure/Viscoelastic_Layer3.jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "x", (c=xc, v=xv), "y", (c=yc, v=yv), "P", P, "phases", phases)
    else
        save("./examples/_TwoPhases/TwoPhasesPressure/ViscousLimit_Layer3.jld2", "Ωl", Ωl, "Ωη", Ωη, "probes", probes, "x", (c=xc, v=xv), "y", (c=yc, v=yv), "P", P, "phases", phases)
    end
    return P, Δ, (c=xc, v=xv), (c=yc, v=yv)
end

function Run()

    nc = (x=250, y=250)

    # Mode 0   
    Ωl = 10^(-1.7)
    Ωη = 10^(2)
    main(nc,  Ωl, Ωη)
    
end

Run()
