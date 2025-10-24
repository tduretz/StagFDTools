using StagFDTools, StagFDTools.ThermoMechanics, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, JLD2
import Statistics:mean
using Enzyme, GLMakie, GridGeometryUtils 

hours = 3600

# This example shows how thermal loading (heating) leads to pressurisation
# The pressure is predicted numerically and exactly using the adiabatic relation:
# ΔP = α/K*ΔT 

@views function main(nc)

    sc = (L=1e0, t=1e-2, σ=1e9, T=1)
    m  = sc.σ * sc.L * sc.t^2.0
    J  = m * sc.L^2.0 / sc.t^2.0
    W  = J/sc.t

    ηi           = 1e18 / (sc.σ*sc.t)
    ηinc         = 1e18 / (sc.σ*sc.t)
    ηrim         = 1e8 / (sc.σ*sc.t)
    Gi           = 535e9/ sc.σ  
    Ginc         = 80e9/ sc.σ
    Grim         = 80e26/ sc.σ
    Ki           = 444e9 / sc.σ 
    Kinc         = 126e9 / sc.σ
    Krim         = 2.2e9 / sc.σ
    αi           = 1e-6 / (1/sc.T)
    αinc         = 3.2e-5 / (1/sc.T)
    αrim         = 2.6e-4 / (1/sc.T)
    ki           = 2e3    / (W/sc.L/sc.T)
    kinc         = 4.0    / (W/sc.L/sc.T)
    krim         = 0.6    / (W/sc.L/sc.T)
    ρi           = 3500.0 / (m/sc.L^3)
    ρinc         = 3300.0 / (m/sc.L^3)
    ρrim         = 997.0 / (m/sc.L^3)
    cpi          = 509.0 / (J/m/sc.T)
    cpinc        = 800.0 / (J/m/sc.T)
    cprim        = 4184.0 / (J/m/sc.T)
    Pinc         = 1e9 / sc.σ
    Prim         = 0e9 / sc.σ

    nt           = 10
    niter        = 5
    ϵ_nl         = 1e-10
    Δt0          = ηi/Gi/4.0/100
    ε̇            = 0*1e-6   / (1/sc.t)
    L            = 2e-3     / sc.L
    T_ini        = 300.0  / sc.T
    T_fin        = 1100.0 / sc.T
    dTdt         = (T_fin - T_ini) / (nt*Δt0)
    P_ini        = 1e6    / sc.σ
    t            = 0.0
    r            = 0.3/1000    / sc.L

    # Material geometries
    shape  = :hexagon
    rimmed = true
    r2     = 1.2*r 

    if shape === :circle
        inclusion = Ellipse((0.0, 0.0), r, r; θ = 1 * π / 4)
        rim       = Ellipse((0.0, 0.0), r2, r2; θ = 1 * π / 4)
    elseif shape === :ellipse
        inclusion = Ellipse((0.0, 0.0), r/3, 2r; θ = 1 * π / 4)
    elseif shape === :rectangle
        inclusion = Rectangle((0.0, -0.0), r*sqrt(π), r*sqrt(π); θ = -0*π / 4)
    elseif shape === :hexagon
        inclusion = Hexagon((0.0, -0.0), r; θ = -1*π / 4)
        rim       = Hexagon((0.0, -0.0), r2; θ = -1*π / 4)
    end

    # Velocity gradient matrix
    D_BC = @SMatrix( [ε̇ 0; 0 -ε̇] )

    # Material parameters
    materials = ( 
        oneway       = false,
        compressible = true,
        Dzz          = 0.0,
        n            = [1.0 1.0   1.0  ],
        ηs0          = [ηi  ηinc  ηrim ], 
        G            = [Gi  Ginc  Grim ], 
        K            = [Ki  Kinc  Krim ],
        α            = [αi  αinc  αrim ],
        k            = [ki  kinc  krim ],
        cp           = [cpi cpinc cprim],
        ρr           = [ρi  ρinc  ρrim ],
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
    type.Vx[end-0,iny_Vx]   .= :Neumann_normal
    type.Vx[1,iny_Vx]       .= :Neumann_normal 
    type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
    type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy]  .= :in       
    type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
    type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
    type.Vy[inx_Vy,1]       .= :Neumann_normal 
    type.Vy[inx_Vy,end-0]   .= :Neumann_normal 
    #-------- Vx -------- #
    # type.Vx[inx_Vx,iny_Vx]  .= :in       
    # type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
    # type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
    # type.Vx[inx_Vx,2]       .= :Dirichlet_tangent
    # type.Vx[inx_Vx,end-1]   .= :Dirichlet_tangent
    # # -------- Vy -------- #
    # type.Vy[inx_Vy,iny_Vy]  .= :in       
    # type.Vy[2,iny_Vy]       .= :Dirichlet_tangent
    # type.Vy[end-1,iny_Vy]   .= :Dirichlet_tangent
    # type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    # type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
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

    # printxy(type.Vx)
    # printxy(number.Vx)
    # error()

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
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), zz = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
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

    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([xc[i-1], yc[j-1]])
        if rimmed && inside(𝐱, rim)
            phases.c[i, j] = 3
            P.t[i, j] = Prim
        end
        if inside(𝐱, inclusion)
            phases.c[i, j] = 2
            P.t[i, j] = Pinc
        end
    end
       for i in inx_v, j in iny_v  # loop on vertices
        𝐱 = @SVector([xv[i-1], yv[j-1]])
        if rimmed && inside(𝐱, rim)
            phases.v[i, j] = 3
        end
        if inside(𝐱, inclusion)
            phases.v[i, j] = 2
        end
    end

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...), Pt = zeros(size_c...), T = zeros(size_c...))
    @views begin
        BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
        BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
        BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
        BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
        BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
        BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)
    end
    #--------------------------------------------#

    err    = (x = zeros(niter), y = zeros(niter), Pt = zeros(niter), T = zeros(niter))

    probes = (
            T   = zeros(nt),
            Pt  = zeros(nt),
            t   = zeros(nt),
            τII = zeros(nt),
    )
    
    for it=1:nt

        @printf("Step %04d\n", it)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.Pt, 0e0)
        fill!(err.T, 0e0)

        # Swap old values 
        T0.c  .= T.c
        P0.t  .= P.t
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy

        # Update time
        t += Δ.t

        # Ramp up boundary t
        BC.T .= T_ini .+ dTdt*t

        @show BC.T[2,2]*sc.T
        nRT0 = 1.0

        # Time integration loop
        for iter=1:niter

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

            #--------------------------------------------#
            UpdateSolution!(V, T, P, dx, number, type, nc)

            #--------------------------------------------#
            # Residual check
            TangentOperator!( 𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, T, P, ΔP, type, BC, materials, phases, Δ)
            ResidualMomentum2D_x!(R, V, T, T0, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualMomentum2D_y!(R, V, T, T0, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            ResidualContinuity2D!(R, V, T, T0, P, P0, phases, materials, number, type, BC, nc, Δ) 
            ResidualHeatDiffusion2D!(R, V, T, T0, P, P0, phases, materials, number, type, BC, nc, Δ) 

            @info "Iteration $(iter)"
            if iter==1
                nRT0 = norm(R.T[inx_c,iny_c])
            end
            @printf("f_x = %1.2e\n", norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx))
            @printf("f_y = %1.2e\n", norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy))
            @printf("f_P = %1.2e\n", norm(R.pt[inx_c,iny_c]) /sqrt(nPt))
            @printf("f_T = %1.2e %1.2e\n", norm(R.T[inx_c,iny_c])  /sqrt(nT ), norm(R.T[inx_c,iny_c])/nRT0)
            err.x[iter]  = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter]  = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.Pt[iter] = @views norm(R.pt[inx_c,iny_c])/sqrt(nPt)
            err.T[iter]  = @views norm(R.T[inx_c,iny_c])/sqrt(nPt)
            max(err.x[iter], err.y[iter], err.Pt[iter], err.T[iter]) < ϵ_nl ? break : nothing

        end
        
        #--------------------------------------------#

        # Post process stress and strain rate
        τxyc = av2D(τ.xy)

        probes.T[it]   = mean(T.c[phases.c .== 2])
        probes.Pt[it]  = maximum(P.t[phases.c .== 2])
        probes.t[it]   = t
        probes.τII[it] = maximum(τ.II[phases.c .== 1])
        @show mean(T.c[inx_c,iny_c])*sc.T

        # Post process 
        Vxsc = 0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1])
        Vysc = 0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end])
        Vs   = sqrt.( Vxsc.^2 .+ Vysc.^2)

        # Visualise
        function figure()
            ftsz = 25

            fig = Figure()
            empty!(fig)
            ax = Axis(fig[1,1], aspect=DataAspect(), title=L"$$Pressure", xlabel="x", ylabel="y")
            # hm = heatmap!(ax, xc, yc,  (R.T[inx_c,iny_c]), colormap=:bluesreds)
            # heatmap!(ax, xc, yc,  (phases.c[inx_c,iny_c]), colormap=:bluesreds)
            # hm =heatmap!(ax, xc, yc,  (T.c[inx_c,iny_c]*sc.σ/1e9), colormap=:bluesreds)
            hm =heatmap!(ax, xc, yc,  (P.t[inx_c,iny_c]*sc.σ/1e9), colormap=:bluesreds)
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:white)

            Colorbar(fig[2, 1], hm, label = L"$P$ (GPa)", height=10, width = 200, labelsize = 15, ticklabelsize = 15, vertical=false, valign=true, flipaxis = true )
            
            ax = Axis(fig[1,2], aspect=DataAspect(), title=L"$$Deviatoric stress", xlabel="x", ylabel="y")
            hm = heatmap!(ax, xc, yc,  (τ.II[inx_c,iny_c]*sc.σ/1e9), colormap=:bluesreds)
            contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:white)
            Colorbar(fig[2, 2], hm, label = L"$τ$ (GPa)", height=10, width = 200, labelsize = 15, ticklabelsize = 15, vertical=false, valign=true, flipaxis = true )
            
            st = 10
            # arrows!(ax, xc[1:st:end], yc[1:st:end], σ1.x[inx_c,iny_c][1:st:end,1:st:end], σ1.y[inx_c,iny_c][1:st:end,1:st:end], arrowsize = 0, lengthscale=0.04, linewidth=2, color=:white)
            # ax  = Axis(fig[3,2], xlabel="Time (h)", ylabel="τ dia. (GPa)")
            # scatter!(ax, probes.t[1:nt]./hours, probes.τII[1:nt]*sc.σ./1e9 ) 
            ax  = Axis(fig[3,2], xlabel=L"$T$ (\degree~C)", ylabel=L"$\tau$ dia. (GPa)")
            scatter!(ax, probes.T[1:it]*sc.T .-273.15, probes.τII[1:it]*sc.σ./1e9 ) 
            ax  = Axis(fig[3,1], xlabel=L"$T$  (\degree~C)", ylabel=L"$P$ ol. (GPa)")
            scatter!(ax, probes.T[1:it]*sc.T  .-273.15, probes.Pt[1:it]*sc.σ./1e9 )
            # ax  = Axis(fig[3,3], xlabel="Time (h)", ylabel="Temperature (K)")
            # scatter!(ax, probes.t[1:nt]./hours, probes.T[1:nt]*sc.T )
            # ax  = Axis(fig[2,2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error")
            # scatter!(ax, 1:niter, log10.(err.x[1:niter]) )
            # scatter!(ax, 1:niter, log10.(err.y[1:niter]) )
            # scatter!(ax, 1:niter, log10.(err.Pt[1:niter]) )
            # scatter!(ax, 1:niter, log10.(err.T[1:niter]) )
            
            if rimmed
                save("./results/DiOl_rimmed_$(shape).png", fig, px_per_unit = 4) 
            else
                save("./results/DiOl_$(shape).png", fig, px_per_unit = 4) 
            end
            display(fig)
        end
        with_theme(figure, theme_latexfonts())
      
    end

    #--------------------------------------------#

    return nothing
end

function Run()

    nc = (x=150, y=150)

    main(nc)
    
end

Run()
