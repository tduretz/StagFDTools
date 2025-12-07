using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GridGeometryUtils
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use
using TimerOutputs, CairoMakie

function line(p, K, Δt, η_ve, ψ, p1, t1)
    p2 = p1 + K*Δt*sind(ψ)
    t2 = t1 - η_ve  
    a  = (t2-t1)/(p2-p1)
    b  = t2 - a*p2
    return a*p + b
end

@views function main(nc, θgouge)
    #--------------------------------------------#

    # Scaling
    sc  = (σ=1e9, L=1, t=1e6)

    # Parameters
    width     = 1.0/sc.L
    height    = 1.0/sc.L
    thickness = 0.2/sc.L
    θgouge    = (90-θgouge) /180*π
    ε̇xx       = 1e-6*sc.t
    Pbg       = 5e7/sc.σ

    # Boundary loading type
    # config = :EW_Neumann
    config = :free_slip

    # # mode 1
    # nt     = 85*2
    # Δt0    = 5e0/sc.t
    # D_BC   = @SMatrix( [  ε̇xx  0.;
    #                       0  -ε̇xx*0 ])

    # mode 2
    nt     = 85
    Δt0       = 5e1/sc.t
    D_BC   = @SMatrix( [  ε̇xx  0.;
                          0  -ε̇xx ])

    # Material parameters
    materials = ( 
        compressible = true,
        # plasticity   = :tensile,
        # plasticity   = :DruckerPrager1,
        plasticity   = :GolchinMCC,
        # plasticity   = :Hyperbolic,
        # plasticity   = :DruckerPrager,
        # plasticity   = :Kiss2023,
        #      rock   gouge  salt 
        n    = [1.0    1.0    1.0 ],      # Power law exponent
        η0   = [1e48   1e28   1e13]./sc.σ./sc.t,      # Reference viscosity 
        G    = [1e10   1e9    1e60]./sc.σ,      # Shear modulus
        C    = [10e6   10e6   15e60]./sc.σ,      # Cohesion
        ϕ    = [35.    35.    35. ],      # Friction angle
        ψ    = [20.0   20.0   20.0],      # Dilation angle
        ηvp  = [5e12   5e12   5e12].*1e-4/sc.σ./sc.t, # Viscoplastic regularisation
        β    = [1e-11  1e-10 1e-12].*sc.σ,      # Compressibility
        B    = [0.0    0.0    0.0 ],      # (calculated after) power-law creep pre-factor
        cosϕ = [0.0    0.0    0.0 ],      # (calculated after) frictional parameters
        sinϕ = [0.0    0.0    0.0 ],      # (calculated after) frictional parameters
        cosψ = [0.0    0.0    0.0 ],      # (calculated after) frictional parameters
        sinψ = [0.0    0.0    0.0 ],      # (calculated after) frictional parameters
        M    = [0.0    0.0    0.0 ],
        N    = [0.0    0.0    0.0 ],
        Pc   = [6e7    6e7    6e7 ]./sc.σ,  
        a    = [0.5    0.5    0.5 ],
        b    = [0.0    0.0    0.0 ],
        c    = [0.5    0.5    0.5 ],
        σT   = [5e6   5.0e6  5.0e6]./sc.σ, # Kiss2023 / Tensile / Hyperbolic
        δσT  = [1e6   1.0e6  1e6  ]./sc.σ, # Kiss2023
        P1   = [0.0   0.0    0.0  ], # Kiss2023
        τ1   = [0.0   0.0    0.0  ], # Kiss2023
        P2   = [0.0   0.0    0.0  ], # Kiss2023
        τ2   = [0.0   0.0    0.0  ], # Kiss2023
    )
    # For power law
    materials.B   .= (2*materials.η0).^(-materials.n)

    # For Kiss2023: calculate corner coordinates 
    @. materials.P1 = -(materials.σT - materials.δσT)                                         # p at the intersection of cutoff and Mode-1
    @. materials.τ1 = materials.δσT                                                           # τII at the intersection of cutoff and Mode-1
    @. materials.P2 = -(materials.σT - materials.C*cosd(materials.ϕ))/(1.0-sind(materials.ϕ)) # p at the intersection of Drucker-Prager and Mode-1
    @. materials.τ2 = materials.P2 + materials.σT   

    # For plasticity
    @. materials.cosϕ  = cosd(materials.ϕ)
    @. materials.cosψ  = cosd(materials.ψ)
    @. materials.sinϕ  = sind(materials.ϕ)
    @. materials.sinψ  = sind(materials.ψ)
    @. materials.M     = 6*sind(materials.ϕ) / (3 - sind(materials.ϕ))
    @. materials.N     = 6*sind(materials.ψ) / (3 - sind(materials.ψ))
    
    # Geometry
    L     = (x=width/sc.L, y=height/sc.L)

    # Newton solver
    niter = 15
    ϵ_nl  = 1e-9
    α     = LinRange(0.05, 1.0, 10)

    # Grid bounds
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    set_boundaries_template!(type, config, nc)

    #--------------------------------------------#
    # Equation numbering
    number = Fields(
        fill(0, size_x),
        fill(0, size_y),
        fill(0, size_c),
    )
    Numbering!(number, type, nc)

    #--------------------------------------------#
    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([1 1 1; 1 1 1])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([1 1; 1 1; 1 1])), 
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊  = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐  = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏  = ExtendableSparseMatrix(nPt, nPt)
    dx = zeros(nVx + nVy + nPt)
    r  = zeros(nVx + nVy + nPt)

    #--------------------------------------------#
    # Discretisation
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), II = zeros(size_c...) )
    Pt      = zeros(size_c...)
    Pti     = zeros(size_c...)
    Pt0     = zeros(size_c...)
    ΔPt     = (c=zeros(size_c...), Vx = zeros(size_x...), Vy = zeros(size_y...))

    Dc      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c = Dc, v = Dv)
    D_ctl_c =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v =  [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v)

    # Mesh coordinates
    xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    # Initial velocity & pressure field
    @views V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    @views V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
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

    # Set material geometry 
    phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2

    Pt  .= Pbg#*rand(size(Pt)...)
    Pt0 .= Pt
    Pti .= Pt

    #--------------------------------------------#

    rvec   = zeros(length(α))
    err    = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    probes = (τII = zeros(nt), fric = zeros(nt), t = zeros(nt), εxx=zeros(nt), εyy=zeros(nt), σyyN=zeros(nt), σyyS=zeros(nt), σxxW=zeros(nt), σxxE=zeros(nt))
    to     = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        @printf("Step %04d\n", it)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.p, 0e0)
        
        # Swap old values 
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0   .= Pt

        # Time integration
        for iter=1:niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)
                @show extrema(λ̇.c[inx_c,iny_c])
                @show extrema(λ̇.v[inx_v,iny_v])
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = @views norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = @views norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = @views norm(R.p[inx_c,iny_c])/sqrt(nPt)
            max(err.x[iter], err.y[iter]) < ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            end

            #--------------------------------------------# 
            # Stokes operator as block matrices
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= M.Pt.Pt
            
            #--------------------------------------------#
     
            # Direct-iterative solver
            fu   = @views -r[1:size(𝐊,1)]
            fp   = @views -r[size(𝐊,1)+1:end]
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu, ηb=1e3, niter_l=10, ϵ_l=1e-11)
            @views dx[1:size(𝐊,1)]     .= u
            @views dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)

        end

        # Update pressure
        Pt .+= ΔPt.c

        #--------------------------------------------#

        # Post process stress and strain rate
        τII_rock  = τ.II[inx_c,iny_c][phases.c[inx_c,iny_c].==1]
        P_rock    =   Pt[inx_c,iny_c][phases.c[inx_c,iny_c].==1]
        λ̇_rock    =  λ̇.c[inx_c,iny_c][phases.c[inx_c,iny_c].==1]

        # τII_gouge = τ.II[inx_c,iny_c][phases.c[inx_c,iny_c].==2]
        # P_gouge   =  Pt[inx_c,iny_c][phases.c[inx_c,iny_c].==2]

        # Principal stress
        σ1 = (x = zeros(size(Pt)), y = zeros(size(Pt)), v = zeros(size(Pt)))
        τxyc = 0.25*(τ.xy[1:end-1,1:end-1] .+ τ.xy[2:end-0,1:end-1] .+ τ.xy[1:end-1,2:end-0] .+ τ.xy[2:end-0,2:end-0])

        for i in inx_c, j in iny_c
            σ  = @SMatrix[-Pt[i,j]+τ.xx[i,j] τxyc[i,j] 0.; τxyc[i,j] -Pt[i,j]+τ.yy[i,j] 0.; 0. 0. -Pt[i,j]+(-τ.xx[i,j]-τ.yy[i,j])]
            v  = eigvecs(σ)
            σp = eigvals(σ)
            σ1
            scale = sqrt(v[1,1]^2 + v[2,1]^2)
            σ1.x[i,j] = v[1,1]/scale
            σ1.y[i,j] = v[2,1]/scale
            σ1.v[i] = σp[1]
        end

        # Store probes data
        probes.t[it]    = it*Δ.t
        probes.τII[it]  = mean(τ.II[inx_c, iny_c])
        probes.σxxW[it] = τ.xx[2,     Int64(floor(nc.y/2))] - Pt[2,     Int64(floor(nc.y/2))] 
        probes.σxxE[it] = τ.xx[end-1, Int64(floor(nc.y/2))] - Pt[end-1, Int64(floor(nc.y/2))] 
        probes.σyyS[it] = τ.yy[Int64(floor(nc.x/2)),     2] - Pt[Int64(floor(nc.x/2)),     2] 
        probes.σyyN[it] = τ.yy[Int64(floor(nc.x/2)), end-1] - Pt[Int64(floor(nc.x/2)), end-1] 

        i_midx = Int64(floor(nc.x))
        probes.fric[it] = mean(.-τxyc[i_midx, end-3]./(-Pt[i_midx, end-3] .+ τ.yy[i_midx, end-3])) 

        @show minimum(Pt)*sc.σ,  maximum(Pt)*sc.σ

        # Visualise
        function figure()
            ftsz = 25
            fig = Figure(size=(1000, 1000)) 
            empty!(fig)
            ax  = Axis(fig[1:2,1], aspect=DataAspect(), title="Plastic Strain rate", xlabel="x", ylabel="y", xlabelsize=ftsz,  ylabelsize=ftsz, titlesize=ftsz)
            eps   = 1e-1
            # field = Pt[inx_c,iny_c] .* sc.σ
            field = log10.((λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            hm = heatmap!(ax, xc.*sc.L, yc.*sc.L, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            contour!(ax, xc.*sc.L, yc.*sc.L,  phases.c[inx_c,iny_c], color=:white)
            Colorbar(fig[3, 1], hm, label = L"$\dot\lambda$", height=30, width = 300, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            Vxc = (0.5*(V.x[1:end-1,2:end-1] + V.x[2:end,2:end-1]))[2:end-1,2:end-1].*sc.L/sc.t
            Vyc = (0.5*(V.y[2:end-1,1:end-1] + V.y[2:end-1,2:end]))[2:end-1,2:end-1].*sc.L/sc.t
            step = 20
            arrows2d!(ax, xc[1:step:end].*sc.L, yc[1:step:end].*sc.L, Vxc[1:step:end,1:step:end], Vyc[1:step:end,1:step:end], lengthscale=500000.4, color=:white)
            # arrows2d!(ax, xc[1:st:end], yc[1:st:end], σ1.x[inx_c,iny_c][1:st:end,1:st:end], σ1.y[inx_c,iny_c][1:st:end,1:st:end], arrowsize = 0, lengthscale=0.04, linewidth=2, color=:white)
            xlims!(ax, minimum(xv).*sc.L, maximum(xv).*sc.L)
            # ax  = Axis(fig[1,2], xlabel="Displacement", ylabel="Axial stress [MPa]", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            # scatter!(ax, probes.t[1:nt]/sc.t, probes.τII[1:nt]*sc.σ./1e6 )
            # scatter!(ax, probes.t[1:nt]*ε̇xx*L.y*sc.L, probes.σxxW[1:nt]*sc.σ./1e6 )
            # scatter!(ax, probes.t[1:nt]*ε̇xx*L.y*sc.L, probes.σxxE[1:nt]*sc.σ./1e6, marker=:star5, markersize=20 )
            # scatter!(ax, probes.t[1:nt]*ε̇xx*L.y*sc.L, probes.σyyN[1:nt]*sc.σ./1e6 )
            # scatter!(ax, probes.t[1:nt]*ε̇xx*L.y*sc.L, probes.σyyS[1:nt]*sc.σ./1e6 )
            ax  = Axis(fig[1,2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            scatter!(ax, 1:niter, log10.(err.x[1:niter]./err.x[1]) )
            scatter!(ax, 1:niter, log10.(err.y[1:niter]./err.y[1]) )
            scatter!(ax, 1:niter, log10.(err.p[1:niter]./err.p[1]) )
            ylims!(ax, -15, 1)
            ax  = Axis(fig[2,2], title=L"$$Stress space", xlabel=L"$P$", ylabel=L"$\tau_{II}$", xlabelsize=ftsz, ylabelsize=ftsz, titlesize=ftsz)
            P_ax       = LinRange(-10/1e3, 100/1e3, 100)
            # τ_ax_rock = materials.C[1]*materials.cosϕ[1] .+ P_ax.*materials.sinϕ[1]
            # lines!(ax, P_ax*sc.σ/1e6, τ_ax_rock*sc.σ/1e6, color=:black)
            
            # Plot yield
            P_ax       = LinRange(-materials.σT[1]+1e-4, 80/1e3, 100)
            τ_ax       = LinRange( 0, 60/1e3, 100)
            f          = zeros(length(P_ax), length(τ_ax))
            q          = zeros(length(P_ax), length(τ_ax))
            for i in eachindex(P_ax), j in eachindex(τ_ax)
                m = materials
                if m.plasticity == :DruckerPrager1 
                    yieldf = DruckerPrager1()
                    p = (m.C[1], m.cosϕ[1], m.sinϕ[1], m.cosψ[1], m.sinψ[1], 0*m.ηvp[1])
                elseif m.plasticity == :GolchinMCC     
                    yieldf = GolchinMCC()
                    p = (m.M[1], m.N[1], -m.σT[1], m.Pc[1], m.a[1], m.b[1], m.c[1], 0*m.ηvp[1])
                elseif m.plasticity == :Hyperbolic    
                    yieldf = Hyperbolic()
                    p = (m.C[1], m.cosϕ[1], m.sinϕ[1], m.cosψ[1], m.sinψ[1], m.σT[1], 0*m.ηvp[1])
                end
                f[i,j] = Yield(@SVector([τ_ax[j], P_ax[i], 0.0]), p, yieldf)
                q[i,j] = Potential(@SVector([τ_ax[j], P_ax[i], 0.0]), p, yieldf)
            end
            contour!(ax, P_ax*sc.σ/1e6, τ_ax*sc.σ/1e6, f*sc.σ./1e6, levels=[0., 0.0], color=:red)
            contour!(ax, P_ax*sc.σ/1e6, τ_ax*sc.σ/1e6, q*sc.σ./1e6, levels=[0., 0.0], color=:red, linestyle=:dash)

            cosΨ, sinΨ, C, σT = materials.cosϕ[1], materials.sinϕ[1], materials.sinϕ[1], materials.σT[1]
            B = C * cosΨ - σT*sinΨ
            dQdtau = @. τII_rock /sqrt(τII_rock^2 + B^2) 
            scatter!(ax, (P_rock .+ 0*sinΨ .* λ̇_rock.*materials.ηvp[1])*sc.σ/1e6, (τII_rock .+ dQdtau.*λ̇_rock.*materials.ηvp[1])*sc.σ/1e6, color=:black )
                    
            # τ_ax_gouge = materials.C[2]*materials.cosϕ[2] .+ P_ax.*materials.sinϕ[2]
            # lines!(ax, P_ax*sc.σ/1e6, τ_ax_gouge*sc.σ/1e6, color=:red)
            # scatter!(ax, P_gouge*sc.σ/1e6, τII_gouge*sc.σ/1e6, color=:red )
            display(fig)
        end
        with_theme(figure, theme_latexfonts())
    end

    display(to)
    
end

let
    main((x = 100, y = 100), 60)
end