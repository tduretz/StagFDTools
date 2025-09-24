using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use
using TimerOutputs


@views function PorousMediumCircles!(phase, inx, iny, X, Y)
    # Ellipse 1
    x0, y0 = 0., 0.
    α  = 30.0
    ar = 1.0
    r  = 0.007
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 1
    x0, y0 = 0.25, 0.
    α  = -80.0
    ar = 1.0
    r  = 0.005
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 3
    x0, y0 = -0.15, 0.
    α  = -30.0
    ar = 1.0
    r  = 0.005
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 4
    x0, y0 = 0.35, -0.3
    α  = 86.0
    ar = 200.0
    r  = 0.005
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 5
    x0, y0 = 0.35, -0.3
    α  = -20.0
    ar = 1.0
    r  = 0.01
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 5
    x0, y0 = -0.35, -0.3
    α  = 15.0
    ar = 1.0
    r  = 0.004
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

end

@views function PorousMediumEllipses!(phase, inx, iny, X, Y)

    # Ellipse 1
    x0, y0 = 0., 0.
    α  = 30.0
    ar = 100.0
    r  = 0.007
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 1
    x0, y0 = 0.25, 0.
    α  = -80.0
    ar = 150.0
    r  = 0.005
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 3
    x0, y0 = -0.15, 0.
    α  = -30.0
    ar = 100.0
    r  = 0.005
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 4
    x0, y0 = 0.35, -0.3
    α  = 86.0
    ar = 200.0
    r  = 0.005
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 5
    x0, y0 = 0.35, -0.3
    α  = -20.0
    ar = 250.0
    r  = 0.01
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

    # Ellipse 5
    x0, y0 = -0.35, -0.3
    α  = 15.0
    ar = 200.0
    r  = 0.004
    𝑋 = cosd(α)*X .- sind(α).*Y'
    𝑌 = sind(α)*X .+ cosd(α).*Y'
    phase[inx, iny][ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 2

end

@views function main(nc)
    #--------------------------------------------#

    # Scales
    sc = (σ = 1e8, L = 1e-2, t=1e12)

    # Boundary loading type
    config = :free_slip
    ε̇bg    = -1e-12*sc.t
    D_BC   = @SMatrix( [ -ε̇bg 0.;
                          0.  ε̇bg ])

    # Material parameters
    materials = ( 
        compressible = true,
        plasticity   = :DruckerPrager,
        n    = [1.0    1.0  ],
        η0   = [1e22   1e10 ]./(sc.σ * sc.t), 
        G    = [3e10   1e5  ]./(sc.σ),
        C    = [50e6   1e60 ]./(sc.σ),
        ϕ    = [30.    0.   ],
        ηvp  = [1e17   0.   ]./(sc.σ * sc.t),
        β    = [1e-11  4e-10].*(sc.σ),
        ψ    = [0.0    0.0  ],
        B    = [0.0    0.0  ],
        cosϕ = [0.0    0.0  ],
        sinϕ = [0.0    0.0  ],
        sinψ = [0.0    0.0  ],
    )
    # For power law
    materials.B   .= (2*materials.η0).^(-materials.n)

    # For plasticity
    @. materials.cosϕ  = cosd(materials.ϕ)
    @. materials.sinϕ  = sind(materials.ϕ)
    @. materials.sinψ  = sind(materials.ψ)

    # Time steps
    Δt0   = 1e8/sc.t
    nt    = nc.t

    # Newton solver
    niter = 20
    ϵ_nl  = 1e-8
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

    M_PC = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )
    𝐊_PC  = ExtendableSparseMatrix(nVx + nVy, nVx + nVy)
    𝐐_PC  = ExtendableSparseMatrix(nVx + nVy, nPt)
    𝐐ᵀ_PC = ExtendableSparseMatrix(nPt, nVx + nVy)
    𝐏_PC  = ExtendableSparseMatrix(nPt, nPt)

    #--------------------------------------------#
    # Intialise field
    L   = (x=1.0, y=1.0)
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
    @views Pt[inx_c, iny_c ]  .= 1e7/sc.σ                 
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

    # PorousMediumEllipses!(phases.c, inx_c, iny_c, xc, yc)
    # PorousMediumEllipses!(phases.v, inx_v, iny_v, xv, yv)

    PorousMediumCircles!(phases.c, inx_c, iny_c, xc, yc)
    PorousMediumCircles!(phases.v, inx_v, iny_v, xv, yv)

    # # Set material geometry 
    # @views phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    # @views phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2

    #--------------------------------------------#

    rvec = zeros(length(α))
    err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    
    ϕ = sum(phases.c.==2)/ *(size(phases.c)...)
    @show ϕ
    
    probes = ( 
        Pt =  zeros(nt),
        Pf =  zeros(nt),
        Ps =  zeros(nt),
        τt =  zeros(nt),
        τf =  zeros(nt),
        τs =  zeros(nt),
    )
    to   = TimerOutput()

    #--------------------------------------------#

    anim = @animate for it=1:nt

        @printf("Step %04d\n", it)
        fill!(err.x, 0e0)
        fill!(err.y, 0e0)
        fill!(err.p, 0e0)
        
        # Swap old values 
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Pt0   .= Pt

        for iter=1:niter

            @info "Newton iteration $(iter)"

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, type, BC, materials, phases, Δ)
                @show extrema(λ̇.c)
                @show extrema(λ̇.v)
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

            @timeit to "Assembly" begin
                AssembleContinuity2D!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M_PC, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, pattern, type, BC, nc, Δ)
            end

            #--------------------------------------------# 
            # Stokes operator as block matrices
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= M.Pt.Pt

            # Stokes operator as block matrices
            𝐊_PC  .= [M_PC.Vx.Vx M_PC.Vx.Vy; M_PC.Vy.Vx M_PC.Vy.Vy]
            𝐐_PC  .= [M_PC.Vx.Pt; M_PC.Vy.Pt]
            𝐐ᵀ_PC .= [M_PC.Pt.Vx M_PC.Pt.Vy]
            𝐏_PC  .= M_PC.Pt.Pt
            
            #--------------------------------------------#
     
            # Direct-iterative solver
            fu   = @views -r[1:size(𝐊,1)]
            fp   = @views -r[size(𝐊,1)+1:end]
            @timeit to "Solver" u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu,  ηb=1e3, niter_l=10, ϵ_l=1e-11, 𝐊_PC=𝐊_PC)
            @views dx[1:size(𝐊,1)]     .= u
            @views dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, type, BC, materials, phases, Δ)

        end

        # Update pressure
        Pt .+= ΔPt.c

        #--------------------------------------------#

        τxyc = av2D(τ.xy)
        τII  = sqrt.( 0.5.*(τ.xx[inx_c,iny_c].^2 + τ.yy[inx_c,iny_c].^2 + (-τ.xx[inx_c,iny_c]-τ.yy[inx_c,iny_c]).^2) .+ τxyc[inx_c,iny_c].^2 )
        ε̇xyc = av2D(ε̇.xy)
        ε̇II  = sqrt.( 0.5.*(ε̇.xx[inx_c,iny_c].^2 + ε̇.yy[inx_c,iny_c].^2 + (-ε̇.xx[inx_c,iny_c]-ε̇.yy[inx_c,iny_c]).^2) .+ ε̇xyc[inx_c,iny_c].^2 )
        
        fluid = phases.c .== 2
        solid = phases.c .== 1
        if sum(fluid) == 0
            τ_fluid = 0.
            P_fluid = 0.
        else
            τ_fluid = sum(τII[fluid[inx_c,iny_c]])/sum(fluid)
            P_fluid = sum(Pt[fluid])/sum(fluid)
        end
        P_solid = sum(Pt[solid])/sum(solid)
        P_total = ϕ*P_fluid + (1-ϕ)*P_solid
        τ_solid = sum(τII[solid[inx_c,iny_c]])/sum(solid)
        τ_total = ϕ*τ_fluid + (1-ϕ)*τ_solid

        τ_dry  = materials.C[1]*cosd(materials.ϕ[1]) +            P_total *sind(materials.ϕ[1])
        τ_terz = materials.C[1]*cosd(materials.ϕ[1]) + (P_total-  P_fluid)*sind(materials.ϕ[1])
        τ_shi  = materials.C[1]*cosd(materials.ϕ[1]) + (P_total-ϕ*P_fluid)*sind(materials.ϕ[1])

        @show (ϕ)
        @show (P_total)
        @show (P_total -  P_fluid)

        @show sum(fluid), P_fluid, τ_fluid
        @show sum(solid), P_solid, τ_solid
        probes.Pf[it] = P_fluid
        probes.Ps[it] = P_solid
        probes.Pt[it] = P_total
        probes.τf[it] = τ_fluid
        probes.τs[it] = τ_solid
        probes.τt[it] = τ_total

        # p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vx")
        p2 = heatmap(xc, yc,  Pt[inx_c,iny_c]'.*sc.σ, aspect_ratio=1, xlim=extrema(xc), title="Pt")
        # p3 = heatmap(xc, yc,  log10.(ε̇II)', aspect_ratio=1, xlim=extrema(xc), title="ε̇II", c=:coolwarm)
        p3 = plot(xlabel="time", ylabel="stress")
        p3 = plot!([1:it].*Δ.t,  probes.τf[1:it].*sc.σ, label="τ fluid")
        p3 = plot!([1:it].*Δ.t,  probes.τs[1:it].*sc.σ, label="τ solid")
        p3 = plot!([1:it].*Δ.t,  probes.τt[1:it].*sc.σ, label="τ total")
        p3 = plot!([1:it].*Δ.t,  probes.Pf[1:it].*sc.σ, label="P fluid")
        p3 = plot!([1:it].*Δ.t,  probes.Ps[1:it].*sc.σ, label="P solid")
        p3 = plot!([1:it].*Δ.t,  probes.Pt[1:it].*sc.σ, label="P total")
        p3 = plot!([1:it].*Δ.t,  τ_dry *ones(it).*sc.σ, label="τ dry",  linewidth=2)
        p3 = plot!([1:it].*Δ.t,  τ_terz*ones(it).*sc.σ, label="τ Terz", linewidth=2)
        p3 = plot!([1:it].*Δ.t,  τ_shi *ones(it).*sc.σ, label="τ Shi",  linewidth=2, legend=:outertopright)

        p4 = heatmap(xc, yc,  τII'.*sc.σ, aspect_ratio=1, xlim=extrema(xc), title="τII", c=:turbo)
        p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright)
        p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        display(plot(p1, p2, p3, p4, layout=(2,2)))

        @show (3/materials.β[1] - 2*materials.G[1])/(2*(3/materials.β[1] + 2*materials.G[1]))

    end
    # gif(anim, "./results/ShearBanding.gif", fps = 5)

    display(to)
    
end

let
    main((x = 200, y = 200, t=100))
end