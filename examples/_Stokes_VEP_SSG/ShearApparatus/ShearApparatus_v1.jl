using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GridGeometryUtils
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use
using TimerOutputs, CairoMakie

import Makie.SpecApi as S

@views function main(nc, θgouge)
    #--------------------------------------------#

    # Resolution

    # Boundary loading type
    config = :free_slip
    D_BC   = @SMatrix( [  1.  0.;
                          0  -1 ])

    # Material parameters
    materials = ( 
        compressible = true,
        plasticity   = :DruckerPrager,
        n    = [1.0    1.0    1.0 ],
        η0   = [1e3    1e3    1e-3], 
        G    = [1e1    2e1    1e1 ],
        C    = [150    100    150 ],
        ϕ    = [35.    30.    35. ],
        ηvp  = [1.0    1.0    1.0 ].*0.6,
        β    = [1e-2   0.5e-2   1e-2],
        ψ    = [0.0    5.0    0.0 ],
        B    = [0.0    0.0    0.0 ],
        cosϕ = [0.0    0.0    0.0 ],
        sinϕ = [0.0    0.0    0.0 ],
        sinψ = [0.0    0.0    0.0 ],
    )
    # For power law
    materials.B   .= (2*materials.η0).^(-materials.n)

    # For plasticity
    @. materials.cosϕ  = cosd(materials.ϕ)
    @. materials.sinϕ  = sind(materials.ϕ)
    @. materials.sinψ  = sind(materials.ψ)
    
    # Geometry
    gouge = (
        Rectangle((0.0, 0.0), 0.2, 2.0; θ = θgouge),
    )
    salt = (
        Rectangle((-.5, 0.0), 0.5, 2.0; θ = 0),
        Rectangle((0.5, 0.0), 0.5, 2.0; θ = 0),
    )

    # Time steps
    Δt0   = 0.25
    nt    = 100

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

    #--------------------------------------------#
    # Intialise field
    L   = (x=1.0, y=1.5)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    Pt      = zeros(size_c...)
    Pti     = zeros(size_c...)
    Pt0     = zeros(size_c...)
    ΔPt     = zeros(size_c...)
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
    @views Pt[inx_c, iny_c ]  .= 10.                 
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
    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([xc[i-1], yc[j-1]])

        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                phases.c[i, j] = 2
            end
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                phases.c[i, j] = 3
            end
        end
    end

    for i in inx_c, j in iny_c  # loop on vertices
        𝐱 = @SVector([xv[i-1], yv[j-1]])

        for igeom in eachindex(gouge) # Gouge: phase 2
            if inside(𝐱, gouge[igeom])
                phases.v[i, j] = 2
            end  
        end
        for igeom in eachindex(salt) # Salt: phase 3
            if inside(𝐱, salt[igeom])
                phases.v[i, j] = 3
            end  
        end
    end

    Pt  .= 200*rand(size(Pt)...)
    Pt0 .= Pt
    Pti .= Pt

    #--------------------------------------------#

    rvec   = zeros(length(α))
    err    = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    probes = (τII = zeros(nt), t = zeros(nt))
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

        for iter=1:niter

            @printf("Iteration %04d\n", iter)

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
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
            @views dx[1:size(𝐊,1)]     .= u
            @views dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, type, BC, materials, phases, Δ)

        end

        # Update pressure
        Pt .+= ΔPt

        #--------------------------------------------#

        # Post process stress and strain rate
        τxyc = av2D(τ.xy)
        τII  = sqrt.( 0.5.*(τ.xx[inx_c,iny_c].^2 + τ.yy[inx_c,iny_c].^2 + (-τ.xx[inx_c,iny_c]-τ.yy[inx_c,iny_c]).^2) .+ τxyc[inx_c,iny_c].^2 )
        ε̇xyc = av2D(ε̇.xy)
        ε̇II  = sqrt.( 0.5.*(ε̇.xx[inx_c,iny_c].^2 + ε̇.yy[inx_c,iny_c].^2 + (-ε̇.xx[inx_c,iny_c]-ε̇.yy[inx_c,iny_c]).^2) .+ ε̇xyc[inx_c,iny_c].^2 )
        
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
        probes.t[it]   = it*Δ.t
        probes.τII[it] = mean(τII)

        # Visualise
        fig = Figure()
        ax  = Axis(fig[1:2,1], aspect=DataAspect(), title="Strain rate", xlabel="x", ylabel="y")
        heatmap!(ax, xc, yc,  log10.(λ̇.c), colormap=:bluesreds)
        contour!(ax, xc, yc,  phases.c[inx_c,iny_c], color=:black)
        st = 10
        # arrows!(ax, xc[1:st:end], yc[1:st:end], σ1.x[inx_c,iny_c][1:st:end,1:st:end], σ1.y[inx_c,iny_c][1:st:end,1:st:end], arrowsize = 0, lengthscale=0.04, linewidth=2, color=:white)
        ax  = Axis(fig[1,2], xlabel="Time", ylabel="mean(τII)")
        scatter!(ax, probes.t[1:nt], probes.τII[1:nt] )
        ax  = Axis(fig[2,2], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error")
        scatter!(ax, 1:niter, log10.(err.x[1:niter]) )
        scatter!(ax, 1:niter, log10.(err.y[1:niter]) )
        scatter!(ax, 1:niter, log10.(err.p[1:niter]) )
        ylims!(ax, -10, 5)
        display(fig)
       
        # @show (3/materials.β[1] - 2*materials.G[1])/(2*(3/materials.β[1] + 2*materials.G[1]))

    end

    display(to)
    
end

let
    main((x = 250, y = 200), -π/2.4)
end