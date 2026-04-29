using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
#using DifferentiationInterface
using TimerOutputs

#using JLD2
#using Makie

using ExactFieldSolutions


include("stokes_variablegrid.jl")
include("rheology_var.jl")

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x = 300, y = 300)

    # Setting for Schmid & Podladchikov (2003)
    params = (mm = 1.0, mc = 100.0, rc = 0.1 + 1e-13, gr = 0.0, er = 1.0)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template

    # Material parameters
    materials = (
        compressible = false,
        plasticity   = :none,
        g    = [0.0    0.0  ],
        ρ    = [1.0    1.0  ],
        n    = [1.0    1.0  ],
        η0   = [1e0    1e2  ],
        G    = [1e20   1e20 ],
        C    = [150    150  ],
        ϕ    = [30.    30.  ],
        ηvp  = [0.5    0.5  ],
        β    = [1e-2   1e-2 ],
        ψ    = [3.0    3.0  ],
        B    = [0.     0.   ],
        cosϕ = [0.0    0.0  ],
        sinϕ = [0.0    0.0  ],
        sinψ = [0.0    0.0  ],
    )
    materials.B   .= (2*materials.η0).^(-materials.n)

    # Time steps
    Δt0   = 0.5
    nt    = 1

    # Newton solver
    niter = 2
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


    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    ξ       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
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

    # Intialize field
    L   = (x=1.0, y=1.0)

    uniform_grid = false
    if uniform_grid
        original = false
        if original
            Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)
            # Mesh coordinates
            xv = LinRange(-L.x/2, L.x/2, nc.x+1)
            yv = LinRange(-L.y/2, L.y/2, nc.y+1)
            xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
            yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
        else
            Δ   = (x = fill(L.x/nc.x,nc.x+2), y = fill(L.y/nc.y,nc.y+2), t=fill(Δt0,1))

            xmin = -L.x/2
            xmax = L.x/2
            ymin = -L.y/2
            ymax = L.y/2
            xhalf = Δ.x[1]/2
            yhalf = Δ.y[1]/2
            # Mesh coordinates
            xv = LinRange(xmin, xmax, nc.x+1)
            yv = LinRange(ymin, ymax, nc.y+1)

            xc  = LinRange(xmin+xhalf, xmax-xhalf, nc.x)
            yc  = LinRange(ymin+yhalf, ymax-yhalf, nc.y)
            # With ghosts
            xv_g = LinRange(xmin-Δ.x[1], xmax+Δ.x[1], nc.x+3)
            yv_g = LinRange(ymin-Δ.y[1], ymax+Δ.y[1], nc.y+3)
            xc_g = LinRange(xmin-xhalf, xmax+xhalf, nc.x+2)
            yc_g = LinRange(ymin-yhalf, ymax+yhalf, nc.y+2)

        end
    else
        μ = ( x = 0.0, y = 0.0)
        σ = ( x = 0.2, y = 0.2)
        inflimit = (x = -L.x/2, y = -L.y/2)
        suplimit = (x =  L.x/2, y =  L.y/2)

        # nodes
        xv = normal_linspace_interval(inflimit.x, suplimit.x, μ.x, σ.x, nc.x+1)
        yv = normal_linspace_interval(inflimit.y, suplimit.y, μ.y, σ.y, nc.y+1)

        # spaces between nodes
        Δ = (x = zeros(nc.x+2), y = zeros(nc.y+2), t=fill(Δt0,1)) # nb cells
            
        Δ.x[2:end-1]   .= diff(xv)
        Δ.x[[1, end]] .= Δ.x[[2, end-1]]
        Δ.y[2:end-1]   .= diff(yv)
        Δ.y[[1, end]] .= Δ.y[[2, end-1]]

        endv = nc.x+1
            
        xc = 0.5*(xv[2:endv] + xv[1:endv-1])
        yc = 0.5*(yv[2:endv] + yv[1:endv-1])

        # With ghosts
        xv_g = normal_linspace_interval(inflimit.x-Δ.x[1], suplimit.x+Δ.x[end], μ.x, σ.x, nc.x+3)
        yv_g = normal_linspace_interval(inflimit.y-Δ.y[1], suplimit.y+Δ.y[end], μ.y, σ.y, nc.y+3)
        endv_g = nc.x+3
        xc_g = 0.5*(xv_g[2:endv_g] + xv_g[1:endv_g-1])
        yc_g = 0.5*(yv_g[2:endv_g] + yv_g[1:endv_g-1])

    end

    
    # Mesh coordinates
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    # Initial velocity & pressure field
    display(Ranges(nc))
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc'
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    Pt[inx_c, iny_c ]  .= 10.
    UpdateSolution_var!(V, Pt, dx, number, type, nc)


    # Boundary Conditions
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))

    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]

    # left
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    # right
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])

    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    #north
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    # south
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)

    # compute analytic solution and set BC Dirichlet for X and Y
    p_ana = zeros(nc.x, nc.y)
    for i=1:nc.x, j=1:nc.y
        sol       = Stokes2D_Schmid2003( [xc[i]; yc[j]] )
        p_ana[i,j]    = sol.p
    end
    Vx_ana = zero(BC.Vx)
    for i=1:size(BC.Vx,1), j=2:size(BC.Vx,2)-1
        sol       = Stokes2D_Schmid2003( [xv_g[i]; yc_g[j-1]] ; params )
        Vx_ana[i,j]   = sol.V[1]
        V.x[i,j] = sol.V[1]
        BC.Vx[i,j]    = sol.V[1]
    end
    Vy_ana = zero(BC.Vy)
    for i=2:size(BC.Vy,1)-1, j=1:size(BC.Vy,2)
        sol       = Stokes2D_Schmid2003( [xc_g[i-1]; yv_g[j]] ; params)
        Vy_ana[i,j]   = sol.V[2]
        V.y[i,j] = sol.V[2]
        BC.Vy[i,j]    = sol.V[2]
    end

    # Set material geometry
    rad = 0.1 + 1e-13
    phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= rad^2] .= 2
    phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .<= rad^2] .= 2

    #--------------------------------------------#

    rvec = zeros(length(α))
    err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    to   = TimerOutput()

    #--------------------------------------------#

    for it=1:nt

        @printf("Step %04d\n", it)
        err.x .= 0.
        err.y .= 0.
        err.p .= 0.
        
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
                TangentOperator_var!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)
                ResidualContinuity2D_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_x_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            end

            err.x[iter] = norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
            err.y[iter] = norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
            err.p[iter] = norm(R.p[inx_c,iny_c])/sqrt(nPt)
            max(err.x[iter], err.y[iter]) < ϵ_nl ? break : nothing

            #--------------------------------------------#
            # Set global residual vector
            SetRHS!(r, R, number, type, nc)

            #--------------------------------------------#
            # Assembly
            @timeit to "Assembly" begin
                    AssembleContinuity2D_var!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                    AssembleMomentum2D_x_var!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                    AssembleMomentum2D_y_var!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
            end

            #--------------------------------------------# 
            # Stokes operator as block matrices
            #println(M.Vx.Vx[90:100,90])
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= [M.Pt.Pt;]

            #--------------------------------------------#
     
            # Direct-iterative solver
            fu   = -r[1:size(𝐊,1)]
            fp   = -r[size(𝐊,1)+1:end]
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
            dx[1:size(𝐊,1)]     .= u
            dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch_var!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            UpdateSolution_var!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator_var!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)
        end

        # Update pressure
        Pt .+= ΔPt.c 

        # Remove mean 
        Pt[inx_c,iny_c]' .-= mean(Pt[inx_c,iny_c])

        ϵV = (
        x   = zero(BC.Vx),
        y   = zero(BC.Vy),
        )
        ϵP   = zero(Pt)

        # Compute errors
        for I in eachindex(Pt[inx_c,iny_c])        
            ϵP[I] = abs(p_ana[I] - Pt[inx_c,iny_c][I])
        end

        for I in eachindex(V.x)
            ϵV.x[I] = abs(Vx_ana[I] - V.x[I])
        end

        for I in eachindex(V.y)
            ϵV.y[I] = abs(Vy_ana[I] - V.y[I])
        end

        @info mean(abs.(ϵV.x))
        @info mean(abs.(ϵV.y))
        @info mean(abs.(ϵP))

        # Visualisation Analytical solutions V.x V.y Pt
        p3 = heatmap(xv, yc, Vx_ana[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xv_g), title="Vx analytic", color=:vik)
        p4 = heatmap(xc, yv, Vy_ana[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc_g), title="Vy analytic", color=:vik)
        p2 = heatmap(xc, yc, p_ana'.-mean(p_ana), aspect_ratio=1, xlim=extrema(xc), title="Pt analytic", color=:vik, clim=extrema(p_ana))
        p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright, title=BC_template)
        p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        display(plot(p1, p2, p3, p4, layout=(2,2)))
        sleep(6)

        # Solution V.x V.y Pt
        p3 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xv), title="Vx", color=:vik)
        p4 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc), title="Vy", color=:vik)
        p2 = heatmap(xc, yc,  Pt[inx_c,iny_c]'.-mean( Pt[inx_c,iny_c]), aspect_ratio=1, xlim=extrema(xc), title="Pt", color=:vik, clim=extrema(p_ana))
        p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright, title=BC_template)
        p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        display(plot(p1, p2, p3, p4, layout=(2,2)))
        sleep(6)

        # Visulisation Epsi
        p3 = heatmap(xv, yc, ϵV.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xv), title="Vx Epsi", color=:vik)
        p4 = heatmap(xc, yv, ϵV.y[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vy Epsi", color=:vik)
        p2 = heatmap(xc, yc, ϵP[inx_c,iny_c], aspect_ratio=1, xlim=extrema(xc_g), title="Pt Epsi", color=:vik)
        p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright, title="Convergence")
        p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        display(plot(p1, p2, p3, p4, layout=(2,2)))
        sleep(6)

        # Visualisation of difference between analytical solution and numerical solution
        p3 = heatmap(xv, yc, (V.x[inx_Vx,iny_Vx]'-Vx_ana[inx_Vx,iny_Vx]') / sqrt(size(Vx_ana[inx_Vx,iny_Vx]',1)), aspect_ratio=1, xlim=extrema(xv_g), title="Vx diff", color=:vik)
        p4 = heatmap(xc, yv, (V.y[inx_Vx,iny_Vx]'-Vy_ana[inx_Vx,iny_Vx]') / sqrt(size(Vy_ana[inx_Vx,iny_Vx]',1)), aspect_ratio=1, xlim=extrema(xc_g), title="Vy diff", color=:vik)
        p2 = heatmap(xc, yc, ((Pt'[inx_c,iny_c].-mean( Pt[inx_c,iny_c])) - (p_ana'.-mean(p_ana))) / sqrt(size(p_ana,1)), aspect_ratio=1, xlim=extrema(xc), title="Pt diff", color=:vik)
        p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright, title=BC_template)
        p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        display(plot(p1, p2, p3, p4, layout=(2,2)))
        sleep(6)

    end

    display(to)
    
end


let
    # # Boundary condition templates
    #BCs = [
    #    :free_slip,
    #]

    # # Boundary deformation gradient matrix
    # D_BCs = [
    #     @SMatrix( [1 0; 0 -1] ),
    # ]

    BCs = [
    #     # :EW_periodic,
         :all_Dirichlet,
     ]

    # Boundary velocity gradient matrix
    D_BCs = [
         @SMatrix( [1 0; 0 -1] ),
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(BCs[iBC], D_BCs[iBC])
    end
end
