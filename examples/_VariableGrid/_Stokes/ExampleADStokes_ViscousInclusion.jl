using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
#using DifferentiationInterface
using Enzyme  # AD backends you want to use
using TimerOutputs

include("stokes_vargrid_corrected.jl")
include("rheology_var.jl")

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x = 5, y = 5)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template

    # Material parameters
    materials = ( 
        compressible = false,
        plasticity   = :none,
        n    = [1.0    1.0  ],
        η0   = [1e0    1e5  ], 
        G    = [1e6    1e6  ],
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

    #--------------------------------------------#
    # Intialise field
    #L   = (x=1.0, y=1.0)
    #Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    
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
            display(Δ.x)
            display(Δ.y)
            display(Δ.t)

            # Mesh coordinates
            xv = LinRange(-L.x/2, L.x/2, nc.x+1)
            yv = LinRange(-L.y/2, L.y/2, nc.y+1)
            #xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x) # sans les ghosts??
            #yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y) # idem ici ?? why?
            # je reproduis ça quand même sans les ghosts pour les tests
            xc  = LinRange(-L.x/2+Δ.x[1]/2, L.x/2-Δ.x[end]/2, nc.x)
            yc  = LinRange(-L.y/2+Δ.y[1]/2, L.y/2-Δ.y[end]/2, nc.y)
            display(xc)
            display(yc)        
        end
    else

        μ = ( x = 0.0, y = 0.0)
        σ = ( x = 0.2, y = 0.2)
        inflimit = (x = -L.x/2, y = -L.y/2)
        suplimit = (x =  L.x/2, y =  L.y/2)

        # nodes
        xv_in = normal_linspace_interval(inflimit.x, suplimit.x, μ.x, σ.x, nc.x-1)
        yv_in = normal_linspace_interval(inflimit.y, suplimit.y, μ.y, σ.y, nc.y-1)

        # spaces between nodes
        Δ = (x = zeros(nc.x+2), y = zeros(nc.y+2), t=fill(Δt0,1)) # nb cells
        enddelta = nc.x+2
        println(diff(xv_in))
        println(Δ.x[2:enddelta-1])
        Δ.x[3:enddelta-2] = diff(xv_in) # le diff fait perdre une dimension
        Δ.y[3:enddelta-2] = diff(yv_in)
        Δ.x[1]   = Δ.x[3]
        Δ.x[2]   = Δ.x[3]
        Δ.x[enddelta] = Δ.x[enddelta-2]
        Δ.x[enddelta-1] = Δ.x[enddelta-2]
        println("delta x")
        println(enddelta)
        println(Δ.x[enddelta])
        println(Δ.x)
        Δ.y[1]   = Δ.y[3]
        Δ.y[2]   = Δ.y[3]
        Δ.y[enddelta] = Δ.y[enddelta-2]
        Δ.y[enddelta-1] = Δ.y[enddelta-2]

        endv = nc.x+1
        xv  = zeros(endv)
        yv  = zeros(endv)
        xv[2:endv-1] .= xv_in # nc.x+1-2+1=nc.x-1, c'est bien xvin
        xv[1]   = xv[2] - Δ.x[2]
        xv[endv] = xv[endv-1] + Δ.x[enddelta-1]
        yv[2:endv-1] .= yv_in
        yv[1]   = yv[2] - Δ.y[2]
        yv[endv] = yv[endv-1] + Δ.y[enddelta-1]
        xc = 0.5*(xv[2:endv] + xv[1:endv-1])
        yc = 0.5*(yv[2:endv] + yv[1:endv-1])

        display("xc")
        println(xc)
        println("xv")
        println(xv)
        println("Δ.x")
        println(Δ.x)

    end

    display(size(xc))
    display(size(yc))
    display(size(xv))
    display(size(yv))
    
    # Mesh coordinates
    #xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    #yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    #xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    #yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    # Initial velocity & pressure field
    display(Ranges(nc))
    println(xv)
    println(xc)
    println(yv')
    println(yc')
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    Pt[inx_c, iny_c ]  .= 10.                 
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)

    # Set material geometry 
    phases.c[inx_c, iny_c][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    phases.v[inx_v, iny_v][(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2

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
                TangentOperator_var!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, type, BC, materials, phases, Δ)
                ResidualContinuity2D_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                ResidualMomentum2D_x_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
            end

            display(R.x[3,3])
            display(R.y[3,3])

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
            𝐊  .= [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
            𝐐  .= [M.Vx.Pt; M.Vy.Pt]
            𝐐ᵀ .= [M.Pt.Vx M.Pt.Vy]
            𝐏  .= [M.Pt.Pt;]             
            
            #--------------------------------------------#
     
            # Direct-iterative solver
            fu   = -r[1:size(𝐊,1)]
            fp   = -r[size(𝐊,1)+1:end]
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
            dx[1:size(𝐊,1)]     .= u
            dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch_var!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator_var!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, type, BC, materials, phases, Δ)
        end

        # Update pressure
        Pt .+= ΔPt.c 

        #--------------------------------------------#

        p3 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xv), title="Vx", color=:vik)
        p4 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc), title="Vy", color=:vik)
        p2 = heatmap(xc, yc,  Pt[inx_c,iny_c]'.-mean( Pt[inx_c,iny_c]), aspect_ratio=1, xlim=extrema(xc), title="Pt", color=:vik)
        p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright, title=BC_template)
        p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        display(plot(p1, p2, p3, p4, layout=(2,2)))
        sleep(10)

    end

    display(to)
    
end


let
    # # Boundary condition templates
    BCs = [
        :free_slip,
    ]

    # # Boundary deformation gradient matrix
    # D_BCs = [
    #     @SMatrix( [1 0; 0 -1] ),
    # ]

    # BCs = [
    #     # :EW_periodic,
    #     :all_Dirichlet,
    # ]

    # Boundary deformation gradient matrix
    D_BCs = [
        #  @SMatrix( [0 1; 0  0] ),
        @SMatrix( [1 0; 0 -1] ),
    ]

    # Run them all
    for iBC in eachindex(BCs)
        @info "Running $(string(BCs[iBC])) and D = $(D_BCs[iBC])"
        main(BCs[iBC], D_BCs[iBC])
    end
end