using StagFDTools, StagFDTools.StokesJustPIC, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf, GLMakie
import Statistics:mean
using JustPIC, JustPIC._2D
import JustPIC.@index
const backend = JustPIC.CPUBackend 
using DifferentiationInterface
using TimerOutputs

function compute_shear_bulk_moduli!(G, β, materials, phase_ratios, nc, size_c, size_v)
    sum       = (c  =  ones(size_c...), v  =  ones(size_v...) )

    for I in CartesianIndices(β.c) 
        i, j = I[1], I[2]
        β.c[i,j] = 0.0
        G.c[i,j] = 0.0
        sum.c[i,j] = 0.0
        for p = 1:2 # loop on phases
            if i>1 && j>1 && i<nc.x+2 && j<nc.y+2 
                phase_ratio = @index phase_ratios.center[p, i-1, j-1]
                β.c[i,j]   += phase_ratio * materials.β[p]
                G.c[i,j]   += phase_ratio * materials.G[p]
                sum.c[i,j] += phase_ratio
            end
        end
    end
    G.c[[1 end],:] .=  G.c[[2 end-1],:]
    G.c[:,[1 end]] .=  G.c[:,[2 end-1]]
    β.c[[1 end],:] .=  β.c[[2 end-1],:]
    β.c[:,[1 end]] .=  β.c[:,[2 end-1]]

    for I in CartesianIndices(G.v) 
        i, j = I[1], I[2]
        G.v[i,j]   = 0.0
        sum.v[i,j] = 0.0
        for p = 1:2 # loop on phases
            if i>1 && j>1 && i<nc.x+3 && j<nc.y+3 
                phase_ratio = @index phase_ratios.vertex[p, i-1, j-1]
                G.v[i,j]   += phase_ratio * materials.G[p]
                sum.v[i,j] += phase_ratio
            end
        end
    end
    G.v[[1 end],:] .=  G.v[[2 end-1],:]
    G.v[:,[1 end]] .=  G.v[:,[2 end-1]]
    @show extrema(sum.c),  extrema(sum.v)
end

function set_phases!(phases, particles)
    Threads.@threads for j in axes(phases, 2)
        for i in axes(phases, 1)
            for ip in cellaxes(phases)
                # quick escape
                @index(particles.index[ip, i, j]) == 0 && continue
                x = @index particles.coords[1][ip, i, j]
                y = @index particles.coords[2][ip, i, j]
                if (x^2 + (y)^2) <= 0.1^2
                    @index phases[ip, i, j] = 2.0
                else
                    @index phases[ip, i, j] = 1.0
                end
            end
        end
    end
end

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x = 25, y = 25)

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
        η0   = [1e0    1e5  ], 
        G    = [1e60   2e60 ],
        C    = [150    150  ],
        ϕ    = [30.    30.  ],
        ηvp  = [0.5    0.5  ],
        β    = [1e-20  2e-20],
        ψ    = [3.0    3.0  ],
        B    = [0.     0.   ],
        cosϕ = [0.0    0.0  ],
        sinϕ = [0.0    0.0  ],
        sinψ = [0.0    0.0  ],
    )
    materials.B   .= (2*materials.η0).^(-materials.n)

    # Time steps
    Δt0   = 0.5
    nt    = 20
    ALE   = true

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

    #--------------------------------------------#
    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    G       = (c  =  ones(size_c...), v  =  ones(size_v...) )
    β       = (c  =  ones(size_c...),)
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...) )
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
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
    xlims = [-L.x/2, L.x/2]
    ylims = [-L.y/2, L.y/2]
    xv  = LinRange(xlims[1], xlims[2], nc.x+1)
    yv  = LinRange(ylims[1], ylims[2], nc.y+1)
    xc  = LinRange(xlims[1]+Δ.x/2, xlims[2]-Δ.x/2, nc.x)
    yc  = LinRange(ylims[1]+Δ.y/2, ylims[2]-Δ.y/2, nc.y)
    xce = LinRange(xlims[1]-Δ.x/2, xlims[2]+Δ.x/2, nc.x+2)
    yce = LinRange(ylims[1]-Δ.y/2, ylims[2]+Δ.y/2, nc.y+2)

    # Initial velocity & pressure field
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    Pt[inx_c, iny_c ]  .= 0.0                 
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

    # Initialize particles
    nxcell    = (6,6) # initial number of particles per cell
    max_xcell = 36*2 # maximum number of particles per cell
    min_xcell = 1 # minimum number of particles per cell
    xci = (xc, yc)
    xvi = (xv, yv)
    d  = (Δ.x, Δ.y)
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, values(xvi), values(d), values(nc)
    )

    # Initialise phase field
    particle_args = phases, = init_cell_arrays(particles, Val(1))  # cool

    # Set material geometry 
    set_phases!(phases, particles)
    phase_ratios = JustPIC._2D.PhaseRatios(backend, 2, values(nc));
    update_phase_ratios!(phase_ratios, particles, xci, xvi, phases)

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

        # Compute bulk and shear moduli
        compute_shear_bulk_moduli!(G, β, materials, phase_ratios, nc, size_c, size_v)

        for iter=1:niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, β, V, Pt, ΔPt, type, BC, materials, phase_ratios, Δ)
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, β, materials, number, type, BC, nc, Δ) 
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, G, materials, number, type, BC, nc, Δ)
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
                AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, β, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, G, materials, number, pattern, type, BC, nc, Δ)
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
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol,  ηb=1e4, niter_l=10, ϵ_l=1e-10)
            dx[1:size(𝐊,1)]     .= u
            dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, G, β, 𝐷, 𝐷_ctl, number, type, BC, materials, phase_ratios, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, G, β, V, Pt, ΔPt, type, BC, materials, phase_ratios, Δ)
        end

        # Update pressure    
        Pt .+= ΔPt.c 

        # Advection with JustPIC
        C       = 0.1
        Vmax    = max(maximum(abs.(V.x)), maximum(abs.(V.y)))
        Δ       = (x=L.x/nc.x, y=L.y/nc.y, t = C * min(Δ.x, Δ.y)/Vmax)
        grid_vx = (xv, yce)
        grid_vy = (xce, yv)
        V_adv   = (x=V.x[2:end-1,2:end-1], y=V.y[2:end-1,2:end-1])
        advection!(particles, RungeKutta4(), values(V_adv), (grid_vx, grid_vy), Δ.t)
        move_particles!(particles, values(xvi), particle_args)
        inject_particles_phase!(particles, phases, (), (), values(xvi))
        update_phase_ratios!(phase_ratios, particles, xci, xvi, phases)
        compute_shear_bulk_moduli!(G, β, materials, phase_ratios, nc, size_c, size_v)

        if ALE
            ε̇bg = D_BC[1,1]
            xlims[1] += xlims[1]*ε̇bg*Δ.t 
            xlims[2] += xlims[2]*ε̇bg*Δ.t
            ylims[1] -= ylims[1]*ε̇bg*Δ.t 
            ylims[2] -= ylims[2]*ε̇bg*Δ.t
            @show L  = ( x =(xlims[2]-xlims[1]), y =(ylims[2]-ylims[1]) )  
            Δ  = (x=L.x/nc.x, y=L.y/nc.y )
            xv  = LinRange(xlims[1], xlims[2], nc.x+1)
            yv  = LinRange(ylims[1], ylims[2], nc.y+1)
            xc  = LinRange(xlims[1]+Δ.x/2, xlims[2]-Δ.x/2, nc.x)
            yc  = LinRange(ylims[1]+Δ.y/2, ylims[2]-Δ.y/2, nc.y)
            xce = LinRange(xlims[1]-Δ.x/2, xlims[2]+Δ.x/2, nc.x+2)
            yce = LinRange(ylims[1]-Δ.y/2, ylims[2]+Δ.y/2, nc.y+2)
            grid_vx = (xv, yce)
            grid_vy = (xce, yv)
            # Δt = C * min(Δ...) / max(maximum(abs.(V.x)), maximum(abs.(V.y)))
            # @parallel SetVelocity(V, verts, ε̇bg)

            # Initial velocity & pressure field
            V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
            V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
            Pt[inx_c, iny_c ]  .= 0.0                 
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

            Δ       = (x=L.x/nc.x, y=L.y/nc.y, t = C * min(Δ.x, Δ.y)/Vmax)
            move_particles!(particles, values(xvi), particle_args)
            inject_particles_phase!(particles, phases, (), (), values(xvi))
            update_phase_ratios!(phase_ratios, particles, xci, xvi, phases)
        end

        #--------------------------------------------#

        # Visualise
        function visualisation()
            phc = [p[1] for p in phase_ratios.center]
            phv = [p[1] for p in phase_ratios.vertex]
            #-----------  
            fig = Figure(size=(500,800))
            #-----------
            ax  = Axis(fig[1,1], aspect=DataAspect(), title="Pressure", xlabel="x", ylabel="y")
            heatmap!(ax, xc, yc,  (Pt[inx_c,iny_c]), colormap=:bluesreds)
            Vxc = 0.5.*(V_adv.x[1:end-1,2:end-1] .+ V_adv.x[2:end,2:end-1])
            Vyc = 0.5.*(V_adv.y[2:end-1,1:end-1] .+ V_adv.y[2:end-1,2:end])
            arrows2d!(ax, xc, yc, Vxc, Vyc, lengthscale = 0.05)
            ax  = Axis(fig[1,2], aspect=DataAspect(), title="Particles", xlabel="x", ylabel="y")
            p    = particles.coords
            ppx, ppy = p
            pxv  = ppx.data[:]
            pyv  = ppy.data[:]
            clr  = phases.data[:]
            idxv = particles.index.data[:]
            scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=5)
            ax  = Axis(fig[2,1], aspect=DataAspect(), title="Txx", xlabel="x", ylabel="y")
            heatmap!(ax, xc, yc,  τ.xx, colormap=:bluesreds)
            ax  = Axis(fig[2,2], aspect=DataAspect(), title="Tyy", xlabel="x", ylabel="y")
            heatmap!(ax, xc, yc,  τ.yy, colormap=:bluesreds)
            ax  = Axis(fig[3,1], aspect=DataAspect(), title="phc", xlabel="x", ylabel="y")
            heatmap!(ax, xc, yc,  G.c[inx_c,iny_c], colormap=:bluesreds)
            ax  = Axis(fig[3,2], aspect=DataAspect(), title="phv", xlabel="x", ylabel="y")
            heatmap!(ax, xv, yv,  G.v[inx_v,iny_v], colormap=:bluesreds)
            #-----------
            display(fig)
        end
        with_theme(visualisation, theme_latexfonts())
    end
    # display(to)
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

    # Boundary velocity gradient matrix
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