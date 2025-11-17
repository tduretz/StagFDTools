using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use
using TimerOutputs

@views function main(nc)
    #--------------------------------------------#

    # Boundary loading type
    config = :free_slip
    D_BC   = @SMatrix( [ -5e-6 0.;
                          0  5e-6 ])

    # Material parameters
    materials = ( 
        compressible = true,
        plasticity   = :DruckerPrager,
        n    = [1.0    1.0  ],
        η0   = [1e50    1e45 ], 
        G    = [1.0    0.25  ],
        C    = [1.74e-4    1.74e-4  ],
        ϕ    = [30.    30.  ],
        ηvp  = [2.5e2    2.5e2  ],
        β    = [0.5   0.5 ],
        ψ    = [10.0    10.0  ],
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
    Δt0   = 1e5
    nt    = 8

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
    L   = (x=1.0, y=0.7)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y, t = Δt0)

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...) )
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
    ccord = (x=-L.x/2, y=-L.y/2)
    @views phases.c[inx_c, iny_c][((xc.-ccord.x).^2 .+ ((yc').-ccord.y).^2) .<= (25e-4)] .= 2
    @views phases.v[inx_v, iny_v][((xv.-ccord.x).^2 .+ ((yv').-ccord.y).^2) .<= (25e-4)] .= 2

    #--------------------------------------------#

    # Post-processing initialisation
    rvec = zeros(length(α))
    err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    ε0 = zeros(nc.x,nc.y)
    ε_acc = zeros(nc.x,nc.y)
    to   = TimerOutput()
    strain_evo = true
    if strain_evo
       z7 = plot(xlabel = "x", ylabel = "εᵢᵢ[x 10⁻⁹]", title = "Accumulated strain")
    end

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
        ε0    .= ε_acc

        for iter=1:niter

            @printf("Iteration %04d\n", iter)

            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)
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
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)

        end

        # Update pressure
        Pt .+= ΔPt.c

        #--------------------------------------------#

        τxyc = av2D(τ.xy)
        τII  = sqrt.( 0.5.*(τ.xx[inx_c,iny_c].^2 + τ.yy[inx_c,iny_c].^2 + (-τ.xx[inx_c,iny_c]-τ.yy[inx_c,iny_c]).^2) .+ τxyc[inx_c,iny_c].^2 )
        ε̇xyc = av2D(ε̇.xy)
        ε̇II  = sqrt.( 0.5.*(ε̇.xx[inx_c,iny_c].^2 + ε̇.yy[inx_c,iny_c].^2 + (-ε̇.xx[inx_c,iny_c]-ε̇.yy[inx_c,iny_c]).^2) .+ ε̇xyc[inx_c,iny_c].^2 )
        
        z1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vx")
        z2 = heatmap(xc, yc,  Pt[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Pt")
        z3 = heatmap(xc, yc,  log10.(ε̇II)', aspect_ratio=1, xlim=extrema(xc), title="ε̇II", c=:coolwarm)
        z4 = heatmap(xc, yc,  τII', aspect_ratio=1, xlim=extrema(xc), title="τII", c=:turbo)
        z1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright)
        z1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        z1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        z1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        display(plot(z1, z2, z3, z4, layout=(2,2)))

        @show (3/materials.β[1] - 2*materials.G[1])/(2*(3/materials.β[1] + 2*materials.G[1]))

        if strain_evo 
            ## Accumulated strain 
        ε1 = ε̇II./Δ.t
        ε_acc = ε1 + ε0

        # Angle of the section: Roscoe angle
        θ = 45. - (materials.ϕ[1] + materials.ψ[1])/4
        θrad = deg2rad(θ)

        # Section initialisation
        lenght = Δ.y*nc.y
        C = zeros(2,Int64(round(lenght/Δ.y)))
        ind′ = zeros(Int,2,Int64(round(lenght/Δ.y)))
        C[1,:] .= L.x*0.5 - L.x*0.5 
        C[2,:]  = LinRange(-lenght*0.5, lenght*0.5,Int64(round(lenght/Δ.y)))

        # Rotation matrix
        Rot = [cos(θrad) -sin(θrad); cos(θrad) sin(θrad)]
    
        # Rotate the section to be normal to shear band angle
        C′ = Rot *C

        # Find indices of the line points
        for i = 1 : Int64(round(lenght/Δ.y))
            ind′[1,i] = Int64(round(C′[1,i]/Δ.x + nc.x*0.5 + 0.5))
            ind′[2,i] = Int64(round(C′[1,i]/Δ.y + nc.y*0.5 + 0.5))
        end

        cross_sec = map(CartesianIndex, ind′[2,:], ind′[1,:])
        ε_prof = ε_acc[cross_sec]

        # Plot time evolution
        plot!(z7,C[2,:],(ε_prof).*10e9, label = "$(it*Δ.t*10e-5) s [x 10⁵]")
        display(z7)
        end

    end
    
    # -----------------------------------------------------------------
    # Profiles across the shear band
        
    ## Strain rate
    ε̇xyc  = av2D(ε̇.xy)
    ε̇II   = sqrt.( 0.5.*(ε̇.xx[inx_c,iny_c].^2 + ε̇.yy[inx_c,iny_c].^2 + (-ε̇.xx[inx_c,iny_c]-ε̇.yy[inx_c,iny_c]).^2) .+ ε̇xyc[inx_c,iny_c].^2 )

    ## Accumulated strain 
    ε1 = ε̇II./Δ.t
    ε_acc = ε1 + ε0

    # Angle of the section: Roscoe angle
    θ = 45. - (materials.ϕ[1] + materials.ψ[1])/4
    θrad = deg2rad(θ)

    # Section initialisation
    lenght = Δ.y*nc.y
    C = zeros(2,Int64(round(lenght/Δ.y)))
    ind′ = zeros(Int,2,Int64(round(lenght/Δ.y)))
    C[1,:] .= L.x*0.5 - L.x*0.5 
    C[2,:]  = LinRange(-lenght*0.5, lenght*0.5,Int64(round(lenght/Δ.y)))

    # Rotation matrix
    Rot = [cos(θrad) -sin(θrad); cos(θrad) sin(θrad)]
    
    # Rotate the section to be normal to shear band angle
    C′ = Rot *C

    # Find indices of the line points
    for i = 1 : Int64(round(lenght/Δ.y))
        ind′[1,i] = Int64(round(C′[1,i]/Δ.x + nc.x*0.5 + 0.5))
        ind′[2,i] = Int64(round(C′[1,i]/Δ.y + nc.y*0.5 + 0.5))
    end

    cross_sec = map(CartesianIndex, ind′[2,:], ind′[1,:])
    ε_prof = ε_acc[cross_sec]

    # # To see the section:
    # z6 = plot!(z3, C′[1,:], C′[2,:], color=:white, linewidth=2, legend=false)
    # -------------------------------------------------------------------

    display(to)
    return ε_prof, C
end

# _____________________________________________________________________
# ---------------------------------------------------------------------
# Main

resolution = [100]
z5 = plot(xlabel="x", ylabel="ε_{II} [x 10⁻⁹]", size = (700,300), title = "Accumulated ε across shear bands" )

for i in eachindex(resolution)

    res = resolution[i]

    (ε_prof, C) = main((x = resolution[i], y = resolution[i]))
    plot!(z5,C[2,:],(ε_prof).*10e9, label="$(res)²")

end

display(z5)

# ---------------------------------------------------------------------
# _____________________________________________________________________