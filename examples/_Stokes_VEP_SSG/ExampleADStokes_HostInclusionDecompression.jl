using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian
using TimerOutputs

function line(p, K, Δt, η_ve, ψ, p1, t1)
    p2 = p1 + K*Δt*sind(ψ)
    t2 = t1 - η_ve  
    a  = (t2-t1)/(p2-p1)
    b  = t2 - a*p2
    return a*p + b
end

@views function main(nc, radius)
    #--------------------------------------------#

    # Scales
    sc = (σ = 3e10, L = 1e-2, t = 1e10)
    L  = (x=1e-2/sc.L, y=1e-2/sc.L)

    # Boundary loading type
    config = :free_slip
    ε̇kk    = 0.5e-14.*sc.t
    P0     = 1e9/sc.σ
    D_BC   = @SMatrix( [ ε̇kk  0.0;
                         0.0  ε̇kk ])

    # Material parameters
    G0   = 3e10
    K0   = 4*G0

    materials = ( 
        compressible = true,
        plasticity   = :DruckerPrager,
        n    = [1.0    1.0    1.0 ],
        η0   = [1e50   1e50   1e50]./(sc.σ*sc.t), 
        G    = [G0     G0/4   2*G0]./sc.σ,
        C    = [50e6   50e6   50e6]./sc.σ,
        σT   = [50e6   50e6   50e6]./sc.σ, # Kiss2023
        δσT  = [1e6    1e6    1e6 ]./sc.σ, # Kiss2023
        P1   = [0.0    0.0    0.0 ], # Kiss2023
        τ1   = [0.0    0.0    0.0 ], # Kiss2023
        P2   = [0.0    0.0    0.0 ], # Kiss2023
        τ2   = [0.0    0.0    0.0 ], # Kiss2023
        ϕ    = [35.0   35.0  35.0 ],
        ηvp  = [1e19   1e19  1e19 ]./(sc.σ*sc.t),
        β    = [1/K0   1/(K0/4)  1/(2*K0)].*sc.σ,
        ψ    = [5.0    5.0    5.0 ],
        B    = [0.0    0.0    0.0 ],
        cosϕ = [0.0    0.0    0.0 ],
        sinϕ = [0.0    0.0    0.0 ],
        sinψ = [0.0    0.0    0.0 ],
    )
    # For power law
    @. materials.B  = (2*materials.η0)^(-materials.n)

    # For plasticity
    @. materials.cosϕ  = cosd(materials.ϕ)
    @. materials.sinϕ  = sind(materials.ϕ)
    @. materials.sinψ  = sind(materials.ψ)
    
    # For Kiss2023: calculate corner coordinates 
    @. materials.P1 = -(materials.σT - materials.δσT)                                         # p at the intersection of cutoff and Mode-1
    @. materials.τ1 = materials.δσT                                                           # τII at the intersection of cutoff and Mode-1
    @. materials.P2 = -(materials.σT - materials.C*cosd(materials.ϕ))/(1.0-sind(materials.ϕ)) # p at the intersection of Drucker-Prager and Mode-1
    @. materials.τ2 = materials.P2 + materials.σT                                             # τII at the intersection of Drucker-Prager and Mode-1

    # Time steps
    Δt0   = 5e9/sc.t
    nt    = 1#145

    # Newton solver
    niter = 15
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
    # xmin, xmax = -L.x/2, L.x/2
    # ymin, ymax = -L.y/2, L.y/2
    xmin, xmax = -0.0, L.x
    ymin, ymax = -0.0, L.y
    xv = LinRange(xmin,       xmax, nc.x+1)
    yv = LinRange(ymin,       ymax, nc.y+1)
    xc = LinRange(xmin+Δ.x/2, xmax-Δ.x/2, nc.x)
    yc = LinRange(ymin+Δ.y/2, ymax-Δ.y/2, nc.y)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    # Initial velocity & pressure field
    @views V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    @views V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'
    @views Pt[inx_c, iny_c ]  .= P0
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
    a, b = -1., 1.0
    xc2 = xc .+ 0*yc'
    yc2 = 0*xc .+ yc'
    xv2 = xv .+ 0*yv'
    yv2 = 0*xv .+ yv'
    @views @. phases.c[inx_c, iny_c][yc2<0.75 && xc2<0.75 && yc2<(xc2*a + b)] .= 3
    @views @. phases.v[inx_v, iny_v][yv2<0.75 && xv2<0.75 && yv2<(xv2*a + b)] .= 3
    @views @. phases.c[inx_c, iny_c][yc2<radius && xc2<radius] .= 2
    @views @. phases.v[inx_v, iny_v][yv2<radius && xv2<radius] .= 2
    # @views phases.c[inx_c, iny_c][((xc.-(xmax+xmin)/2).^2 .+ ((yc.-(xmax+xmin)/2)').^2) .<= 0.1^2] .= 2
    # @views phases.v[inx_v, iny_v][((xv.-(ymax+ymin)/2).^2 .+ ((yv.-(ymax+ymin)/2)').^2) .<= 0.1^2] .= 2

    p1 = heatmap(xc, yc, phases.c[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc))
    p2 = heatmap(xv, yv, phases.v[inx_v,iny_v]', aspect_ratio=1, xlim=extrema(xc))
    display(plot(p1, p2))

    #--------------------------------------------#

    rvec = zeros(length(α))
    err  = (x = zeros(niter), y = zeros(niter), p = zeros(niter))
    to   = TimerOutput()

    #--------------------------------------------#

    anim = @animate for it=1:nt

        @printf("Step %04d --- mean(Pt) = %1.2f GPa\n", it, mean(Pt).*sc.σ/1e9)
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
        
        p_tr1 = LinRange(-1, 0, 10)
        p_tr2 = LinRange(0, 1, 100)
        p_tr3 = LinRange(-1, 1, 100)

        K      = 1 / materials.β[1]
        η_ve   = materials.G[1] * Δ.t
        pc1    = materials.P1[1]
        pc2    = materials.P2[1]
        τc1    = materials.τ1[1]
        τc2    = materials.τ2[1]
        φ      = materials.ϕ[1]
        C      = materials.C[1]
        ψ      = materials.ψ[1]
        σT     = materials.σT[1]
    
        P_end =  0.05
 
        p3 = plot(aspect_ratio=1, xlabel="P [GPa]", ylabel="τII [GPa]")
        if materials.plasticity === :DruckerPrager
            plot!([0.0, P_end.*sc.σ/1e9],[C*cosd(φ).*sc.σ/1e9, (P_end*sind(φ)+C*cosd(φ)).*sc.σ/1e9], label=:none)
        
            function F_hyperbolic(τ, P, φ, C, σT)    
                return sqrt.( τ.^2 .+ (C*cosd(φ)-σT*sind(φ)).^2 ) .- (C*cosd(φ) .+ P*sind(φ))
            end
    
            P_ax = LinRange(-σT, P_end, 100)
            τ_ax = collect(P_ax*sind(φ) .+ C*cosd(φ))
            dFdτ = zero(τ_ax)
            for iter=1:10
                F_yield = F_hyperbolic(τ_ax, P_ax, φ, C, σT) 
                @show norm(F_yield)

                # forwarddiff_gradients!(F_hyperbolic, Duplicated(τ_ax, dFdτ), Const(P_ax), Const(φ), Const(C), Const(σT) )
                τ_ax .-= F_yield./1
            end

            plot!(P_ax.*sc.σ/1e9, τ_ax.*sc.σ/1e9, c=:black)

        
        elseif materials.plasticity === :tensile
            plot!([-σT.*sc.σ/1e9, P_end.*sc.σ/1e9],[0., (P_end+σT).*sc.σ/1e9], label=:none)
        elseif materials.plasticity === :Kiss2023
            # l1    = line.(p_tr1, K, Δ.t, η_ve, 90., pc1, τc1)
            # l2    = line.(p_tr2, K, Δ.t, η_ve, 90., pc2, τc2)
            # l3    = line.(p_tr3, K, Δ.t, η_ve,   ψ, pc2, τc2)
            # p3 = plot!(p_tr1,  l1, label=:none)
            # p3 = plot!(p_tr2,  l2, label=:none)
            # p3 = plot!(p_tr3,  l3, label=:none)
            p3 = plot!([pc1.*sc.σ/1e9, pc1.*sc.σ/1e9, pc2.*sc.σ/1e9, P_end.*sc.σ/1e9],[0.0, τc1.*sc.σ/1e9, τc2.*sc.σ/1e9, (P_end*sind(φ)+C*cosd(φ)).*sc.σ/1e9], label=:none)
        end
        p3 = scatter!( Pt[inx_c,iny_c][:].*sc.σ/1e9, τII[:].*sc.σ/1e9, label=:none)

        # p1 = heatmap(xv, yc, R.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vx")
        # p2 = heatmap(xc, yc,  Pt[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Pt", c=:coolwarm)
        p2 = heatmap(xc*sc.L*1e2, yc*sc.L*1e2,  log10.(ε̇II./sc.t)', aspect_ratio=1, xlim=extrema(xc*sc.L*1e2), title="log10 ε̇II [1/s]", c=:coolwarm)
        p4 = heatmap(xc*sc.L*1e2, yc.*sc.L*1e2,  τII'.*sc.σ./1e6,   aspect_ratio=1, xlim=extrema(xc*sc.L*1e2), title="τII [MPa]", c=:turbo)
        # p4 = heatmap(xv*sc.L, yv.*sc.L,  τ.xy[inx_v,iny_v]'.*sc.σ, aspect_ratio=1, xlim=extrema(xc*sc.L), title="τ.xy", c=:turbo)
        # p4 = heatmap(xv*sc.L, yv.*sc.L,  η.v[inx_v,iny_v]'.*sc.σ, aspect_ratio=1, xlim=extrema(xc*sc.L), title="τII", c=:turbo)

        # p3 = heatmap(xv, yc, (V.x[inx_Vx,iny_Vx])', aspect_ratio=1, xlim=extrema(xv), title="Vx")
        # p4 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc), title="Vy")
        # p2 = heatmap(xc, yc,  Pt[inx_c,iny_c]', aspect_ratio=1, xlim=extrema(xc), title="Pt")

        p1 = plot(xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error", legend=:topright)
        p1 = scatter!(1:niter, log10.(err.x[1:niter]), label="Vx")
        p1 = scatter!(1:niter, log10.(err.y[1:niter]), label="Vy")
        p1 = scatter!(1:niter, log10.(err.p[1:niter]), label="Pt")
        display(plot(p1, p2, p3, p4, layout=(2,2)))

        @show (3/materials.β[1] - 2*materials.G[1])/(2*(3/materials.β[1] + 2*materials.G[1]))

    end
    gif(anim, "./results/HostInclusion_$(materials.plasticity)_r$(radius).gif", fps = 15)

    display(to)
    
end

let
    # r = [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4]
    r = [0.4 0.45]
    for i in eachindex(r)
        main((x = 100, y = 100), r[i])
    end
end