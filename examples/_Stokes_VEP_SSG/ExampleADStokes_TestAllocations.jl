using StagFDTools, StagFDTools.Stokes, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use
using TimerOutputs

using ProfileCanvas, BenchmarkTools

function PowerLaw(ε̇, materials, phases, Δ)
    ε̇II  = sqrt.(1/2*(ε̇[1].^2 .+ ε̇[2].^2) + ε̇[3].^2)
    P    = ε̇[4]
    n    = materials.n[phases]
    η0   = materials.η0[phases]
    G    = materials.G[phases]
    C    = materials.C[phases]
    ϕ    = materials.ϕ[phases]
    ηvp  = materials.ηvp[phases]
    ψ    = materials.ψ[phases]    
    β    = materials.β[phases]
    η    =  (η0 .* ε̇II.^(1 ./ n .- 1.0 ))[1]
    ηvep = inv(1/η + 1/(G*Δ.t))

    τII  = 2*ηvep*ε̇II
    λ̇    = 0.0
    F    = τII - C*cosd(ϕ) - P*sind(ϕ )- λ̇*ηvp

    if F > 1e-10
        λ̇    = F / (ηvep + ηvp + Δ.t / β * sind(ϕ) * sind(ψ)) 
        τII -= λ̇ * ηvep
        P   += λ̇  * sind(ψ) * Δ.t / β
        # τII = C*cosd(ϕ) + P*sind(ϕ) + ηvp*λ̇
        ηvep = τII/(2*ε̇II)
        F    = τII - C*cosd(ϕ) - P*sind(ϕ )- λ̇*ηvp
        (F>1e-10) && error("Failed return mapping")
        (τII<0.0) && error("Plasticity without condom")
    end

    return ηvep, λ̇, P
end

function Rheology!(ε̇, materials, phases, Δ) 
    η, λ̇, P = PowerLaw(ε̇, materials, phases, Δ)
    τ       = @SVector([2 * η * ε̇[1],
                        2 * η * ε̇[2],
                        2 * η * ε̇[3],
                                  P])
    return τ, η, λ̇
end

function TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, Pt, Ptc, type, BC, materials, phases, Δ)

    _ones = @SVector ones(4)

    # Loop over centroids
    for j=1:size(ε̇.xx,2)-0, i=1:size(ε̇.xx,1)-0
        Vx     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1,   jj in j:j+2)
        Vy     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2,   jj in j:j+1)
        bcx    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        bcy    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
        typex  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
        typey  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
        τxy0   = SMatrix{2,2}(    τ0.xy[ii,jj] for ii in i:i+1,   jj in j:j+1)

        Vx = SetBCVx1(Vx, typex, bcx, Δ)
        Vy = SetBCVy1(Vy, typey, bcy, Δ)

        Dxx = ∂x_inn(Vx) / Δ.x 
        Dyy = ∂y_inn(Vy) / Δ.y 
        Dxy = ∂y(Vx) / Δ.y
        Dyx = ∂x(Vy) / Δ.x
        
        Dkk = Dxx .+ Dyy
        ε̇xx = @. Dxx - Dkk ./ 3
        ε̇yy = @. Dyy - Dkk ./ 3
        ε̇xy = @. (Dxy + Dyx) ./ 2
        ε̇̄xy = av(ε̇xy)
       
        # Visco-elasticity
        G     = materials.G[phases.c[i,j]]
        τ̄xy0  = av(τxy0)
        ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t), Pt[i,j]])
        
        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, Rheology!, ε̇vec, Const(materials), Const(phases.c[i,j]), Const(Δ))
        
        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        @views 𝐷_ctl.c[i,j][:,1] .= jac.derivs[1][1][1]
        @views 𝐷_ctl.c[i,j][:,2] .= jac.derivs[1][2][1]
        @views 𝐷_ctl.c[i,j][:,3] .= jac.derivs[1][3][1]
        @views 𝐷_ctl.c[i,j][:,4] .= jac.derivs[1][4][1]

        # Tangent operator used for Picard Linearisation
        𝐷.c[i,j] .= diagm(2*jac.val[2] * _ones)
        𝐷.c[i,j][4,4] = 1

        # Update stress
        τ.xx[i,j] = jac.val[1][1]
        τ.yy[i,j] = jac.val[1][2]
        ε̇.xx[i,j] = ε̇xx[1]
        ε̇.yy[i,j] = ε̇yy[1]
        λ̇.c[i,j]  = jac.val[3]
        η.c[i,j]  = jac.val[2]
        Ptc[i,j]  = jac.val[1][4]
    end

    # Loop over vertices
    for j=1:size(ε̇.xy,2)-2, i=1:size(ε̇.xy,1)-2
        Vx     = SMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        Vy     = SMatrix{2,3}(      V.y[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        bcx    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        bcy    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        typex  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        typey  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        τxx0   = SMatrix{2,2}(    τ0.xx[ii,jj] for ii in i:i+1,   jj in j:j+1)
        τyy0   = SMatrix{2,2}(    τ0.yy[ii,jj] for ii in i:i+1,   jj in j:j+1)
        P      = SMatrix{2,2}(       Pt[ii,jj] for ii in i:i+1,   jj in j:j+1)

        Vx     = SetBCVx1(Vx, typex, bcx, Δ)
        Vy     = SetBCVy1(Vy, typey, bcy, Δ)
    
        Dxx    = ∂x(Vx) / Δ.x
        Dyy    = ∂y(Vy) / Δ.y
        Dxy    = ∂y_inn(Vx) / Δ.y
        Dyx    = ∂x_inn(Vy) / Δ.x

        Dkk   = @. Dxx + Dyy
        ε̇xx   = @. Dxx - Dkk / 3
        ε̇yy   = @. Dyy - Dkk / 3
        ε̇xy   = @. (Dxy + Dyx) /2
        ε̇̄xx   = av(ε̇xx)
        ε̇̄yy   = av(ε̇yy)
        
        # Visco-elasticity
        G     = materials.G[phases.v[i,j]]
        τ̄xx0  = av(τxx0)
        τ̄yy0  = av(τyy0)
        P̄     = av(   P)
        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t), ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t), P̄[1]])
        
        # Tangent operator used for Newton Linearisation
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, Rheology!, ε̇vec, Const(materials), Const(phases.v[i+1,j+1]), Const(Δ))

        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        @views 𝐷_ctl.v[i+1,j+1][:,1] .= jac.derivs[1][1][1]
        @views 𝐷_ctl.v[i+1,j+1][:,2] .= jac.derivs[1][2][1]
        @views 𝐷_ctl.v[i+1,j+1][:,3] .= jac.derivs[1][3][1]
        @views 𝐷_ctl.v[i+1,j+1][:,4] .= jac.derivs[1][4][1]

        # Tangent operator used for Picard Linearisation
        𝐷.v[i+1,j+1] .= diagm(2*jac.val[2] * _ones)
        𝐷.v[i+1,j+1][4,4] = 1

        # Update stress
        τ.xy[i+1,j+1] = jac.val[1][3]
        ε̇.xy[i+1,j+1] = ε̇xy[1]
        λ̇.v[i+1,j+1]  = jac.val[3]
        η.v[i+1,j+1]  = jac.val[2]
    end
end

@views function main(nc)
    #--------------------------------------------#

    # Resolution

    # Boundary loading type
    config = :free_slip
    D_BC   = @SMatrix( [ -1. 0.;
                          0  1 ])

    # Material parameters
    materials = ( 
        n   = [1.0  1.0],
        η0  = [1e2  1e-1], 
        G   = [1e1  1e1],
        C   = [150  150],
        ϕ   = [30.  30.],
        ηvp = [0.5  0.5],
        β   = [1e-2 1e-2],
        ψ   = [3    3],
    )

    # Time steps
    Δt0   = 0.5
    nt    = 1

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
    dx   = zeros(nVx + nVy + nPt)

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
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...) )
    Pt      = zeros(size_c...)
    Pti     = zeros(size_c...)
    Pt0     = zeros(size_c...)
    Ptc     = zeros(size_c...)
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

    @info "Benchmark AssembleMomentum2D_x!"
    ProfileCanvas.@profview AssembleMomentum2D_x!(M, V, Pt, Pt0, λ̇, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
    display( @benchmark AssembleMomentum2D_x!($(M, V, Pt, Pt0, λ̇, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)...) )

    @info "Benchmark AssembleMomentum2D_y!"
    ProfileCanvas.@profview AssembleMomentum2D_y!(M, V, Pt, Pt0, λ̇, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
    display( @benchmark AssembleMomentum2D_y!($(M, V, Pt, Pt0, λ̇, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)...) )
   
    @info "Benchmark AssembleContinuity2D!"
    ProfileCanvas.@profview AssembleContinuity2D!(M, V, Pt, Pt0, λ̇, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
    display( @benchmark AssembleMomentum2D_x!($(M, V, Pt, Pt0, λ̇, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)...) )
    
end

let
    main((x = 100, y = 100))
end

