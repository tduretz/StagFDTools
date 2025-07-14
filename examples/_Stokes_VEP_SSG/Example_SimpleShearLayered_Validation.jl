using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use
using TimerOutputs, Interpolations, GridGeometryUtils
import CairoMakie as cm

function Analytical(θ, η, δ, D_BC)
    #= define velocity gradient components and resulting deviatoric strain rate components
    pure shear   ε̇ = [ε̇xx  0 ;  0  -ε̇xx]
    simple shear ε̇ = [ 0  ε̇xy; ε̇xy   0 ] =#
    Dxx = D_BC[1,1]
    Dyy = - Dxx
    Dxy = D_BC[1,2]
    Dkk = Dxx + Dyy

    ε̇	= @SVector([Dxx - Dkk/3, Dyy - Dkk/3, Dxy])

    # Normal vector of anisotropic direction
    n1 = -cos(θ)
    n2 = sin(θ)

    # compute isotropic and layered components for 𝐷
    Δ0 = 2 * n1^2 * n2^2
    Δ1 = n1 * n2^3 - n2 * n1^3
    Δ = @SMatrix([ Δ0 -Δ0 2*Δ1; -Δ0 Δ0 -2*Δ1; Δ1 -Δ1 1-2*Δ0])
    A = @SMatrix([ 1 0 0; 0 1 0; 0 0 1] )

    # compute 𝐷
    𝐷 = 2 * η * A - 2 * (η - η/δ) * Δ

    τ = 𝐷 * ε̇

    τ_II = sqrt(0.5 * (τ[1]^2 + τ[2]^2 + (-τ[1] - τ[2])^2) + τ[3]^2)
    return τ_II
end

function StressVectorCartesian!(ε̇, η_n, θ, δ)
    # Transformation from cartesian to material coordinates
    Q         = @SMatrix([cos(θ) sin(θ); -sin(θ) cos(θ)])
    ε̇_tensor  = @SMatrix([ε̇[1] ε̇[3]; ε̇[3] ε̇[2]])
    ε̇_mat     = Q * ε̇_tensor * Q'

    # calculate stress in material coordinates
    τ_mat_vec = @SVector([2 * η_n   * ε̇_mat[1,1],
                          2 * η_n   * ε̇_mat[2,2],
                          2 * η_n/δ * ε̇_mat[1,2]])

    # convert stress to cartesian coordinates
    τ_mat   = @SMatrix([τ_mat_vec[1] τ_mat_vec[3]; τ_mat_vec[3] τ_mat_vec[2]])
    τ_cart  = Q' * τ_mat * Q
    τ_cart_vec = @SVector([τ_cart[1,1], τ_cart[2,2], τ_cart[1,2]])
    return τ_cart_vec
end


function ViscousRheology(θ, η_n, δ, D_BC)
    #= define velocity gradient components and resulting deviatoric strain rate components
    pure shear ε̇ = [ε̇xx 0; 0 -ε̇xx]
    simple shear ε̇ = [0 ε̇xy; ε̇xy 0]
    =#
    # pureshear = 1 # = 0 for simple shear
    # Dxx = pureshear * 1
    # Dyy = -Dxx
    # Dxy = (1-pureshear) * 1.0
    Dxx = D_BC[1,1]
    Dyy = - Dxx
    Dxy = D_BC[1,2]
    Dkk = Dxx + Dyy

    ε̇	= @SVector([Dxx - Dkk/3, Dyy - Dkk/3, Dxy])

    D_clt = zeros(3,3)

    jac = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVectorCartesian!, ε̇, Const(η_n), Const(θ), Const(δ))

    D_clt[:,:] .= jac.derivs[1]

    τxx  = jac.val[1]
    τyy  = jac.val[2]
    τxy  = jac.val[3]

    τ_II = sqrt(0.5 * (τxx^2 + τyy^2 + (-τxx - τyy)^2) + τxy^2)
    return τ_II
end

@views function main(nc, layering, BC_template, D_template, factorization, η1 , η2)
    #--------------------------------------------#   

    # Boundary loading type
    config = BC_template
    D_BC   = D_template

    # Material parameters
    materials = ( 
        compressible = false,
        plasticity   = :none,
        n    = [1.0    1.0  1.0  ],
        η0   = [η1     η2   1e-1 ], 
        G    = [1e6    1e6  1e6  ],
        C    = [150    150  150  ],
        ϕ    = [30.    30.  30.  ],
        ηvp  = [0.5    0.5  0.5  ],
        β    = [1e-6   1e-6 1e-6 ],
        ψ    = [3.0    3.0  3.0  ],
        B    = [0.     0.   0.   ],
        cosϕ = [0.0    0.0  0.0  ],
        sinϕ = [0.0    0.0  0.0  ],
        sinψ = [0.0    0.0  0.0  ],
    )
    materials.B   .= (2*materials.η0).^(-materials.n)

    # Time steps
    Δt0   = 0.5
    nt    = 1

    # Newton solver
    niter = 3
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
    τII     = ones(size_c...)
    ε̇II     = ones(size_c...)

    # Mesh coordinates
    xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc  = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc  = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    xce = LinRange(-L.x/2-Δ.x/2, L.x/2+Δ.x/2, nc.x+2)
    yce = LinRange(-L.y/2-Δ.y/2, L.y/2+Δ.y/2, nc.y+2)
    phases = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

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
    for i in inx_c, j in iny_c   # loop on centroids
        𝐱 = @SVector([xc[i-1], yc[j-1]])
        isin = inside(𝐱, layering)
        if isin 
            phases.c[i, j] = 2
        end 
    end

    for i in inx_v, j in iny_v  # loop on vertices
        𝐱 = @SVector([xv[i-1], yv[j-1]])
        isin = inside(𝐱, layering)
        if isin 
            phases.v[i, j] = 2
        end  
    end

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
                TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, type, BC, materials, phases, Δ)
                ResidualContinuity2D!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                ResidualMomentum2D_x!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                ResidualMomentum2D_y!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
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
                AssembleContinuity2D!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_x!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                AssembleMomentum2D_y!(M, V, Pt, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
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
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=factorization,  ηb=1e3, niter_l=10, ϵ_l=1e-9)
            dx[1:size(𝐊,1)]     .= u
            dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            UpdateSolution!(V, Pt, α[imin]*dx, number, type, nc)
            TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, ΔPt, type, BC, materials, phases, Δ)

        end

        # Update pressure
        Pt .+= ΔPt 

        #--------------------------------------------#

        # Principal stress
        σ1 = (x = zeros(size(Pt)), y = zeros(size(Pt)), v = zeros(size(Pt)))

        τxyc = av2D(τ.xy)
        ε̇xyc = av2D(ε̇.xy)
        τII[inx_c,iny_c]  .= sqrt.( 0.5.*(τ.xx[inx_c,iny_c].^2 + τ.yy[inx_c,iny_c].^2 + 0*(-τ.xx[inx_c,iny_c]-τ.yy[inx_c,iny_c]).^2) .+ τxyc[inx_c,iny_c].^2 )
        ε̇II[inx_c,iny_c]  .= sqrt.( 0.5.*(ε̇.xx[inx_c,iny_c].^2 + ε̇.yy[inx_c,iny_c].^2 + 0*(-ε̇.xx[inx_c,iny_c]-ε̇.yy[inx_c,iny_c]).^2) .+ ε̇xyc[inx_c,iny_c].^2 )

        for i in inx_c, j in iny_c
            σ         = @SMatrix[-Pt[i,j]+τ.xx[i,j] τxyc[i,j] 0.; τxyc[i,j] -Pt[i,j]+τ.yy[i,j] 0.; 0. 0. -Pt[i,j]+(-τ.xx[i,j]-τ.yy[i,j])]
            v         = eigvecs(σ)
            σp        = eigvals(σ)
            scale     = sqrt(v[1,1]^2 + v[2,1]^2)
            σ1.x[i,j] = v[1,1]/scale
            σ1.y[i,j] = v[2,1]/scale
            σ1.v[i]   = σp[1]
        end

        fig = cm.Figure()
        ax  = cm.Axis(fig[1,1], aspect=cm.DataAspect())
        cm.heatmap!(ax, xc, yc,  τII[inx_c,iny_c], colormap=:bluesreds)
        st = 10
        cm.arrows2d!(ax, xc[1:st:end], yc[1:st:end], σ1.x[inx_c,iny_c][1:st:end,1:st:end], σ1.y[inx_c,iny_c][1:st:end,1:st:end], tiplength = 0, lengthscale=0.02, tipwidth=1, color=:white)
        display(fig)
    end

    # display(to)

    # Only account for the subdomain
    imin_x = argmin(abs.(xce .+ 0.3))
    imax_x = argmin(abs.(xce .- 0.3))
    imin_y = argmin(abs.(yce .+ 0.3))
    imax_y = argmin(abs.(yce .- 0.3))
    inner_x = imin_x:imax_x
    inner_y = imin_y:imax_y

    return mean(τII[inner_x, inner_y])

end

let
    # Boundary condition templates
    BCs = [
        # :EW_periodic,
        # :all_Dirichlet,
        :free_slip,
    ]

    # Boundary deformation gradient matrix
    D_BCs = [
        #  @SMatrix( [0 1; 0  0] ),
         @SMatrix( [1 0; 0 -1] ),
    ]

    nc = (x = 200, y = 200)

    # Discretise angle of layer 
    nθ         = 30
    θ          = LinRange(0, π, nθ) 
    τ_cart     = zeros(nθ)
    τ_cart_lay = zeros(nθ)
    τ_cart_ani = zeros(nθ)
    τ_cart_ana = zeros(nθ) 
    

    #  Anisotropy parameters
    η2 = 2.0
    m  = 10
    η1 = η2 / m

    α1 = 0.5
    α2 = 1 - α1 

    ηn = α1 * η1 + α2 * η2
    δ  = (α1 + α2 * m) * (α1 + α2 / m)

    # Run them all
    for iθ in eachindex(θ)

        layering = Layering(
            (0*0.25, 0.025), 
            0.1, 
            α2; 
            θ = θ[iθ],  
            perturb_amp=0*1.0, 
            perturb_width=1.0
        )

        τ_cart_lay[iθ] = main( nc, layering, BCs[1], D_BCs[1], :chol, η1 ,η2)
        τ_cart_ani[iθ] = ViscousRheology(θ[iθ], ηn, δ, D_BCs[1])
        τ_cart_ana[iθ] = Analytical(θ[iθ], ηn, δ, D_BCs[1])

    end

    ε̇bg = sqrt( sum(1/2 .* D_BCs[1][:].^2))

    # Strongest end-member
    ηeff = α1*η1 + α2*η2
    @show τstrong    = 2*ηeff*ε̇bg

    # Weakest end-member
    ηeff = (α1/η1 + α2/η2)^(-1)
    @show τweak      = 2*ηeff*ε̇bg

    τ_cart .= τstrong * sqrt.(((δ^2 - 1) * cos.(2 .* θ).^2 .+ 1) / (δ^2))

    fig = cm.Figure()
    ax  = cm.Axis(fig[1,1], xlabel= "θ [°]", ylabel="τII [-]")
    cm.lines!(ax, θ*180/π, τ_cart_lay, label="layered")
    cm.lines!(ax, θ*180/π, τ_cart_ani, label="transformation")
    cm.scatter!(ax, θ*180/π, τ_cart, label="expression", markersize=15)
    cm.scatter!(ax, θ*180/π, τ_cart_ana, label="analytical", marker=:utriangle, markersize=15)
    cm.lines!(ax, θ*180/π, τstrong*ones(size(θ)), color=:gray, linestyle=:dash)
    cm.lines!(ax, θ*180/π, τweak*ones(size(θ)), color=:gray, linestyle=:dash)
    cm.Legend(fig[2,1], ax, framevisible=false, "τII (δ ≈ $(round(δ, digits=1)))", orientation=:horizontal)
    display(fig)

end