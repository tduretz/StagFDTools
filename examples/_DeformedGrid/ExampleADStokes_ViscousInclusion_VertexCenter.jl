using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian
using TimerOutputs
using ExactFieldSolutions
import CairoMakie as cm
import CairoMakie.Makie.GeometryBasics as geom

include("Stokes_Deformed_VertexCenter.jl")

function TransformCoordinates(ξ, params)
    h = params.Amp*exp(-(ξ[1] - params.x0)^2 / params.σx^2)
    if params.deform 
        X = @SVector([ξ[1], (ξ[2]/params.ymin0)*(params.m-h)+h])
    else
        X = @SVector([ξ[1], ξ[2]])
    end
end

@views function main(nc, BC_template, D_template)
    #--------------------------------------------#

    # Boundary loading type
    config = BC_template
    D_BC   = D_template
    old_stencil = true 

    params_inc = (mm = 1.0, mc = 1e4, rc = 2.0, gr = 0.0, er = D_BC[1,1])

    # Material parameters
    materials = ( 
        compressible = false,
        plasticity   = :none,
        g    = [0.0    0.0  ],
        ρ    = [1.0    1.0  ],
        n    = [1.0    1.0  ],
        η0   = [params_inc.mm    params_inc.mc], 
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
        Fields(@SMatrix([1 1 1; 1 1 1]),                        @SMatrix([1 1; 1 1; 1 1]),                      @SMatrix([1]))
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

    # Allocations
    R       = (x  = zeros(size_x...), y  = zeros(size_y...), p  = zeros(size_c...))
    V       = (x  = zeros(size_x...), y  = zeros(size_y...))
    Vi      = (x  = zeros(size_x...), y  = zeros(size_y...))
    η       = (c  =  ones(size_c...), v  =  ones(size_v...), Vx = zeros(size_x...), Vy = zeros(size_y...))
    λ̇       = (c  = zeros(size_c...), v  = zeros(size_v...), Vx = zeros(size_x...), Vy = zeros(size_y...))
    ε̇       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), Vx=[@MVector(zeros(3)) for _ in axes(V.x,1), _ in axes(V.x,2)], Vy=[@MVector(zeros(3)) for _ in axes(V.y,1), _ in axes(V.y,2)] )
    τ0      = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), Vx=[@MVector(zeros(3)) for _ in axes(V.x,1), _ in axes(V.x,2)], Vy=[@MVector(zeros(3)) for _ in axes(V.y,1), _ in axes(V.y,2)] )
    τ       = (xx = zeros(size_c...), yy = zeros(size_c...), xy = zeros(size_v...), Vx=[@MVector(zeros(3)) for _ in axes(V.x,1), _ in axes(V.x,2)], Vy=[@MVector(zeros(3)) for _ in axes(V.y,1), _ in axes(V.y,2)] )
    Pt      = (c=zeros(size_c...), Vx = zeros(size_x...), Vy = zeros(size_y...))
    Pti     = zeros(size_c...)
    Pt0     = zeros(size_c...)
    ΔPt     = (c=zeros(size_c...), Vx = zeros(size_x...), Vy = zeros(size_y...))
    D_Vx    = [@MMatrix(zeros(4,4)) for _ in axes(V.x,1), _ in axes(V.x,2)]
    D_Vy    = [@MMatrix(zeros(4,4)) for _ in axes(V.y,1), _ in axes(V.y,2)]

    Dc      = [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    Dv      = [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷       = (c=Dc, v=Dv, Vx=D_Vx, Vy=D_Vy)

    D_ctl_Vx= [@MMatrix(zeros(4,4)) for _ in axes(V.x,1), _ in axes(V.x,2)]
    D_ctl_Vy= [@MMatrix(zeros(4,4)) for _ in axes(V.y,1), _ in axes(V.y,2)]
    D_ctl_c = [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xx,1), _ in axes(ε̇.xx,2)]
    D_ctl_v = [@MMatrix(zeros(4,4)) for _ in axes(ε̇.xy,1), _ in axes(ε̇.xy,2)]
    𝐷_ctl   = (c = D_ctl_c, v = D_ctl_v, Vx=D_ctl_Vx, Vy=D_ctl_Vy)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...), Vx= ones(Int64, size_x...), Vy= ones(Int64, size_y...))  # phase on velocity points

    # Reference domain
    𝜉  = (min=-5.0, max=5.0)
    𝜂  = (min=-10.0, max=0.0)
    Δ  = (ξ=(𝜉.max-𝜉.min)/nc.x, η=(𝜂.max-𝜂.min)/nc.y,           x=(𝜉.max-𝜉.min)/nc.x, y=(𝜂.max-𝜂.min)/nc.y, t = 1.0)
    ξv = LinRange(𝜉.min-Δ.ξ,   𝜉.max+Δ.ξ,   nc.x+3)
    ηv = LinRange(𝜂.min-Δ.η,   𝜂.max+Δ.η,   nc.y+3)
    ξc = LinRange(𝜉.min-Δ.ξ/2, 𝜉.max+Δ.ξ/2, nc.x+2)
    ηc = LinRange(𝜂.min-Δ.η/2, 𝜂.max+Δ.η/2, nc.y+2)
    ξVy= LinRange(𝜉.min-3*Δ.ξ/2, 𝜉.max+3*Δ.ξ/2, nc.x+4)
    ηVx= LinRange(𝜂.min-3*Δ.η/2, 𝜂.max+3*Δ.η/2,   nc.y+4)

    # Reference coordinates ξ
    ξ = (
        v =  [@MVector(zeros(2)) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c =  [@MVector(zeros(2)) for _ in axes(ξc,1), _ in axes(ηc,1)],
        Vx =  [@MVector(zeros(2)) for _ in axes(ξv,1), _ in axes(ηVx,1)],
        Vy =  [@MVector(zeros(2)) for _ in axes(ξVy,1), _ in axes(ηv,1)],
    )
    for I in CartesianIndices(ξ.v)
        i, j = I[1], I[2]
        ξ.v[I] .= @SVector([ξv[i], ηv[j]]) 
    end
    for I in CartesianIndices(ξ.c)
        i, j = I[1], I[2]
        ξ.c[I] .= @SVector([ξc[i], ηc[j]]) 
    end
    for I in CartesianIndices(ξ.Vx)
        i, j = I[1], I[2]
        ξ.Vx[I] .= @SVector([ξv[i], ηVx[j]]) 
    end
    for I in CartesianIndices(ξ.Vy)
        i, j = I[1], I[2]
        ξ.Vy[I] .= @SVector([ξVy[i], ηv[j]]) 
    end
    
    # Physical coordinates X 
    X = (
        v  =  [@MVector(zeros(2)) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c  =  [@MVector(zeros(2)) for _ in axes(ξc,1), _ in axes(ηc,1)],
        Vx =  [@MVector(zeros(2)) for _ in axes(ξv,1), _ in axes(ηVx,1)],
        Vy =  [@MVector(zeros(2)) for _ in axes(ξVy,1), _ in axes(ηv,1)],
    )

    # Mesh deformation parameters
    params = (
        deform = false,
        m      = -1,
        Amp    = 0.25,
        σx     = 0.1,
        ymin0  = -1,
        ymax0  = 0.5,
        y0     = 0.5,
        x0     = 0.0,
    )
   
    # Deform mesh and determine the inverse Jacobian  
    Jinv = (
        v  =  [@MMatrix(zeros(2,2)) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c  =  [@MMatrix(zeros(2,2)) for _ in axes(ξc,1), _ in axes(ηc,1)],
        Vx =  [@MMatrix(zeros(2,2)) for _ in axes(ξv,1), _ in axes(ηVx,1)],
        Vy =  [@MMatrix(zeros(2,2)) for _ in axes(ξVy,1), _ in axes(ηv,1)],
    )

    Iinv = (
        v  =  [@MMatrix([1.0 0.0; 0.0 1.0]) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c  =  [@MMatrix([1.0 0.0; 0.0 1.0]) for _ in axes(ξc,1), _ in axes(ηc,1)],
        Vx =  [@MMatrix([1.0 0.0; 0.0 1.0]) for _ in axes(ξv,1), _ in axes(ηVx,1)],
        Vy =  [@MMatrix([1.0 0.0; 0.0 1.0]) for _ in axes(ξVy,1), _ in axes(ηv,1)],
    )
    
    I2  = LinearAlgebra.I(2)     # Identity matrix

    for I in CartesianIndices(X.v)
        J          = forwarddiff_jacobian(TransformCoordinates, ξ.v[I], Const(params))
        Jinv.v[I] .= J.derivs[1] \ I2
        X.v[I]    .= J.val
    end

    for I in CartesianIndices(X.c)
        J          = forwarddiff_jacobian(TransformCoordinates, ξ.c[I], Const(params))
        Jinv.c[I] .= J.derivs[1] \ I2
        X.c[I]    .= J.val
    end

    for I in CartesianIndices(X.Vx)
        J          = forwarddiff_jacobian(TransformCoordinates, ξ.Vx[I], Const(params))
        Jinv.Vx[I] .= J.derivs[1] \ I2
        X.Vx[I]    .= J.val
    end

    for I in CartesianIndices(X.Vy)
        J          = forwarddiff_jacobian(TransformCoordinates, ξ.Vy[I], Const(params))
        Jinv.Vy[I] .= J.derivs[1] \ I2
        X.Vy[I]    .= J.val
    end

    Xv, Yv = zeros(nc.x+1, nc.y+1), zeros(nc.x+1, nc.y+1)
    Xc, Yc = zeros(nc.x+0, nc.y+0), zeros(nc.x+0, nc.y+0)

    for I in CartesianIndices(ξ.v)
        i, j = I[1], I[2]
        if i<=nc.x+1 && j<=nc.y+1
            Xv[i,j] = X.v[i+1,j+1][1]
            Yv[i,j] = X.v[i+1,j+1][2]
        end
    end
    for I in CartesianIndices(ξ.c)
        i, j = I[1], I[2]
        if i<=nc.x+0 && j<=nc.y+0
            Xc[i,j] = X.c[i+1,j+1][1]
            Yc[i,j] = X.c[i+1,j+1][2]
        end
    end

    # 2D coordinate arrays
    Xvx = 0.5.*(Xv[:,1:end-1] .+ Xv[:,2:end])
    Yvx = 0.5.*(Yv[:,1:end-1] .+ Yv[:,2:end])
    Xvy = 0.5.*(Xv[1:end-1,:] .+ Xv[2:end,:])
    Yvy = 0.5.*(Yv[1:end-1,:] .+ Yv[2:end,:])

    # Initial velocity & pressure field
    V.x[inx_Vx,iny_Vx]  .= D_BC[1,1].*Xvx .+ D_BC[1,2].*Yvx 
    V.y[inx_Vy,iny_Vy]  .= D_BC[2,1].*Xvy .+ D_BC[2,2].*Yvy
    Pt.c[inx_c, iny_c ] .= 10.                 
    UpdateSolution!(V, Pt.c, dx, number, type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal)  .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1].*Xvx[:,  1] .+ D_BC[1,2].*Yvx[:,  1])
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1].*Xvx[:,end] .+ D_BC[1,2].*Yvx[:,end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal)  .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal)  .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1].*Xvy[  1,:] .+ D_BC[2,2].*Yvy[  1,:])
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1].*Yvy[end,:] .+ D_BC[2,2].*Yvy[end,:])

    # Set material geometry 
    phases.c[  inx_c, iny_c][(Xc.^2  .+  (Yc .+ 5.0).^2) .<= params_inc.rc^2] .= 2
    phases.v[  inx_v, iny_v][(Xv.^2  .+  (Yv .+ 5.0).^2) .<= params_inc.rc^2] .= 2
    phases.Vx[inx_Vx,iny_Vx][(Xvx.^2 .+ (Yvx .+ 5.0).^2) .<= params_inc.rc^2] .= 2
    phases.Vy[inx_Vy,iny_Vy][(Xvy.^2 .+ (Yvy .+ 5.0).^2) .<= params_inc.rc^2] .= 2

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
        Pt0   .= Pt.c

        for iter=1:niter

            @printf("Iteration %04d\n", iter)
            
            #--------------------------------------------#
            # Residual check        
            @timeit to "Residual" begin

                if !old_stencil
                    TangentOperator_Def!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V,   Pt, ΔPt, Jinv, type, BC, materials, phases, Δ)
                    ResidualContinuity2D_Def!(R, V, Pt.c, Pt0, ΔPt, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ) 
                    # ResidualMomentum2D_x_Def!(R, V, Pt.c, Pt0, ΔPt, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ)
                    # ResidualMomentum2D_y_Def!(R, V, Pt.c, Pt0, ΔPt, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ)
                    ResidualMomentum2D_x!(R, V, Pt.c, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                    ResidualMomentum2D_y!(R, V, Pt.c, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                else
                    TangentOperator!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V,   Pt.c, ΔPt, type, BC, materials, phases, Δ)
                    ResidualContinuity2D!(R, V, Pt.c, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                    ResidualMomentum2D_x!(R, V, Pt.c, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                    ResidualMomentum2D_y!(R, V, Pt.c, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                end
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
                if !old_stencil
                    AssembleContinuity2D_Def!(M, V, Pt.c, Pt0, ΔPt, τ0, 𝐷_ctl, Jinv, phases, materials, number, pattern, type, BC, nc, Δ)
                    # AssembleMomentum2D_x_Def!(M, V, Pt.c, Pt0, ΔPt, τ0, 𝐷_ctl, Jinv, phases, materials, number, pattern, type, BC, nc, Δ)
                    # AssembleMomentum2D_y_Def!(M, V, Pt.c, Pt0, ΔPt, τ0, 𝐷_ctl, Jinv, phases, materials, number, pattern, type, BC, nc, Δ)
                    AssembleMomentum2D_x!(M, V, Pt.c, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                    AssembleMomentum2D_y!(M, V, Pt.c, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                else
                    AssembleContinuity2D!(M, V, Pt.c, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                    AssembleMomentum2D_x!(M, V, Pt.c, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                    AssembleMomentum2D_y!(M, V, Pt.c, Pt0, ΔPt, τ0, 𝐷_ctl, phases, materials, number, pattern, type, BC, nc, Δ)
                end
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
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:lu,  ηb=1e5, niter_l=10, ϵ_l=1e-9)
            dx[1:size(𝐊,1)]     .= u
            dx[size(𝐊,1)+1:end] .= p

            #--------------------------------------------#
            # Line search & solution update
            if !old_stencil
                @timeit to "Line search" imin = LineSearch_Def!(rvec, α, dx, R, V, Pt.c, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, 𝐷, 𝐷_ctl, Jinv, number, type, BC, materials, phases, nc, Δ)
            else
                @timeit to "Line search" imin = LineSearch!(rvec, α, dx, R, V, Pt.c, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇, η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
            end
            UpdateSolution!(V, Pt.c, α[imin]*dx, number, type, nc)
            TangentOperator_Def!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V,   Pt, ΔPt, Jinv, type, BC, materials, phases, Δ)
        end

        # Update pressure
        Pt.c .+= ΔPt.c 

        #--------------------------------------------#

        Pt.c .= Pt.c .- mean(Pt.c)

        Pt_ana = zero(Pt.c)

        for I in CartesianIndices(Pt_ana)
            # coordinate transform
            sol = Stokes2D_Schmid2003( X.c[I].+[0,5]; params=params_inc )
            Pt_ana[I] = sol.p
        end

        Pt_err = abs.(Pt_ana .- Pt.c)

        # Post-process
        cells = (
            x = zeros((nc.x+2)*(nc.y+2), 4),
            y = zeros((nc.x+2)*(nc.y+2), 4)
        )
        for I in CartesianIndices(X.c)
            i, j = I[1], I[2]
            c = i + (j-1)*(nc.x+2)
            cells.x[c,:] .= @SVector([X.v[i,j][1], X.v[i+1,j][1], X.v[i+1,j+1][1], X.v[i,j+1][1] ]) 
            cells.y[c,:] .= @SVector([X.v[i,j][2], X.v[i+1,j][2], X.v[i+1,j+1][2], X.v[i,j+1][2] ]) 
        end

        pc = [cm.Polygon( geom.Point2f[ (cells.x[i,j], cells.y[i,j]) for j=1:4] ) for i in 1:(nc.x+2)*(nc.y+2)]
        
        # Visu
        res = 800
        fig = cm.Figure(size = (res, res), fontsize=25)
        # ----
        ax  = cm.Axis(fig[1, 1], title = "p - numerics", xlabel = "x", ylabel = "y", aspect=1.0)
        cm.poly!(ax, pc, color = Pt.c[:], colormap = :vik, strokewidth = 0, strokecolormap = :white, colorrange=extrema(Pt_ana))#, colorrange=limits
        
        # cm.poly!(ax, pc, color = η.c[:], colormap = :vik, strokewidth = 0, strokecolormap = :white, colorrange=extrema(Pt[2:end-1,2:end-1]))#, colorrange=limits
        # cm.poly!(ax, pc, color = 1/2*(η.Vx[1:end-1,2:end-1].+η.Vx[2:end-0,2:end-1])[:], colormap = :vik, strokewidth = 0, strokecolormap = :white, colorrange=extrema(Pt.c[2:end-1,2:end-1]))#, colorrange=limits
        cm.Colorbar(fig[1, 2], colormap = :vik, flipaxis = true, size = 10, colorrange=extrema(Pt_ana) )    

        # ----
        ax  = cm.Axis(fig[2, 1], title = "p - analytics", xlabel = "x", ylabel = "y", aspect=1.0)
        cm.poly!(ax, pc, color = Pt_ana[:], colormap = :vik, strokewidth = 0, strokecolormap = :white, colorrange=extrema(Pt_ana))
        cm.Colorbar(fig[2, 2], colormap = :vik, flipaxis = true, size = 10, colorrange=extrema(Pt_ana[2:end-1,2:end-1]) )    
       
        # ----
        ax  = cm.Axis(fig[3, 1], title = "p - error", xlabel = "x", ylabel = "y", aspect=1.0)
        cm.poly!(ax, pc, color = Pt_err[:], colormap = :vik, strokewidth = 0, strokecolormap = :white, colorrange=extrema(Pt_err[2:end-1,2:end-1]))
        cm.Colorbar(fig[3, 2], colormap = :vik, flipaxis = true, size = 10, colorrange=extrema(Pt_err[2:end-1,2:end-1]) )    
        
        display(fig)
    end

    display(err)
    display(to)
end


let

    # Resolution
    nc = (x = 151, y = 151)

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
        main(nc, BCs[iBC], D_BCs[iBC])
    end
end