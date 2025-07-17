using StagFDTools, StagFDTools.Stokes, StagFDTools.Rheology, ExtendableSparse, StaticArrays, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use
using TimerOutputs
import CairoMakie as cm
import CairoMakie.Makie.GeometryBasics as geom

function TransformCoordinates(ξ, params)
    h = params.Amp*exp(-(ξ[1] - params.x0)^2 / params.σx^2)
    if params.deform 
        X = @SVector([ξ[1], (ξ[2]/params.ymin0)*(params.m-h)+h])
    else
        X = @SVector([ξ[1], ξ[2]])
    end
end

function Continuity_Def(Vx_loc, Vy_loc, Pt, Pt0, D, J, phase, materials, type_loc, bcv_loc, Δ)
    invΔx = 1 / Δ.ξ
    invΔy = 1 / Δ.η
    invΔt = 1 / Δ.t
    # BC
    Vx    = SetBCVx1(Vx_loc, type_loc.x, bcv_loc.x, Δ)
    Vy    = SetBCVy1(Vy_loc, type_loc.y, bcv_loc.y, Δ)
    V̄x    = av(Vx)
    V̄y    = av(Vy)
    β     = materials.β[phase]
    η     = materials.β[phase]
    comp  = materials.compressible
    ∂Vx∂x = (Vx[2,2] - Vx[1,2]) * invΔx
    ∂Vy∂y = (Vy[2,2] - Vy[2,1]) * invΔy
    f     =  (∂Vx∂x + ∂Vy∂y) + comp * β * (Pt[1] - Pt0) * invΔt #+ 1/(1000*η)*Pt[1]
    f    *= max(invΔx, invΔy)
    return f
end

function ResidualContinuity2D_Def!(R, V, P, P0, ΔP, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ) 
                
    for j in 2:size(R.p,2)-1, i in 2:size(R.p,1)-1
        if type.Pt[i,j] !== :constant 
            Vx_loc     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
            Vy_loc     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
            bcx_loc    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1, jj in j:j+2)
            bcy_loc    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
            typex_loc  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1, jj in j:j+2)
            typey_loc  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2, jj in j:j+1)
            Jinv_c     = SMatrix{1,1}(  Jinv.c[ii,jj] for ii in i:i,   jj in j:j  )
            D          = (;)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            R.p[i,j]   = Continuity_Def(Vx_loc, Vy_loc, P[i,j], P0[i,j], D, Jinv_c, phases.c[i,j], materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

@views function main(BC_template, D_template)
    #--------------------------------------------#

    # Resolution
    nc = (x = 150, y = 150)

    # Boundary loading type
    config = BC_template
    D_BC   = D_template

    # Material parameters
    materials = ( 
        compressible = false,
        plasticity   = :none,
        n    = [1.0    1.0  ],
        η0   = [1e2    1e-1 ], 
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
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    # Reference domain
    𝜉  = (min=-1/2, max=1/2)
    𝜂  = (min=-1.0, max=0.0)
    Δ  = (ξ=(𝜉.max-𝜉.min)/nc.x, η=(𝜂.max-𝜂.min)/nc.y,           x=(𝜉.max-𝜉.min)/nc.x, y=(𝜂.max-𝜂.min)/nc.y, t = 1.0)
    ξv = LinRange(𝜉.min-Δ.ξ,   𝜉.max+Δ.ξ,   nc.x+3)
    ηv = LinRange(𝜂.min-Δ.η,   𝜂.max+Δ.η,   nc.y+3)
    ξc = LinRange(𝜉.min-Δ.ξ/2, 𝜉.max+Δ.ξ/2, nc.x+2)
    ηc = LinRange(𝜂.min-Δ.η/2, 𝜂.max+Δ.η/2, nc.y+2)

    # Reference coordinates ξ
    ξ = (
        v =  [@MVector(zeros(2)) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c =  [@MVector(zeros(2)) for _ in axes(ξc,1), _ in axes(ηc,1)],
    )
    for I in CartesianIndices(ξ.v)
        i, j = I[1], I[2]
        ξ.v[I] .= @SVector([ξv[i], ηv[j]]) 
    end
    for I in CartesianIndices(ξ.c)
        i, j = I[1], I[2]
        ξ.c[I] .= @SVector([ξc[i], ηc[j]]) 
    end

    # Physical coordinates X 
    X = (
        v =  [@MVector(zeros(2)) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c =  [@MVector(zeros(2)) for _ in axes(ξc,1), _ in axes(ηc,1)],
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
        v =  [@MMatrix(zeros(2,2)) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c =  [@MMatrix(zeros(2,2)) for _ in axes(ξc,1), _ in axes(ηc,1)],
    )

    Iinv = (
        v =  [@MMatrix([1.0 0.0; 0.0 1.0]) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c =  [@MMatrix([1.0 0.0; 0.0 1.0]) for _ in axes(ξc,1), _ in axes(ηc,1)],
    )
    
    I2  = LinearAlgebra.I(2)     # Identity matrix

    for I in CartesianIndices(X.v)
        J          = Enzyme.jacobian(Enzyme.ForwardWithPrimal, TransformCoordinates, ξ.v[I], Const(params))
        Jinv.v[I] .= J.derivs[1] \ I2
        X.v[I]    .= J.val
    end

    for I in CartesianIndices(X.c)
        J          = Enzyme.jacobian(Enzyme.ForwardWithPrimal, TransformCoordinates, ξ.c[I], Const(params))
        Jinv.c[I] .= J.derivs[1] \ I2
        X.c[I]    .= J.val
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
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1].*Xvx .+ D_BC[1,2].*Yvx 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1].*Xvy .+ D_BC[2,2].*Yvy
    Pt[inx_c, iny_c ]  .= 10.                 
    UpdateSolution!(V, Pt, dx, number, type, nc)

    # Boundary condition values
    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_normal) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet_tangent) .* (D_BC[1,1].*Xvx[:,  1] .+ D_BC[1,2].*Yvx[:,  1])
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann_tangent) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet_tangent) .* (D_BC[1,1].*Xvx[:,end] .+ D_BC[1,2].*Yvx[:,end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     1 ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy,   end ] .== :Neumann_normal) .* D_BC[2,2]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1].*Xvy[  1,:] .+ D_BC[2,2].*Yvy[  1,:])
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann_tangent) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet_tangent) .* (D_BC[2,1].*Yvy[end,:] .+ D_BC[2,2].*Yvy[end,:])

    # Set material geometry 
    phases.c[inx_c, iny_c][(Xc.^2 .+ (Yc .+ 0.5).^2) .<= 0.1^2] .= 2
    phases.v[inx_v, iny_v][(Xv.^2 .+ (Yv .+ 0.5).^2) .<= 0.1^2] .= 2

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
                ResidualContinuity2D_Def!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, Jinv, phases, materials, number, type, BC, nc, Δ) 
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
            u, p = DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
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
        ax  = cm.Axis(fig[1, 1], title = "p - centroids", xlabel = "x", ylabel = "y", aspect=1.0)
        cm.poly!(ax, pc, color = Pt[:], colormap = :turbo, strokewidth = 0, strokecolormap = :white, colorrange=extrema(Pt[2:end-1,2:end-1]))#, colorrange=limits
        cm.Colorbar(fig[1, 2], colormap = :turbo, flipaxis = true, size = 10 )    
        display(fig)

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