using StagFDTools, ExtendableSparse, StaticArrays, LinearAlgebra, IterativeSolvers, SuiteSparse, Statistics, Plots, SparseArrays
using TimerOutputs
using DifferentiationInterface
import ForwardDiff, Enzyme  # AD backends you want to use 

######

function Poisson2D(u_loc, k, s, type_loc, bcv_loc, Δ)
    
    uC       = u_loc[2,2]
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y

    # Necessary for 5-point stencil
    uW = type_loc[1,2] == :periodic || type_loc[1,2] == :in ? u_loc[1,2] :
         type_loc[1,2] == :Dirichlet ? fma(2, bcv_loc[1,2], -u_loc[2,2]) :
         fma(Δ.x, bcv_loc[1,2], u_loc[2,2])

    uE = type_loc[3,2] == :periodic || type_loc[3,2] == :in ? u_loc[3,2] :
         type_loc[3,2] == :Dirichlet ? fma(2, bcv_loc[3,2], -u_loc[2,2]) :
         fma(-Δ.x, bcv_loc[3,2], u_loc[2,2])

    uS = type_loc[2,1] == :periodic || type_loc[2,1] == :in ? u_loc[2,1] :
         type_loc[2,1] == :Dirichlet ? fma(2, bcv_loc[2,1], -u_loc[2,2]) :
         fma(Δ.y, bcv_loc[2,1], u_loc[2,2])

    uN = type_loc[2,3] == :periodic || type_loc[2,3] == :in ? u_loc[2,3] :
         type_loc[2,3] == :Dirichlet ? fma(2, bcv_loc[2,3], -u_loc[2,2]) :
         fma(-Δ.y, bcv_loc[2,3], u_loc[2,2])
    
    # Necessary for 9-point stencil (Newton or anisotropic)
    uSW = type_loc[1,1] == :periodic || type_loc[1,1] == :in ? u_loc[1,1] :
          type_loc[1,1] == :Dirichlet ? fma(2, bcv_loc[1,1], -u_loc[2,2]) :
          fma(Δ.x, bcv_loc[1,1], u_loc[2,1])

    uSE = type_loc[3,1] == :periodic || type_loc[3,1] == :in ? u_loc[3,1] :
          type_loc[3,1] == :Dirichlet ? fma(2, bcv_loc[3,1], -u_loc[2,2]) :
          fma(-Δ.x, bcv_loc[3,1], u_loc[2,1])

    uNW = type_loc[1,3] == :periodic || type_loc[1,3] == :in ? u_loc[1,3] :
          type_loc[1,3] == :Dirichlet ? fma(2, bcv_loc[1,3], -u_loc[2,2]) :
          fma(Δ.y, bcv_loc[1,3], u_loc[2,3])

    uNE = type_loc[3,3] == :periodic || type_loc[3,3] == :in ? u_loc[3,3] :
          type_loc[3,3] == :Dirichlet ? fma(2, bcv_loc[3,3], -u_loc[2,2]) :
          fma(-Δ.y, bcv_loc[3,3], u_loc[2,3])

    ##########################################################

    # 5-point stencil
    ExW = (uC - uW) * invΔx
    ExE = (uE - uC) * invΔx
    EyS = (uC - uS) * invΔy
    EyN = (uN - uC) * invΔy

    # 9-point stencil 
    ExSW = (uS - uSW) * invΔx
    ExSE = (uSE - uS) * invΔx
    ExNW = (uN - uNW) * invΔx
    ExNE = (uNE - uN) * invΔx

    EySW = (uW - uSW) * invΔy
    EySE = (uE - uSE) * invΔy
    EyNW = (uNW - uW) * invΔy
    EyNE = (uNE - uE) * invΔy

    ##########################################################

    # Symmetric scheme Günter et al. (2005) - https://www.pas.rochester.edu/~shuleli/0629/gunter2007.pdf
    
    # Average gradient vector components to vertices
    ĒxSW = 1/2*(ExW + ExSW)
    ĒySW = 1/2*(EyS + EySW)
    ĒxSE = 1/2*(ExE + ExSE)
    ĒySE = 1/2*(EyS + EySE)
    ĒxNW = 1/2*(ExW + ExNW)
    ĒyNW = 1/2*(EyN + EyNW)
    ĒxNE = 1/2*(ExE + ExNE)
    ĒyNE = 1/2*(EyN + EyNE)

    # Average flux vector components to cell center
    qxW = -1/2*( k.xx[1,1]*ĒxSW + k.xx[1,2]*ĒxNW + k.xy[1,1]*ĒySW + k.xy[1,2]*ĒyNW )
    qxE = -1/2*( k.xx[2,1]*ĒxSE + k.xx[2,2]*ĒxNE + k.xy[2,1]*ĒySE + k.xy[2,2]*ĒyNE )
    qyS = -1/2*( k.xy[1,1]*ĒxSW + k.xy[2,1]*ĒxSE + k.yy[1,1]*ĒySW + k.yy[2,1]*ĒySE )
    qyN = -1/2*( k.xy[1,2]*ĒxNW + k.xy[2,2]*ĒxNE + k.yy[1,2]*ĒyNW + k.yy[2,2]*ĒyNE )

    return -(-(qxE - qxW) * invΔx - (qyN - qyS) * invΔy + s)
end

function ResidualPoisson2D_2!(R, u, k, s, numbering, nc, Δ) 
                
    shift    = (x=1, y=1)
    (; bc_val, type, pattern, num) = numbering
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        u_loc     = SMatrix{3,3}(     u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(  type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = SMatrix{2,2}(  k.xx[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_yy  = SMatrix{2,2}(  k.yy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_xy  = SMatrix{2,2}(  k.xy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc     = (xx = k_loc_xx, xy = k_loc_xy, yy = k_loc_yy)
        
        R[i,j]     = Poisson2D(u_loc, k_loc, s[i,j], type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssemblyPoisson_ForwardDiff(u, k, s, numbering, nc, Δ)
    ndof     = maximum(numbering.num)
    K        = ExtendableSparseMatrix(ndof, ndof)
    AssemblyPoisson_ForwardDiff!(K, u, k, s, numbering, nc, Δ)
    return K
end

function AssemblyPoisson_ForwardDiff!(K, u, k, s, numbering, nc, Δ)

    (; bc_val, type, pattern, num) = numbering

    shift    = (x=1, y=1)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        
        num_loc   = SMatrix{3,3}(num[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern
        u_loc     = SMatrix{3,3}(     u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(  type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = SMatrix{2,2}(  k.xx[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_yy  = SMatrix{2,2}(  k.yy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_xy  = SMatrix{2,2}(  k.xy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc     = (xx = k_loc_xx, xy = k_loc_xy, yy = k_loc_yy)
 
        ∂R∂u = ForwardDiff.gradient(
            x -> Poisson2D(x, k_loc, s[i,j], type_loc, bcv_loc, Δ), 
            u_loc
        )

        num_ij = num[i,j]
        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if num_loc[ii,jj] > 0
                K[num_ij, num_loc[ii,jj]] = ∂R∂u[ii,jj] 
            end
        end
    end
    return nothing
end

function Residual_and_AssemblyPoisson_ForwardDiff(R, u, k, s, numbering, nc, Δ)
    ndof     = maximum(numbering.num)
    K        = ExtendableSparseMatrix(ndof, ndof)
    Residual_and_AssemblyPoisson_ForwardDiff!(R, K, u, k, s, numbering, nc, Δ)
    return K
end

function Residual_and_AssemblyPoisson_ForwardDiff!(R, K, u, k, s, numbering, nc, Δ)

    (; bc_val, type, pattern, num) = numbering

    shift    = (x=1, y=1)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        
        num_loc   = SMatrix{3,3}(num[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern
        u_loc     = SMatrix{3,3}(     u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(  type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = SMatrix{2,2}(  k.xx[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_yy  = SMatrix{2,2}(  k.yy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_xy  = SMatrix{2,2}(  k.xy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc     = (xx = k_loc_xx, xy = k_loc_xy, yy = k_loc_yy)
 
        R[i,j], ∂R∂u = value_and_gradient(
            x -> Poisson2D(x, k_loc, s[i,j], type_loc, bcv_loc, Δ),
            AutoForwardDiff(), 
            u_loc
        )
        
        num_ij = num[i,j]
        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if num_loc[ii,jj] > 0
                K[num_ij, num_loc[ii,jj]] = ∂R∂u[ii,jj] 
            end
        end
    end
    return nothing
end

function AssemblyPoisson_Enzyme(u, k, s, numbering, nc, Δ)
    ndof     = maximum(numbering.num)
    K        = ExtendableSparseMatrix(ndof, ndof)
    AssemblyPoisson_Enzyme!(K, u, k, s, numbering, nc, Δ)
    return K
end

function AssemblyPoisson_Enzyme!(K, u, k, s, numbering, nc, Δ)

    (; bc_val, type, pattern, num) = numbering

    ∂R∂u     = @MMatrix zeros(3,3) 
    shift    = (x=1, y=1)

    # to = TimerOutput()
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        
        num_loc   = SMatrix{3,3}(   num[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern
        u_loc     = SMatrix{3,3}(     u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(  type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = SMatrix{2,2}(  k.xx[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_yy  = SMatrix{2,2}(  k.yy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_xy  = SMatrix{2,2}(  k.xy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc     = (xx = k_loc_xx, xy = k_loc_xy, yy = k_loc_yy)

        ∂R∂u     .= 0e0

        autodiff(Enzyme.Reverse, Poisson2D, Duplicated(u_loc, ∂R∂u), Const(k_loc), Const(s[i,j]), Const(type_loc), Const(bcv_loc), Const(Δ), Const(num_loc))

        num_ij = num[i,j]
        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if num_loc[ii,jj] > 0
                K[num_ij, num_loc[ii,jj]] = ∂R∂u[ii,jj] 
            end
        end
    end
    # display(to)

    return nothing
end


function Residual_and_AssemblyPoisson_Enzyme(R, u, k, s, numbering, nc, Δ)
    ndof     = maximum(numbering.num)
    K        = ExtendableSparseMatrix(ndof, ndof)
    Residual_and_AssemblyPoisson_Enzyme!(R, K, u, k, s, numbering, nc, Δ)
    return K
end

function Residual_and_AssemblyPoisson_Enzyme!(R, K, u, k, s, numbering, nc, Δ)

    (; bc_val, type, pattern, num) = numbering

    shift    = (x=1, y=1)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        
        num_loc   = SMatrix{3,3}( num[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern
        u_loc     = SMatrix{3,3}(     u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(  type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = SMatrix{2,2}(  k.xx[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_yy  = SMatrix{2,2}(  k.yy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc_xy  = SMatrix{2,2}(  k.xy[ii,jj] for ii in i-1:i,   jj in j-1:j  )
        k_loc     = (xx = k_loc_xx, xy = k_loc_xy, yy = k_loc_yy)

        R[i,j], ∂R∂u = value_and_gradient(
            x -> Poisson2D(x, k_loc, s[i,j], type_loc, bcv_loc, Δ),
            AutoEnzyme(), 
            u_loc
        )
        
        num_ij = num[i,j]
        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if num_loc[ii,jj] > 0
                K[num_ij, num_loc[ii,jj]] = ∂R∂u[ii,jj] 
            end
        end
    end
    return nothing
end

let
    to = TimerOutput()
    # Resolution in FD cells
    nc = (x = 1000, y = 1000)

    # Generates an empty numbering structure
    numbering = NumberingPoisson2{3}(values(nc))

    ranges = RangesPoisson(nc)
    (; inx, iny) = ranges
    
    # Define node types and set BC flags
    numbering.type          .= fill(:out, (nc.x+2, nc.y+2))
    numbering.type[inx,iny] .= :in
    numbering.type[1,:]     .= :Dirichlet
    numbering.type[end,:]   .= :Dirichlet
    numbering.type[:,1]     .= :Dirichlet
    numbering.type[:,end]   .= :Dirichlet
    numbering.bc_val[1,:]   .= 1.0 
    numbering.bc_val[end,:] .= 1.0 
    numbering.bc_val[:,1]   .= 1.0
    numbering.bc_val[:,end] .= 1.0
    
    @info "Node types"
    Print_xy(numbering.type) 

    # 5-point stencil
    numbering.pattern .= @SMatrix([1 1 1; 1 1 1; 1 1 1]) 
    NumberingPoisson!(numbering, nc)
    # Parameters
    L     = 1.
    k_iso = 1.0
    δ0    = 1.0
    δ1    = 12.0
    θ     = -45*π/180. 
    # Arrays
    r   = zeros(nc.x+2, nc.y+2)
    s   = zeros(nc.x+2, nc.y+2)
    u   = zeros(nc.x+2, nc.y+2)
    k   = (xx=zeros(nc.x+1,nc.y+1), yy= ones(nc.x+1,nc.y+1) , xy=zeros(nc.x+1,nc.y+1))
    Δ   = (x=L/nc.x, y=L/nc.y)
    xc  = LinRange(-L/2-Δ.x/2, L/2+Δ.x/2, nc.x+2)
    yc  = LinRange(-L/2-Δ.y/2, L/2+Δ.y/2, nc.y+2)
    xv  = LinRange(-L/2, L/2, nc.x+1)
    yv  = LinRange(-L/2, L/2, nc.y+1)
    # Configuration
    s  .= 50*exp.(-(xc.^2 .+ (yc').^2)./0.4^2)
    δ   = (x=δ0*ones(nc.x+1,nc.y), y=δ0*ones(nc.x,nc.y+1))

    δv  = δ0.*ones(nc.x+1, nc.y+1)
    # δv[(0*xv .+ (yv')).<0.1]  .= δ1
    δv[(xv.^2 .+ (yv').^2).<0.1]  .= δ1
    k.xx .= k_iso .* cos(θ) .^ 2 .+ k_iso .* sin(θ) .^ 2 ./ δv
    k.xy .= k_iso .* sin(θ) .* cos(θ) .- k_iso .* sin(θ) .* cos(θ) ./ δv
    k.yy .= k_iso .* sin(θ) .^ 2 .+ k_iso .* cos(θ) .^ 2 ./ δv
    
    # Residual check
    @timeit to "Residual" ResidualPoisson2D_2!(r, u, k, s, numbering, nc, Δ) 

    @info norm(r)/sqrt(length(r))
    
    ndof     = maximum(numbering.num)
    K        = ExtendableSparseMatrix(ndof, ndof)
    # @timeit to "Residual+Assembly FD" begin
    #     Residual_and_AssemblyPoisson_ForwardDiff!(r, K, u, k, s, numbering, nc, Δ)
    # end

    @timeit to "Residual+Assembly Enzyme" begin
        Residual_and_AssemblyPoisson_ForwardDiff!(r, K, u, k, s, numbering, nc, Δ)
    end
    @show norm(K-K')
    b  = r[inx,iny][:]
    # Solve
    @info "ndof = $(length(b))"
    @timeit to "Solver" begin
        du           = K\b
    end
    u[inx,iny] .-= reshape(du, nc...)

    # Residual check
    ResidualPoisson2D_2!(r, u, k, s, numbering, nc, Δ) 
    @info norm(r)/sqrt(length(r))
    # Visualization
    p1 = heatmap(xc[inx], yc[iny], u[inx,iny]', aspect_ratio=1, xlim=extrema(xc))

    display(p1)
    display(to)
end

# Enzyme.jl
# ────────────────────────────────────────────────────────────────────
#                               Time                    Allocations      
#                      ───────────────────────   ────────────────────────
# Tot / % measured:         857ms /  83.4%            884MiB /  71.1%    

# Section   ncalls        time    %tot     avg     alloc    %tot      avg
# ────────────────────────────────────────────────────────────────────
# Solver                    1    658ms   92.1%   658ms    507MiB   80.7%   507MiB
# Residual+Assembly Enzyme  1   56.1ms    7.9%  56.1ms    121MiB   19.3%   121MiB
# ────────────────────────────────────────────────────────────────────

# ForwardDiff.jl
# ────────────────────────────────────────────────────────────────────
#                            Time                    Allocations      
#                   ───────────────────────   ────────────────────────
# Tot / % measured:      860ms /  85.8%            884MiB /  71.1%    

# Section   ncalls     time    %tot     avg     alloc    %tot      avg
# ────────────────────────────────────────────────────────────────────
# Solver                1    676ms   91.5%   676ms    507MiB   80.7%   507MiB
# Residual+Assembly FD  1   62.7ms    8.5%  62.7ms    121MiB   19.3%   121MiB
# ───────────