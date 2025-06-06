using StagFDTools, ExtendableSparse, StaticArrays, LinearAlgebra, IterativeSolvers, SuiteSparse, Statistics, Plots
using TimerOutputs
using DifferentiationInterface
import ForwardDiff, Enzyme  # AD backends you want to use 

######


function Poisson2D(u_loc, k, s, type_loc, bcv_loc, Δ)
    
    uC       = u_loc[2,2]

    # Necessary for 5-point stencil
    if type_loc[1,2] == :Dirichlet
        uW = 2*bcv_loc[1,2] - u_loc[2,2]
    elseif type_loc[1,2] == :Neumann
        uW = Δ.x*bcv_loc[1,2] + u_loc[2,2]
    elseif type_loc[1,2] == :periodic || type_loc[1,2] == :in
        uW = u_loc[1,2] 
    end

    if type_loc[3,2] == :Dirichlet
        uE = 2*bcv_loc[3,2] - u_loc[2,2]
    elseif type_loc[3,2] == :Neumann
        uE = -Δ.x*bcv_loc[3,2] + u_loc[2,2]
    elseif type_loc[3,2] == :periodic || type_loc[3,2] == :in
        uE = u_loc[3,2] 
    end

    if type_loc[2,1] == :Dirichlet
        uS = 2*bcv_loc[2,1] - u_loc[2,2]
    elseif type_loc[2,1] == :Neumann
        uS = Δ.y*bcv_loc[2,1] + u_loc[2,2]
    elseif type_loc[2,1] == :periodic || type_loc[2,1] == :in
        uS = u_loc[2,1] 
    end

    if type_loc[2,3] == :Dirichlet
        uN = 2*bcv_loc[2,3] - u_loc[2,2]
    elseif type_loc[2,3] == :Neumann
        uN = -Δ.y*bcv_loc[2,3] + u_loc[2,2]
    elseif type_loc[2,3] == :periodic || type_loc[2,3] == :in
        uN = u_loc[2,3] 
    end

    # 5-point stencil
    ExW = (uC - uW)/Δ.x
    ExE = (uE - uC)/Δ.x
    EyS = (uC - uS)/Δ.y
    EyN = (uN - uC)/Δ.y

    # Necessary for 9-point stencil (Newton or anisotropic)
    if type_loc[1,1] == :Dirichlet
        uSW = 2*bcv_loc[1,1] - u_loc[2,1]
    elseif type_loc[1,1] == :Neumann
        uSW = Δ.x*bcv_loc[1,1] + u_loc[2,1]
    elseif type_loc[1,1] == :periodic || type_loc[1,1] == :in
        uSW = u_loc[1,1] 
    end

    if type_loc[3,1] == :Dirichlet
        uSE = 2*bcv_loc[3,1] - u_loc[2,1]
    elseif type_loc[3,1] == :Neumann
        uSE = -Δ.x*bcv_loc[3,1] + u_loc[2,1]
    elseif type_loc[3,1] == :periodic || type_loc[3,1] == :in
        uSE = u_loc[3,1] 
    end

    if type_loc[1,3] == :Dirichlet
        uNW = 2*bcv_loc[1,3] - u_loc[2,3]
    elseif type_loc[1,3] == :Neumann
        uNW = Δ.y*bcv_loc[1,3] + u_loc[2,3]
    elseif type_loc[1,3] == :periodic || type_loc[1,3] == :in
        uNW = u_loc[1,3] 
    end

    if type_loc[3,3] == :Dirichlet
        uNE = 2*bcv_loc[3,3] - u_loc[2,3]
    elseif type_loc[3,3] == :Neumann
        uNE = -Δ.y*bcv_loc[3,3] + u_loc[2,3]
    elseif type_loc[3,3] == :periodic || type_loc[3,3] == :in
        uNE = u_loc[3,3] 
    end

    ExSW = (uS - uSW)/Δ.x
    ExSE = (uSE - uS)/Δ.x
    ExNW = (uN - uNW)/Δ.x
    ExNE = (uNE - uN)/Δ.x

    # Necessary for 9-point stencil (Newton or anisotropic)
    if type_loc[1,1] == :Dirichlet
        uSW = 2*bcv_loc[1,1] - u_loc[1,2]
    elseif type_loc[1,1] == :Neumann
        uSW = Δ.x*bcv_loc[1,1] + u_loc[1,2]
    elseif type_loc[1,1] == :periodic || type_loc[1,1] == :in
        uSW = u_loc[1,1] 
    end

    if type_loc[3,1] == :Dirichlet
        uSE = 2*bcv_loc[3,1] - u_loc[3,2]
    elseif type_loc[3,1] == :Neumann
        uSE = -Δ.x*bcv_loc[3,1] + u_loc[3,2]
    elseif type_loc[3,1] == :periodic || type_loc[3,1] == :in
        uSE = u_loc[3,1] 
    end

    if type_loc[1,3] == :Dirichlet
        uNW = 2*bcv_loc[1,3] - u_loc[1,2]
    elseif type_loc[1,3] == :Neumann
        uNW = Δ.y*bcv_loc[1,3] + u_loc[1,2]
    elseif type_loc[1,3] == :periodic || type_loc[1,3] == :in
        uNW = u_loc[1,3] 
    end

    if type_loc[3,3] == :Dirichlet
        uNE = 2*bcv_loc[3,3] - u_loc[3,2]
    elseif type_loc[3,3] == :Neumann
        uNE = -Δ.y*bcv_loc[3,3] + u_loc[3,2]
    elseif type_loc[3,3] == :periodic || type_loc[3,3] == :in
        uNE = u_loc[3,3] 
    end

    EySW = (uW - uSW)/Δ.y
    EySE = (uE - uSE)/Δ.y
    EyNW = (uNW - uW)/Δ.y
    EyNE = (uNE - uE)/Δ.y

    # Missing ones
    ĒyW  = 0.25*(EySW + EyNW + EyS + EyN)
    ĒyE  = 0.25*(EySE + EyNE + EyS + EyN)
    ĒxS  = 0.25*(ExSW + ExSE + ExW + ExE)
    ĒxN  = 0.25*(ExNW + ExNE + ExW + ExE)

    # Flux
    qxW = - ( k.xx[1]*ExW + k.xy[1]*ĒyW ) 
    qxE = - ( k.xx[2]*ExE + k.xy[2]*ĒyE )
    qyS = - ( k.yy[1]*EyS + k.yx[1]*ĒxS )
    qyN = - ( k.yy[2]*EyN + k.yx[2]*ĒxN )

    return -(-(qxE - qxW)/Δ.x - (qyN - qyS)/Δ.y + s)
end

function ResidualPoisson2D_2!(R, u, k, s, num, nc, Δ)  # u_loc, s, type_loc, Δ
                
    shift    = (x=1, y=1)
    (; type, bc_val) = num
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        u_loc     =      SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_xy  = @SVector [k.x.xy[i-1,j-1], k.x.xy[i,j-1]]
        k_loc_yx  = @SVector [k.y.yx[i-1,j-1], k.y.yx[i-1,j]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx, xy = k_loc_xy,
                     yx = k_loc_yx, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        
        R[i,j]    = Poisson2D(u_loc, k_loc, s[i,j], type_loc, bcv_loc, Δ)
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
        u_loc     = SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_xy  = @SVector [k.x.xy[i-1,j-1], k.x.xy[i,j-1]]
        k_loc_yx  = @SVector [k.y.yx[i-1,j-1], k.y.yx[i-1,j]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_xy,
                     yx = k_loc_yx, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
 
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
        u_loc     = SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_xy  = @SVector [k.x.xy[i-1,j-1], k.x.xy[i,j-1]]
        k_loc_yx  = @SVector [k.y.yx[i-1,j-1], k.y.yx[i-1,j]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_xy,
                     yx = k_loc_yx, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
 
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
        
        num_loc   = SMatrix{3,3}(num[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern
        u_loc     = MMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_xy  = @SVector [k.x.xy[i-1,j-1], k.x.xy[i,j-1]]
        k_loc_yx  = @SVector [k.y.yx[i-1,j-1], k.y.yx[i-1,j]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_xy,
                     yx = k_loc_yx, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        ∂R∂u     .= 0e0

        autodiff(Enzyme.Reverse, Poisson2D, Duplicated(u_loc, ∂R∂u), Const(k_loc), Const(s[i,j]), Const(type_loc), Const(bcv_loc), Const(Δ))

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
        
        num_loc   = SMatrix{3,3}(num[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern
        u_loc     = SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_xy  = @SVector [k.x.xy[i-1,j-1], k.x.xy[i,j-1]]
        k_loc_yx  = @SVector [k.y.yx[i-1,j-1], k.y.yx[i-1,j]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_xy,
                     yx = k_loc_yx, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
 
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
    nc = (x = 30, y = 40)

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
    # numbering.bc_val         = zeros(nc.x+2, nc.y+2)
    numbering.bc_val[1,:]   .= 1.0 
    numbering.bc_val[end,:] .= 1.0 
    numbering.bc_val[:,1]   .= 1.0
    numbering.bc_val[:,end] .= 1.0
    
    @info "Node types"
    printxy(numbering.type) 

    # 5-point stencil
    numbering.pattern .= @SMatrix([1 1 1; 1 1 1; 1 1 1]) 
    NumberingPoisson!(numbering, nc)
    # Parameters
    L     = 1.
    k_iso = 1.0
    δ     = 5.0
    θ     = -45*π/180. 
    # Arrays
    r   = zeros(nc.x+2, nc.y+2)
    s   = zeros(nc.x+2, nc.y+2)
    u   = zeros(nc.x+2, nc.y+2)
    k   = (x = (xx= ones(nc.x+1,nc.y), xy=zeros(nc.x+1,nc.y)), 
           y = (yx=zeros(nc.x,nc.y+1), yy= ones(nc.x,nc.y+1)))
    Δ   = (x=L/nc.x, y=L/nc.y)
    xc  = LinRange(-L/2-Δ.x/2, L/2+Δ.x/2, nc.x+2)
    yc  = LinRange(-L/2-Δ.y/2, L/2+Δ.y/2, nc.y+2)
    # Configuration
    s  .= 50*exp.(-(xc.^2 .+ (yc').^2)./0.4^2)
    k.x.xx .= k_iso .* cos(θ) .^ 2 + k_iso .* sin(θ) .^ 2 ./ δ
    k.x.xy .= k_iso .* sin(θ) .* cos(θ) - k_iso .* sin(θ) .* cos(θ) ./ δ
    k.y.yx .= k_iso .* sin(θ) .* cos(θ) - k_iso .* sin(θ) .* cos(θ) ./ δ
    k.y.yy .= k_iso .* sin(θ) .^ 2 + k_iso .* cos(θ) .^ 2 ./ δ
    # Residual check
    # @timeit to "Residual" ResidualPoisson2D_2!(r, u, k, s, numbering, nc, Δ) 

    # @info norm(r)/sqrt(length(r))
    
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


    # u  .= 0.
    # R = zero(b)
    # U = zero(b)
    # dU = zero(b)

    # @timeit to "Cholesky PC Iterative solve" begin
    #     PC  = cholesky(1/2 .*(K + K'))
    #     for it = 1:10
    #         R .= K*U - b
    #         println("$(it) r = ", norm(R)/sqrt(length(r)))
    #         # ldiv!(dU, PC, R) 
    #         U .-= PC\R
    #     end
    #     # gmres!(du, K, b; Pl=cholesky(1/2 .*(K + K')))
    # end
    # Residual check
    ResidualPoisson2D_2!(r, u, k, s, numbering, nc, Δ) 
    @info norm(r)/sqrt(length(r))
    # Visualization
    p1 = heatmap(xc[inx], yc[iny], u[inx,iny]', aspect_ratio=1, xlim=extrema(xc))
    # qx = -diff(u[inx,iny],dims=1)/Δ.x
    # qy = -diff(u[inx,iny],dims=2)/Δ.y
    # @show     mean(qx[1,:])
    # @show     mean(qx[end,:])
    # @show     mean(qy[:,1])
    # @show     mean(qy[:,end])
    # heatmap(xc[1:end-3], yc[iny], qx')
    # heatmap(xc[inx], yc[1:end-3], qy')
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