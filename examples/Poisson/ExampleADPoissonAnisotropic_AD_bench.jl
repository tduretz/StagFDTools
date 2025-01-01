using StagFDTools, ExtendableSparse, StaticArrays, LinearAlgebra, Statistics
using TimerOutputs
using DifferentiationInterface
import ForwardDiff, Enzyme, ReverseDiff, Zygote, FastDifferentiation  # AD backends you want to use 

# const fns = AutoEnzyme, AutoForwardDiff,  AutoReverseDiff, AutoZygote, AutoFastDifferentiation
const fns = AutoForwardDiff,  AutoReverseDiff, AutoZygote, AutoFastDifferentiation

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

function Residual_and_AssemblyPoisson(R, u, k, s, numbering, nc, Δ, fn::F) where F
    ndof     = maximum(numbering.num)
    K        = ExtendableSparseMatrix(ndof, ndof)
    Residual_and_AssemblyPoisson!(R, K, u, k, s, numbering, nc, Δ, fn)
    return K
end

function Residual_and_AssemblyPoisson!(R, K, u, k, s, numbering, nc, Δ, fn::F) where F

    (; bc_val, type, pattern, num) = numbering

    shift    = (x=1, y=1)

    to = TimerOutput()

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
 
        @timeit to "$fn" begin
            R[i,j], ∂R∂u = value_and_gradient(
                x -> Poisson2D(x, k_loc, s[i,j], type_loc, bcv_loc, Δ),
                fn(), 
                u_loc
            )
        end
        
        num_ij = num[i,j]
        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if num_loc[ii,jj] > 0
                K[num_ij, num_loc[ii,jj]] = ∂R∂u[ii,jj] 
            end
        end
    end
    display(to)
    return nothing
end

let
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
    
    ndof     = maximum(numbering.num)
    K        = ExtendableSparseMatrix(ndof, ndof)

    for fn in fns
        println("Running with $fn")
        Residual_and_AssemblyPoisson!(r, K, u, k, s, numbering, nc, Δ, fn)
    end

end

# ───────────────────────────────────────────────────────────────────────
# AutoForwardDiff         1.20k   136μs  100.0%   113ns     0.00B     - %    0.00B
# ───────────────────────────────────────────────────────────────────────
# AutoReverseDiff         1.20k  20.7ms  100.0%  17.3μs   16.0MiB  100.0%  13.6KiB
# ───────────────────────────────────────────────────────────────────────
# AutoZygote              1.20k   724ms  100.0%   603μs    100MiB  100.0%  85.3KiB
# ───────────────────────────────────────────────────────────────────────────────
# AutoFastDifferentiation 1.20k   2.06s  100.0%  1.72ms   2.20GiB  100.0%  1.88MiB
# ───────────────────────────────────────────────────────────────────────────────