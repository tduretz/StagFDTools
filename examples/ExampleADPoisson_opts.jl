using StagFDTools, ExtendableSparse, StaticArrays, LinearAlgebra, Statistics, UnPack, Plots
using TimerOutputs
# using Enzyme
using ForwardDiff, Enzyme  # AD backends you want to use 

struct NumberingPoisson2{T1,T2,T3,T4}
    num     ::Matrix{T1}
    type    ::Matrix{T2}
    bc_val  ::Matrix{T3}
    pattern ::T4

    function NumberingPoisson2(ni::NTuple, ::Val{N}) where N
        num    = zeros(Int64, (ni.+2)...)
        bc_val = zeros(Float64, (ni.+2)...)
        type   = Matrix{Symbol}(undef, (ni.+2)...)
        pattern = @MMatrix zeros(Int64, N, N)
        new{
            eltype(num),
            eltype(type),
            eltype(bc_val),
            typeof(pattern),
        }(num, type, bc_val, pattern)
    end

    function NumberingPoisson2{N}(ni::NTuple) where N
        num    = zeros(Int64, (ni.+2)...)
        bc_val = zeros(Float64, (ni.+2)...)
        type   = Matrix{Symbol}(undef, (ni.+2)...)
        pattern = @MMatrix zeros(Int64, 3, 3)
        new{
            eltype(num),
            eltype(type),
            eltype(bc_val),
            typeof(pattern),
        }(num, type, bc_val, pattern)    
    end
end    

import StagFDTools: NumberingPoisson!
function NumberingPoisson!(N::NumberingPoisson2, nc)
    neq                     = nc.x * nc.y
    N.num[2:end-1,2:end-1] .= reshape(1:neq, nc.x, nc.y)

    # Make periodic in x
    for j in axes(N.type,2)
        if N.type[1,j] === :periodic
            N.num[1,j] = N.num[end-1,j]
        end
        if N.type[end,j] === :periodic
            N.num[end,j] = N.num[2,j]
        end
    end

    # Make periodic in y
    for i in axes(N.type,1)
        if N.type[i,1] === :periodic
            N.num[i,1] = N.num[i,end-1]
        end
        if N.type[i,end] === :periodic
            N.num[i,end] = N.num[i,2]
        end
    end
end

######

function Poisson2D(u_loc, k, s, type_loc, bcv_loc, Δ)
    
    uC       = u_loc[2,2]

    if type_loc[1,2] === :Dirichlet
        uW = 2*bcv_loc[1,2] - u_loc[2,2]
    elseif type_loc[1,2] === :Neumann
        uW = Δ.x*bcv_loc[1,2] + u_loc[2,2]
    elseif type_loc[1,2] === :periodic || type_loc[1,2] === :in
        uW = u_loc[1,2] 
    end

    if type_loc[3,2] === :Dirichlet
        uE = 2*bcv_loc[3,2] - u_loc[2,2]
    elseif type_loc[3,2] === :Neumann
        uE = -Δ.x*bcv_loc[3,2] + u_loc[2,2]
    elseif type_loc[3,2] === :periodic || type_loc[3,2] === :in
        uE = u_loc[3,2] 
    end

    if type_loc[2,1] === :Dirichlet
        uS = 2*bcv_loc[2,1] - u_loc[2,2]
    elseif type_loc[2,1] === :Neumann
        uS = Δ.y*bcv_loc[2,1] + u_loc[2,2]
    elseif type_loc[2,1] === :periodic || type_loc[2,1] === :in
        uS = u_loc[2,1] 
    end

    if type_loc[2,3] === :Dirichlet
        uN = 2*bcv_loc[2,3] - u_loc[2,2]
    elseif type_loc[2,3] === :Neumann
        uN = -Δ.y*bcv_loc[2,3] + u_loc[2,2]
    elseif type_loc[2,3] === :periodic || type_loc[2,3] === :in
        uN = u_loc[2,3] 
    end

    qxW = -k.xx[1]*(uC - uW)/Δ.x
    qxE = -k.xx[2]*(uE - uC)/Δ.x
    qyS = -k.yy[1]*(uC - uS)/Δ.y
    qyN = -k.yy[2]*(uN - uC)/Δ.y

    return -(-(qxE - qxW)/Δ.x - (qyN - qyS)/Δ.y + s)
end


function ResidualPoisson2D_2!(R, u, k, s, num, nc, Δ)  # u_loc, s, type_loc, Δ

    k_loc_shear = @SVector(zeros(2))
                
    shift    = (x=1, y=1)
    (; type, bc_val) = num
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        u_loc     =      SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(type[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        
        R[i,j]    = Poisson2D(u_loc, k_loc, s[i,j], type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssemblyPoisson_ForwardDiff!(K, u, k, s, numbering, nc, Δ)

    (; bc_val, type, pattern, num) = numbering

    shift    = (x=1, y=1)

    k_loc_shear = @SVector(zeros(2))

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        
        num_loc   = SMatrix{3,3}(num[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern
        u_loc     = SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)
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

function AssemblyPoisson_Enzyme!(K, u, k, s, numbering, nc, Δ)

    (; bc_val, type, pattern, num) = numbering

    ∂R∂u     = @MMatrix zeros(3,3) 
    shift    = (x=1, y=1)

    k_loc_shear = @SVector(zeros(2))

    # to = TimerOutput()
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        
        num_loc   = SMatrix{3,3}(num[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern
        u_loc     = MMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)
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

function RangesPoisson(nc)
    return (inx = 2:nc.x+1, iny = 2:nc.y+1)
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
    Print_xy(numbering.type) 

    # 5-point stencil
    numbering.pattern .= @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
    NumberingPoisson!(numbering, nc)
    # Parameters
    L     = 1.
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
    # Residual check
    @timeit to "Residual" ResidualPoisson2D_2!(r, u, k, s, numbering, nc, Δ) 

    @info norm(r)/sqrt(length(r))
    # Assembly
    ndof     = maximum(numbering.num)
    K        = ExtendableSparseMatrix(ndof, ndof)
    AssemblyPoisson_Enzyme!(K, u, k, s, numbering, nc, Δ) # allocate pattern 
    @timeit to "Assembly Enzyme" begin
        AssemblyPoisson_Enzyme!(K, u, k, s, numbering, nc, Δ)
    end
    # @timeit to "Assembly ForwardDiff" begin
    #     AssemblyPoisson_ForwardDiff!(K, u, k, s, numbering, nc, Δ)
    # end
    @show norm(K-K')
    b  = r[inx,iny][:]
    # Solve
    du           = K\b
    u[inx,iny] .-= reshape(du, nc...)
    # Residual check
    ResidualPoisson2D_2!(r, u, k, s, numbering, nc, Δ) 
    @info norm(r)/sqrt(length(r))
    # Visualization
    # heatmap(xc[inx], yc[iny], u[inx,iny]')
    # qx = -diff(u[inx,iny],dims=1)/Δ.x
    # qy = -diff(u[inx,iny],dims=2)/Δ.y
    # @show     mean(qx[1,:])
    # @show     mean(qx[end,:])
    # @show     mean(qy[:,1])
    # @show     mean(qy[:,end])
    # heatmap(xc[1:end-3], yc[iny], qx')
    # heatmap(xc[inx], yc[1:end-3], qy')

    display(to)

end


# ────────────────────────────────────────────────────────────────────────────
#                                    Time                    Allocations      
#                           ───────────────────────   ────────────────────────
#     Tot / % measured:          676ms /  82.4%            177MiB /  89.2%    

# Section           ncalls     time    %tot     avg     alloc    %tot      avg
# ────────────────────────────────────────────────────────────────────────────
# Assembly Enzyme        1    115μs   81.7%   137μs    388KiB  100.0%   388KiB
# Residual               1   30.8μs   18.3%  30.8μs     32.0B    0.0%    32.0B
# ────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────
#                                         Time                    Allocations      
#                                ───────────────────────   ────────────────────────
#        Tot / % measured:            438ms /  75.0%            147MiB /  87.0%    

# Section                ncalls     time    %tot     avg     alloc    %tot      avg
# ─────────────────────────────────────────────────────────────────────────────────
# Assembly ForwardDiff        1   94.8μs   79.8%   139μs    294KiB  100.0%   294KiB
# Residual                    1   35.2μs   20.2%  35.2μs     32.0B    0.0%    32.0B
# ─────────────────────────────────────────────────────────────────────────────────
