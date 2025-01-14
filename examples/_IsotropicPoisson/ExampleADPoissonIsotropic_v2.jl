using StagFDTools, StagFDTools.Poisson, ExtendableSparse, StaticArrays, LinearAlgebra, Statistics, UnPack, Plots
using TimerOutputs
# using Enzyme
using ForwardDiff, Enzyme  # AD backends you want to use 

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


function ResidualPoisson2D!(R, u, k, s, num, type, bc_val, nc, Δ)  # u_loc, s, type_loc, Δ

    k_loc_shear = @SVector(zeros(2))
                
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        u_loc     =  SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(  type.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        
        R[i,j]    = Poisson2D(u_loc, k_loc, s[i,j], type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssemblyPoisson_ForwardDiff!(K, u, k, s, number, type, pattern, bc_val, nc, Δ)

    shift    = (x=1, y=1)

    k_loc_shear = @SVector(zeros(2))

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        
        num_loc   = SMatrix{3,3}(number.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern.u.u
        u_loc     = SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(  type.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
 
        ∂R∂u = ForwardDiff.gradient(
            x -> Poisson2D(x, k_loc, s[i,j], type_loc, bcv_loc, Δ), 
            u_loc
        )

        num_ij = number.u[i,j]
        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if num_loc[ii,jj] > 0
                K.u.u[num_ij, num_loc[ii,jj]] = ∂R∂u[ii,jj] 
            end
        end
    end
    return nothing
end

function AssemblyPoisson_Enzyme!(K, u, k, s, number, type, pattern, bc_val, nc, Δ)

    ∂R∂u     = @MMatrix zeros(3,3) 
    shift    = (x=1, y=1)

    k_loc_shear = @SVector(zeros(2))

    # to = TimerOutput()
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        
        num_loc   = SMatrix{3,3}(number.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern.u.u
        u_loc     = MMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(  type.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        ∂R∂u     .= 0e0

        autodiff(Enzyme.Reverse, Poisson2D, Duplicated(u_loc, ∂R∂u), Const(k_loc), Const(s[i,j]), Const(type_loc), Const(bcv_loc), Const(Δ))

        num_ij = number.u[i,j]
        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if num_loc[ii,jj] > 0
                K.u.u[num_ij, num_loc[ii,jj]] = ∂R∂u[ii,jj] 
            end
        end
    end
    # display(to)

    return nothing
end

let
    to = TimerOutput()

    # Resolution in FD cells
    nc = (x = 30, y = 40)

    # Get ranges
    ranges = Ranges(nc)
    (; inx, iny) = ranges

    # Define node types and set BC flags
    type = Fields( fill(:out, (nc.x+2, nc.y+2)) )
    type.u[2:end-1,2:end-1].= :in
    type.u[1,:]            .= :Dirichlet 
    type.u[end,:]          .= :Dirichlet 
    type.u[:,1]            .= :Dirichlet
    type.u[:,end]          .= :Dirichlet
    bc_val = Fields( fill(0., (nc.x+2, nc.y+2)) )
    bc_val.u        .= zeros(nc.x+2, nc.y+2)
    bc_val.u[1,:]   .= 1.0 
    bc_val.u[end,:] .= 1.0 
    bc_val.u[:,1]   .= 1.0
    bc_val.u[:,end] .= 1.0

    # 5-point stencil
    pattern = Fields( Fields( @SMatrix([0 1 0; 1 1 1; 0 1 0]) ) )

    # Equation number
    number = Fields( fill(0, (nc.x+2, nc.y+2)) )
    Numbering!(number, type, nc)
    
    @info "Node types"
    printxy(number.u) 

    # 5-point stencil
    pattern = Fields( Fields( @SMatrix([0 1 0; 1 1 1; 0 1 0]) ) )

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
    @timeit to "Residual" ResidualPoisson2D!(r, u, k, s, number, type, bc_val, nc, Δ) 
    @info norm(r)/sqrt(length(r))
    
    # Sparse matrix assembly
    nu  = maximum(number.u)
    M   = Fields( Fields( ExtendableSparseMatrix(nu, nu) ))
  
    @timeit to "Assembly Enzyme" begin
        AssemblyPoisson_Enzyme!(M, u, k, s, number, type, pattern, bc_val, nc, Δ)
    end
    @timeit to "Assembly ForwardDiff" begin
        AssemblyPoisson_ForwardDiff!(M, u, k, s, number, type, pattern, bc_val, nc, Δ)
    end

    @info "Symmetry"
    @show norm(M.u.u - M.u.u')
    b  = r[inx,iny][:]
    # Solve
    du           = M.u.u\b
    u[inx,iny] .-= reshape(du, nc...)
    # Residual check
    ResidualPoisson2D!(r, u, k, s, number, type, bc_val, nc, Δ)     # @info norm(r)/sqrt(length(r))
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
