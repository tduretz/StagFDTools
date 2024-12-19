using StagFDTools, ExtendableSparse, StaticArrays, Enzyme, LinearAlgebra, Statistics, UnPack, Plots
 
function Poisson2D(u_loc, s, type_loc, bcv_loc, Δ)
    
    uC       = u_loc[2,2]

    if type_loc[1,2] == :Dirichlet
        uW = 2*bcv_loc[1,2] - u_loc[2,2]
    elseif type_loc[1,2] == :Neumann
        uW = -Δ.x*bcv_loc[1,2] + u_loc[2,2]
    elseif type_loc[1,2] == :periodic || type_loc[1,2] == :in
        uW = u_loc[1,2] 
    end

    if type_loc[3,2] == :Dirichlet
        uE = 2*bcv_loc[3,2] - u_loc[2,2]
    elseif type_loc[3,2] == :Neumann
        uE = Δ.x*bcv_loc[3,2] + u_loc[2,2]
    elseif type_loc[3,2] == :periodic || type_loc[3,2] == :in
        uE = u_loc[3,2] 
    end

    if type_loc[2,1] == :Dirichlet
        uS = 2*bcv_loc[2,1] - u_loc[2,2]
    elseif type_loc[2,1] == :Neumann
        uS = -Δ.y*bcv_loc[2,1] + u_loc[2,2]
    elseif type_loc[2,1] == :periodic || type_loc[2,1] == :in
        uS = u_loc[2,1] 
    end

    if type_loc[2,3] == :Dirichlet
        uN = 2*bcv_loc[2,3] - u_loc[2,2]
    elseif type_loc[2,3] == :Neumann
        uN = Δ.y*bcv_loc[2,3] + u_loc[2,2]
    elseif type_loc[2,3] == :periodic || type_loc[2,3] == :in
        uN = u_loc[2,3] 
    end

    qxW = -(uC - uW)/Δ.x
    qxE = -(uE - uC)/Δ.x
    qyS = -(uC - uS)/Δ.y
    qyN = -(uN - uC)/Δ.y

    return -(-(qxE - qxW)/Δ.x - (qyN - qyS)/Δ.y + s)
end

@views function ResidualPoisson2D!(R, u, s, num, nc, Δ)  # u_loc, s, type_loc, Δ
    u_loc    = zeros(3,3)
    bcv_loc  = zeros(3,3)
    type_loc = fill(:out,(3,3))
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        u_loc    .=          u[i-1:i+1,j-1:j+1] 
        bcv_loc  .= num.bc_val[i-1:i+1,j-1:j+1]
        type_loc .=   num.type[i-1:i+1,j-1:j+1]
        R[i,j] = Poisson2D(u_loc, s[i,j], type_loc, bcv_loc, Δ)
    end
end

@views function AssemblyPoisson(u, s, u_dir, num, nc, Δ)

    ∂R∂u     = zeros(3,3) 
    u_loc    = zeros(3,3)
    bcv_loc  = zeros(3,3)
    type_loc = fill(:out,(3,3))
    num_loc  = zeros(Int64,3,3)
    ndof     = maximum(num.num)
    K        = ExtendableSparseMatrix(ndof, ndof)
    shift    = (x=1, y=1)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        num_loc  .=    num.num[i-1:i+1,j-1:j+1] .* num.pattern
        u_loc    .=          u[i-1:i+1,j-1:j+1]
        bcv_loc  .= num.bc_val[i-1:i+1,j-1:j+1] 
        type_loc .=   num.type[i-1:i+1,j-1:j+1]
        ∂R∂u     .= 0.
        autodiff(Enzyme.Reverse, Poisson2D, Duplicated(u_loc, ∂R∂u), Const(s[i,j]), Const(type_loc), Const(bcv_loc), Const(Δ))

        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if (num_loc[ii,jj]>0) 
                K[num.num[i,j], num_loc[ii,jj]] = ∂R∂u[ii,jj] 
            end
        end
    end

    return K
end

function RangesPoisson(nc)
    return (inx = 2:nc.x+1, iny = 2:nc.y+1)
end

let
    # Generates an empty numbering structure
    numbering = NumberingPoisson{3}()
    
    # Resolution in FD cells
    nc = (x = 30, y = 40)

    ranges = RangesPoisson(nc)
    @unpack inx, iny = ranges
    
    # Define node types and set BC flags
    numbering.type           = fill(:out, (nc.x+2, nc.y+2))
    numbering.type[inx,iny] .= :in
    numbering.type[1,:]     .= :Dirichlet 
    numbering.type[end,:]   .= :Dirichlet 
    numbering.type[:,1]     .= :Dirichlet
    numbering.type[:,end]   .= :Dirichlet
    numbering.bc_val         = zeros(nc.x+2, nc.y+2)
    numbering.bc_val[1,:]   .= 1.0 
    numbering.bc_val[end,:] .= 1.0 
    numbering.bc_val[:,1]   .= 1.0
    numbering.bc_val[:,end] .= 1.0
    
    @info "Node types"
    Print_xy(numbering.type) 

    # 5-point stencil
    numbering.pattern = @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
    NumberingPoisson!(numbering, nc)
    # Parameters
    L     = 1.
    u_dir = 1.
    # Arrays
    r   = zeros(nc.x+2, nc.y+2)
    s   = zeros(nc.x+2, nc.y+2)
    u   = zeros(nc.x+2, nc.y+2)
    Δ   = (x=L/nc.x, y=L/nc.y)
    xc  = LinRange(-L/2-Δ.x/2, L/2+Δ.x/2, nc.x+2)
    yc  = LinRange(-L/2-Δ.y/2, L/2+Δ.y/2, nc.y+2)
    s  .= 50*exp.(-(xc.^2 .+ (yc').^2)./0.4^2)
    ResidualPoisson2D!(r, u, s, numbering, nc, Δ) 
    @info norm(r)/sqrt(length(r))
    # Assembly
    K  = AssemblyPoisson(u, s, u_dir, numbering, nc, Δ)
    @show norm(K-K')
    b  = r[inx,iny][:]
    # Solve
    du           = K\b
    u[inx,iny] .-= reshape(du, nc...)
    ResidualPoisson2D!(r, u, s, numbering, nc, Δ) 
    @info norm(r)/sqrt(length(r))
    # Visualization
    heatmap(xc[inx], yc[iny], u[inx,iny]')
    # qx = diff(u[inx,iny],dims=1)/Δ.x
    # qy = diff(u[inx,iny],dims=2)/Δ.y
    # @show     mean(qx[1,:])
    # @show     mean(qx[end,:])
    # @show     mean(qy[:,1])
    # @show     mean(qy[:,end])
    # heatmap(xc[1:end-3], yc[iny], qx')
    # heatmap(xc[inx], yc[1:end-3], qy')

end