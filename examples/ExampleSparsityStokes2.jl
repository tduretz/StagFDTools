using StagFDTools, ExtendableSparse, StaticArrays

struct NumberingV <: AbstractPattern # ??? where is AbstractPattern defined 
    x
    y
    p
end

struct Numbering{Tx,Ty,Tp}
    Vx::Tx
    Vy::Ty
    Pt::Tp
end

function Base.getindex(x::Numbering, i::Int64)
    @assert 0 < i < 4 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
end

struct PatternV <: AbstractPattern
    xx
    xy 
    xp
    yx 
    yy
    yp
    px
    py
    pp 
end

struct Pattern{Txx,Txy,Txp, Tyx,Tyy,Typ, Tpx,Tpy,Tpp}
    xx::Txx
    xy::Txy 
    xp::Txp
    yx::Tyx 
    yy::Tyy
    yp::Typ
    px::Tpx
    py::Tpy
    pp::Tpp
end

function Base.getindex(x::Pattern, i::Int64, j::Int64)
    @assert 0 < i < 4 
    @assert 0 < j < 4 
    isone(i) && isone(j) && return x.xx
    i === 2  && isone(j) && return x.yx
    i === 3  && isone(j) && return x.px
    isone(i) && j === 2  && return x.xy
    i === 2  && j === 2  && return x.yy
    i === 3  && j === 2  && return x.py
    isone(i) && j === 3  && return x.xp
    i === 2  && j === 3  && return x.yp
    i === 3  && j === 3  && return x.pp
end

Base.@kwdef mutable struct StokesPattern
    num     ::Union{Matrix{Int64},   Missing} = missing
    type    ::Union{Matrix{Symbol},  Missing} = missing
    patternVx ::Union{SMatrix,  Missing} = missing  # these could be contained in same object - but they have different sizes
    patternVy ::Union{SMatrix,  Missing} = missing  # these could be contained in same object - but they have different sizes
    patternPt ::Union{SMatrix,  Missing} = missing  # these could be contained in same object - but they have different sizes
    # ideally we would like other fields also part of it (fluid pressure, microrotation)
end

Base.@kwdef mutable struct NumberingStokes
    Vx ::Union{StokesPattern, Missing} = missing
    Vy ::Union{StokesPattern, Missing} = missing
    Pt ::Union{StokesPattern, Missing} = missing
end

function NumberingStokes!(N, nc)
    
    ndof  = 0
    neq   = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west  = sum(any(i->i==:periodic, N.Vx.type[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, N.Vx.type[:,2], dims=1)) > 0

    shift  = (periodic_west) ? 1 : 0 
    N.Vx.num                   = zeros(Int64, nc.x+3, nc.y+4) 
    # Loop through inner nodes of the mesh
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if N.Vx.type[i,j] == :constant || (N.Vx.type[i,j] != :periodic && i==nc.x+3-1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vx.num[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx.num[1,:] .= N.Vx.num[end-2,:]
    end

    # Copy equation indices for periodic cases
    if periodic_south
        # South
        N.Vx.num[:,1] .= N.Vx.num[:,end-3]
        N.Vx.num[:,2] .= N.Vx.num[:,end-2]
        # North
        N.Vx.num[:,end]   .= N.Vx.num[:,4]
        N.Vx.num[:,end-1] .= N.Vx.num[:,3]
    end
    noisy ? Print_xy(N.Vx.num) : nothing

    neq = maximum(N.Vx.num)

    ############ Numbering Vy ############
    periodic_west  = sum(any(i->i==:periodic, N.Vy.type[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, N.Vy.type[:,2], dims=1)) > 0
    shift = periodic_south ? 1 : 0
    N.Vy.num                   = zeros(Int64, nc.x+4, nc.y+3)
    # Loop through inner nodes of the mesh
    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if N.Vy.type[i,j] == :constant || (N.Vy.type[i,j] != :periodic && j==nc.y+3-1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vy.num[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_south
        N.Vy.num[:,1] .= N.Vy.num[:,end-2]
    end

    # Copy equation indices for periodic cases
    if periodic_west
        # West
        N.Vy.num[1,:] .= N.Vy.num[end-3,:]
        N.Vy.num[2,:] .= N.Vy.num[end-2,:]
        # East
        N.Vy.num[end,:]   .= N.Vy.num[4,:]
        N.Vy.num[end-1,:] .= N.Vy.num[3,:]
    end
    noisy ? Print_xy(N.Vy.num) : nothing

    neq = maximum(N.Vy.num)

    ############ Numbering Pt ############
    neq_Pt                     = nc.x * nc.y
    N.Pt.num                   = zeros(Int64, nc.x+2, nc.y+2)
    N.Pt.num[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ neq, nc.x, nc.y)

    if periodic_west
        N.Pt.num[1,:]   .= N.Pt.num[end-1,:]
        N.Pt.num[end,:] .= N.Pt.num[2,:]
    end

    if periodic_south
        N.Pt.num[:,1]   .= N.Pt.num[:,end-1]
        N.Pt.num[:,end] .= N.Pt.num[:,2]
    end
    noisy ? Print_xy(N.Pt.num) : nothing

    neq = maximum(N.Pt.num)

end

function NumberingStokes2!(N, type, nc)
    
    ndof  = 0
    neq   = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west  = sum(any(i->i==:periodic, type.Vx[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vx[:,2], dims=1)) > 0

    shift  = (periodic_west) ? 1 : 0 
    # Loop through inner nodes of the mesh
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vx[i,j] == :constant || (type.Vx[i,j] != :periodic && i==nc.x+3-1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vx[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx[1,:] .= N.Vx[end-2,:]
    end

    # Copy equation indices for periodic cases
    if periodic_south
        # South
        N.Vx[:,1] .= N.Vx[:,end-3]
        N.Vx[:,2] .= N.Vx[:,end-2]
        # North
        N.Vx[:,end]   .= N.Vx[:,4]
        N.Vx[:,end-1] .= N.Vx[:,3]
    end
    noisy ? Print_xy(N.Vx) : nothing

    neq = maximum(N.Vx)

    ############ Numbering Vy ############
    ndof  = 0
    periodic_west  = sum(any(i->i==:periodic, type.Vy[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vy[:,2], dims=1)) > 0
    shift = periodic_south ? 1 : 0
    # Loop through inner nodes of the mesh
    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vy[i,j] == :constant || (type.Vy[i,j] != :periodic && j==nc.y+3-1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vy[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_south
        N.Vy[:,1] .= N.Vy[:,end-2]
    end

    # Copy equation indices for periodic cases
    if periodic_west
        # West
        N.Vy[1,:] .= N.Vy[end-3,:]
        N.Vy[2,:] .= N.Vy[end-2,:]
        # East
        N.Vy[end,:]   .= N.Vy[4,:]
        N.Vy[end-1,:] .= N.Vy[3,:]
    end
    noisy ? Print_xy(N.Vy) : nothing

    neq = maximum(N.Vy)

    ############ Numbering Pt ############
    neq_Pt                     = nc.x * nc.y
    N.Pt[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ 0*neq, nc.x, nc.y)

    if periodic_west
        N.Pt[1,:]   .= N.Pt[end-1,:]
        N.Pt[end,:] .= N.Pt[2,:]
    end

    if periodic_south
        N.Pt[:,1]   .= N.Pt[:,end-1]
        N.Pt[:,end] .= N.Pt[:,2]
    end
    noisy ? Print_xy(N.Pt) : nothing

    neq = maximum(N.Pt)

end


@views function SparsityPatternStokes(num, nc) 
    ############ Start ############
    ndof   = maximum(num.Pt.num)
    K      = ExtendableSparseMatrix(ndof, ndof)
    ############ Numbering Vx ############
    shift  = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vx --- Vx
        Local = num.Vx.num[i-1:i+1,j-1:j+1] .* num.Vx.patternVx
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx.num[i,j]>0
                K[num.Vx.num[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Vy
        Local = num.Vy.num[i:i+1,j-2:j+1] .* num.Vx.patternVy
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx.num[i,j]>0
                K[num.Vx.num[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Pt
        Local = num.Pt.num[i-1:i,j-2:j] .* num.Vx.patternPt
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx.num[i,j]>0
                K[num.Vx.num[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Numbering Vy ############
    shift  = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vy --- Vx
        Local = num.Vx.num[i-2:i+1,j:j+1] .* num.Vy.patternVx
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy.num[i,j]>0
                K[num.Vy.num[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Vy
        Local = num.Vy.num[i-1:i+1,j-1:j+1] .* num.Vy.patternVy
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy.num[i,j]>0
                K[num.Vy.num[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Pt
        Local = num.Pt.num[i-2:i,j-1:j] .* num.Vy.patternPt
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy.num[i,j]>0
                K[num.Vy.num[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Numbering Vy ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pt --- Vx
        Local = num.Vx.num[i:i+1,j:j+2] .* num.Pt.patternVx
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt.num[i,j]>0
                K[num.Pt.num[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Vy
        Local = num.Vy.num[i:i+2,j:j+1] .* num.Pt.patternVy
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt.num[i,j]>0
                K[num.Pt.num[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Pt
        Local = num.Pt.num[i,j] .* num.Pt.patternPt
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt.num[i,j]>0
                K[num.Pt.num[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ End ############
    return K
end

@views function SparsityPatternStokes2!(K, num, pattern, nc) 
    ############ Numbering Vx ############
    shift  = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vx --- Vx
        Local = num.Vx[i-1:i+1,j-1:j+1] .* pattern[1,1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1,1][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Vy
        Local = num.Vy[i:i+1,j-2:j+1] .* pattern[1,2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1,2][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Pt
        Local = num.Pt[i-1:i,j-2:j] .* pattern[1,3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1,3][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Numbering Vy ############
    shift  = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vy --- Vx
        Local = num.Vx[i-2:i+1,j:j+1] .* pattern[2,1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2,1][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Vy
        Local = num.Vy[i-1:i+1,j-1:j+1] .* pattern[2,2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2,2][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Pt
        Local = num.Pt[i-2:i,j-1:j] .* pattern[2,3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2,3][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    # ############ Numbering Pt ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pt --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[3,1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3,1][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[3,2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3,2][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Pt
        Local = num.Pt[i,j] .* pattern[3,3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3,3][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ End ############
    return K
end


let
    physics = Physics()
    physics.Stokes = true
    
    # Resolution
    nc = (x = 4, y = 3)
    
    # Define node types and set BC flags
    type = Numbering(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    # -------- Vx -------- #
    type.Vx[2:end-1,3:end-2] .= :in       
    type.Vx[2,2:end-1]       .= :constant 
    type.Vx[end-1,2:1:end-1] .= :constant 
    type.Vx[2:end-1,2]       .= :Dirichlet
    type.Vx[2:end-1,end-1]   .= :Dirichlet
    # -------- Vy -------- #
    type.Vy[2:end-2,2:end-1] .= :in       
    type.Vy[2,2:end-1]       .= :Dirichlet
    type.Vy[end-1,2:end-1]   .= :Dirichlet
    type.Vy[2:end-1,2]       .= :constant 
    type.Vy[2:end-1,end-1]   .= :constant 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in
    
    # Stencil extent for each block matrix
    pattern = Pattern(
        @SMatrix([0 1 0; 1 1 1; 0 1 0]),
        @SMatrix([0 1 1 0; 0 1 1 0]),
        @SMatrix([0 1 0; 0 1 0]),
        #-----------------------   
        @SMatrix([0 0; 1 1; 1 1; 0 0]),
        @SMatrix([0 1 0; 1 1 1; 0 1 0]), 
        @SMatrix([0 0; 1 1; 0 0]),
        #----------------------- 
        @SMatrix([0 1 0; 0 1 0]),
        @SMatrix([0 0; 1 1 ; 0 0]),
        @SMatrix([1]),
    )

    # Equation numbering
    number = Numbering(
        fill(0, (nc.x+3, nc.y+4)),
        fill(0, (nc.x+4, nc.y+3)),
        fill(0, (nc.x+2, nc.y+2)),
    )
    NumberingStokes2!(number, type, nc)

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Pattern(
        ExtendableSparseMatrix(nVx, nVx),
        ExtendableSparseMatrix(nVx, nVy),
        ExtendableSparseMatrix(nVx, nPt),
        #-----------------------   
        ExtendableSparseMatrix(nVy, nVx), 
        ExtendableSparseMatrix(nVy, nVy),
        ExtendableSparseMatrix(nVy, nPt),
        #----------------------- 
        ExtendableSparseMatrix(nPt, nVx),
        ExtendableSparseMatrix(nPt, nVy),
        ExtendableSparseMatrix(nPt, nPt),
    )

    @info "Assembly, ndof  = $(nVx + nVy + nPt)"
    SparsityPatternStokes2!(M, number, pattern, nc)

    # Stokes blocs
    K  = [M.xx M.xy; M.yx M.yy]
    Q  = [M.xp; M.yp]
    Qᵀ = [M.px M.py]

    @info "Velocity block symmetry"
    display(K - K')

    @info "Grad-Div symmetry"
    display(Q' - Qᵀ)

end