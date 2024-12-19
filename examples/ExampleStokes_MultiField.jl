using StagFDTools, ExtendableSparse, StaticArrays

# Base.@kwdef mutable struct StokesPattern
#     num     ::Union{Matrix{Int64},   Missing} = missing
#     type    ::Union{Matrix{Symbol},  Missing} = missing
#     patternVx ::Union{SMatrix,  Missing} = missing  # these could be contained in same object - but they have different sizes
#     patternVy ::Union{SMatrix,  Missing} = missing  # these could be contained in same object - but they have different sizes
#     patternPt ::Union{SMatrix,  Missing} = missing  # these could be contained in same object - but they have different sizes
#     # ideally we would like other fields also part of it (fluid pressure, microrotation)
# end

# Base.@kwdef mutable struct NumberingStokes
#     Vx ::Union{StokesPattern, Missing} = missing
#     Vy ::Union{StokesPattern, Missing} = missing
#     Pt ::Union{StokesPattern, Missing} = missing
# end

function  NumberingMultifield(Fields, nc)

    Numbering = (;)

    if any(x->x==:V, Fields)
        data = ( 
            Vx= (num=),
        
        )
        @show merge(Numbering, data)

    end


end

function NumberingStokes!(N, nc)
    
    ndof  = 0
    neq   = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west  = sum(any(i->i==:periodic, N.Vx.type[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, N.Vx.type[:,2], dims=1)) > 0

    # One could also directly eliminate Dirichlet dofs here

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
        # if j==3 && i==3 # debug for ncx = 4 and ncy = 3
        #     display(num.Pt.num[i,j])
        #     Print_xy(num.Vx.num[i:i+1,j:j+2])
        #     Print_xy(num.Pt.patternVx)
        # end
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


let


    Fields = [:V, :P]
    
    # Resolution
    nc = (x = 4, y = 3)

    numbering = NumberingMultifield(Fields, nc)
    
    # # Define node types and set BC flags
    # numbering.Vx      = StokesPattern()
    # numbering.Vx.type = fill(:out, (nc.x+3, nc.y+4))
    # numbering.Vx.type[2:end-1,3:end-2] .= :in
    # numbering.Vx.type[2,2:end-1]       .= :constant # make periodic
    # numbering.Vx.type[end-1,2:1:end-1] .= :constant 
    # numbering.Vx.type[2:end-1,2]       .= :Dirichlet
    # numbering.Vx.type[2:end-1,end-1]   .= :Dirichlet
    # @info "Vx Node types"
    # Print_xy(numbering.Vx.type) 

    # numbering.Vy      = StokesPattern()
    # numbering.Vy.type = fill(:out, (nc.x+4, nc.y+3))
    # numbering.Vy.type[2:end-2,2:end-1] .= :in
    # numbering.Vy.type[2,2:end-1]       .= :Dirichlet # make periodic
    # numbering.Vy.type[end-1,2:end-1]   .= :Dirichlet 
    # numbering.Vy.type[2:end-1,2]       .= :constant
    # numbering.Vy.type[2:end-1,end-1]   .= :constant
    # @info "Vy Node types"
    # Print_xy(numbering.Vy.type) 

    # numbering.Pt      = StokesPattern()#NumberingPoisson{3}()
    # numbering.Pt.type = fill(:out, (nc.x+2, nc.y+2))
    # numbering.Pt.type[2:end-1,2:end-1] .= :in
    # @info "Pt Node types"
    # Print_xy(numbering.Pt.type) 

    # # For Stokes matrices have different sizes
    # # ... if we want more coupling (T-H-Coserat) more fields could be dynamically added.
    # numbering.Vx.patternVx = @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
    # numbering.Vx.patternVy = @SMatrix([0 1 1 0; 0 1 1 0]) 
    # numbering.Vx.patternPt = @SMatrix([0 1 0; 0 1 0])

    # numbering.Vy.patternVx = @SMatrix([0 0; 1 1; 1 1; 0 0])
    # numbering.Vy.patternVy = @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
    # numbering.Vy.patternPt = @SMatrix([0 0; 1 1; 0 0])

    # numbering.Pt.patternVx = @SMatrix([0 1 0; 0 1 0])
    # numbering.Pt.patternVy = @SMatrix([0 0; 1 1 ; 0 0]) 
    # numbering.Pt.patternPt = @SMatrix([1])
    # num        = NumberingStokes!(numbering, nc)

    # @info "Assembly"
    # M  = SparsityPatternStokes(numbering, nc)
    # display(M)

    # # Blocks
    # nV = maximum(numbering.Vy.num)
    # K  = M[1:nV,1:nV]
    # Q  = M[(nV+1):end,1:nV]
    # Qᵀ = M[1:nV,(nV+1):end]

    # @info "Velocity block symmetry"
    # display(K - K')

    # @info "Grad-Div symmetry"
    # display(Q' - Qᵀ)

end