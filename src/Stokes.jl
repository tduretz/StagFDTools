struct Fields{Tx,Ty,Tp}
    Vx::Tx
    Vy::Ty
    Pt::Tp
end

function Base.getindex(x::Fields, i::Int64)
    @assert 0 < i < 4 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
end

function Ranges(nc)     
    return (inx_Vx = 2:nc.x+2, iny_Vx = 3:nc.y+2, inx_Vy = 3:nc.x+2, iny_Vy = 2:nc.y+2, inx_Pt = 2:nc.x+1, iny_Pt = 2:nc.y+1, size_x = (nc.x+3, nc.y+4), size_y = (nc.x+4, nc.y+3), size_p = (nc.x+2, nc.y+2))
end

# function Numbering!(N, type, nc)
    
#     ndof  = 0
#     neq   = 0
#     noisy = false

#     ############ Fields Vx ############
#     periodic_west  = sum(any(i->i==:periodic, type.Vx[2,:], dims=2)) > 0
#     periodic_south = sum(any(i->i==:periodic, type.Vx[:,2], dims=1)) > 0

#     shift  = (periodic_west) ? 1 : 0 
#     # Loop through inner nodes of the mesh
#     for j=3:nc.y+4-2, i=2:nc.x+3-1
#         if type.Vx[i,j] == :constant || (type.Vx[i,j] != :periodic && i==nc.x+3-1)
#             # Avoid nodes with constant velocity or redundant periodic nodes
#         else
#             ndof+=1
#             N.Vx[i,j] = ndof  
#         end
#     end

#     # Copy equation indices for periodic cases
#     if periodic_west
#         N.Vx[1,:] .= N.Vx[end-2,:]
#     end

#     # Copy equation indices for periodic cases
#     if periodic_south
#         # South
#         N.Vx[:,1] .= N.Vx[:,end-3]
#         N.Vx[:,2] .= N.Vx[:,end-2]
#         # North
#         N.Vx[:,end]   .= N.Vx[:,4]
#         N.Vx[:,end-1] .= N.Vx[:,3]
#     end
#     noisy ? printxy(N.Vx) : nothing

#     neq = maximum(N.Vx)

#     ############ Fields Vy ############
#     ndof  = 0
#     periodic_west  = sum(any(i->i==:periodic, type.Vy[2,:], dims=2)) > 0
#     periodic_south = sum(any(i->i==:periodic, type.Vy[:,2], dims=1)) > 0
#     shift = periodic_south ? 1 : 0
#     # Loop through inner nodes of the mesh
#     for j=2:nc.y+3-1, i=3:nc.x+4-2
#         if type.Vy[i,j] == :constant || (type.Vy[i,j] != :periodic && j==nc.y+3-1)
#             # Avoid nodes with constant velocity or redundant periodic nodes
#         else
#             ndof+=1
#             N.Vy[i,j] = ndof  
#         end
#     end

#     # Copy equation indices for periodic cases
#     if periodic_south
#         N.Vy[:,1] .= N.Vy[:,end-2]
#     end

#     # Copy equation indices for periodic cases
#     if periodic_west
#         # West
#         N.Vy[1,:] .= N.Vy[end-3,:]
#         N.Vy[2,:] .= N.Vy[end-2,:]
#         # East
#         N.Vy[end,:]   .= N.Vy[4,:]
#         N.Vy[end-1,:] .= N.Vy[3,:]
#     end
#     noisy ? printxy(N.Vy) : nothing

#     neq = maximum(N.Vy)

#     ############ Fields Pt ############
#     neq_Pt                     = nc.x * nc.y
#     N.Pt[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ 0*neq, nc.x, nc.y)

#     if periodic_west
#         N.Pt[1,:]   .= N.Pt[end-1,:]
#         N.Pt[end,:] .= N.Pt[2,:]
#     end

#     if periodic_south
#         N.Pt[:,1]   .= N.Pt[:,end-1]
#         N.Pt[:,end] .= N.Pt[:,2]
#     end
#     noisy ? printxy(N.Pt) : nothing

#     neq = maximum(N.Pt)

# end

@views function SparsityPattern!(K, num, pattern, nc) 
    ############ Fields Vx ############
    shift  = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vx --- Vx
        Local = num.Vx[i-1:i+1,j-1:j+1] .* pattern[1][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][1][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Vy
        Local = num.Vy[i-1:i+2,j-2:j+1] .* pattern[1][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][2][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vx --- Pt
        Local = num.Pt[i-1:i,j-2:j] .* pattern[1][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][3][num.Vx[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Vy ############
    shift  = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Vy --- Vx
        Local = num.Vx[i-2:i+1,j-1:j+2] .* pattern[2][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][1][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Vy
        Local = num.Vy[i-1:i+1,j-1:j+1] .* pattern[2][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][2][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
        # Vy --- Pt
        Local = num.Pt[i-2:i,j-1:j] .* pattern[2][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][3][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    # ############ Fields Pt ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pt --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[3][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pt --- Pt
        Local = num.Pt[i,j] .* pattern[3][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][3][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ End ############
    return K
end


function SetBCVx!(Vx_loc, bcx_loc, bcv, Δ)

    for ii in axes(Vx_loc, 1)

        # Set Vx boundaries at S (this must be done 1st)
        if bcx_loc[ii,begin] == :Neumann 
            Vx_loc[ii,begin] =  Vx_loc[ii,begin+1] - Δ.y*bcv.∂Vx∂y_BC[ii,1]
        elseif bcx_loc[ii,begin] == :Dirichlet 
            Vx_loc[ii,begin] = -Vx_loc[ii,begin+1] + 2*bcv.Vx_BC[ii,1]
        end
        if bcx_loc[ii,begin] == :out 
            if bcx_loc[ii,begin+1] == :Neumann
                Vx_loc[ii,begin+1] =  Vx_loc[ii,begin+2] -   Δ.y*bcv.∂Vx∂y_BC[ii,1]
                Vx_loc[ii,begin]   =  Vx_loc[ii,begin+3] - 3*Δ.y*bcv.∂Vx∂y_BC[ii,1] 
            elseif bcx_loc[ii,begin+1] == :Dirichlet
                Vx_loc[ii,begin+1] = -Vx_loc[ii,begin+2] + 2*bcv.Vx_BC[ii,1]
                Vx_loc[ii,begin]   = -Vx_loc[ii,begin+3] + 2*bcv.Vx_BC[ii,1] 
            end
        end

        # Set Vx boundaries at N (this must be done 1st)
        if bcx_loc[ii,end] == :Neumann 
            Vx_loc[ii,end] =  Vx_loc[ii,end-1] + Δ.y*bcv.∂Vx∂y_BC[ii,2] 
        elseif bcx_loc[ii,end] == :Dirichlet 
            Vx_loc[ii,end] = -Vx_loc[ii,end-1] + 2*bcv.Vx_BC[ii,2]
        end
        if bcx_loc[ii,end] == :out
            if bcx_loc[ii,end-1] == :Neumann
                Vx_loc[ii,end-1] =  Vx_loc[ii,end-2] +   Δ.y*bcv.∂Vx∂y_BC[ii,2] 
                Vx_loc[ii,end]   =  Vx_loc[ii,end-3] + 3*Δ.y*bcv.∂Vx∂y_BC[ii,2]   
            elseif bcx_loc[ii,3] == :Dirichlet
                Vx_loc[ii,end-1] = -Vx_loc[ii,end-2] + 2*bcv.Vx_BC[ii,2] 
                Vx_loc[ii,end]   = -Vx_loc[ii,end-3] + 2*bcv.Vx_BC[ii,2]  
            end
        end
    end

    # for jj in axes(Vx_loc, 2)
    #     # Set Vx boundaries at W (this must be done 2nd)
    #     if bcx_loc[1,jj] == :out
    #         Vx_loc[1,jj] = Vx_loc[2,jj] - Δ.x*bcv.∂Vx∂x_BC[1,jj] 
    #     end
    #     # Set Vx boundaries at E (this must be done 2nd)
    #     if bcx_loc[3,jj] == :out
    #         Vx_loc[3,jj] = Vx_loc[2,jj] + Δ.x*bcv.∂Vx∂x_BC[2,jj] 
    #     end
    # end
end

function SetBCVy!(Vy_loc, bcy_loc, bcv, Δ)
    
    for jj in axes(Vy_loc, 2)

        # Set Vy boundaries at W (this must be done 1st)
        if bcy_loc[begin,jj] == :Neumann 
            Vy_loc[begin,jj] =  Vy_loc[begin+1,jj] - Δ.x*bcv.∂Vy∂x_BC[1,jj] 
        elseif bcy_loc[begin,jj] == :Dirichlet 
            Vy_loc[begin,jj] = -Vy_loc[begin+1,jj] + 2*bcv.Vy_BC[1,jj]
        end
        if bcy_loc[begin,jj] == :out
            if bcy_loc[begin+1,jj] == :Neumann 
                Vy_loc[begin+1,jj] = Vy_loc[begin+2,jj] -   Δ.y*bcv.∂Vy∂x_BC[1,jj] 
                Vy_loc[begin,jj]   = Vy_loc[begin+3,jj] - 3*Δ.y*bcv.∂Vy∂x_BC[1,jj] 
            elseif bcy_loc[begin+1,jj] == :Dirichlet
                Vy_loc[begin+1,jj] = -Vy_loc[begin+2,jj] + 2*bcv.Vy_BC[1,jj]
                Vy_loc[begin,jj]   = -Vy_loc[begin+3,jj] + 2*bcv.Vy_BC[1,jj]
            end 
        end

        # Set Vy boundaries at E (this must be done 1st)
        if bcy_loc[end,jj] == :Neumann 
            Vy_loc[end,jj] = Vy_loc[end-1,jj] + Δ.x*bcv.∂Vy∂x_BC[1,jj] 
        elseif bcy_loc[end,jj] == :Dirichlet 
            Vy_loc[end,jj] = -Vy_loc[end-1,jj] + 2*bcv.Vy_BC[2,jj]
        end
        if bcy_loc[end,jj] == :out
            if bcy_loc[end-1,jj] == :Neumann 
                Vy_loc[end-1,jj] = Vy_loc[end-2,jj] +   Δ.y*bcv.∂Vy∂x_BC[1,jj]
                Vy_loc[end,jj]   = Vy_loc[end-3,jj] + 3*Δ.y*bcv.∂Vy∂x_BC[1,jj]
            elseif bcy_loc[3,jj] == :Dirichlet 
                Vy_loc[end-1,jj] = -Vy_loc[end-2,jj] + 2*bcv.Vy_BC[2,jj]
                Vy_loc[end,jj]   = -Vy_loc[end-3,jj] + 2*bcv.Vy_BC[2,jj]
            end
        end
    end

    # for ii in axes(Vy_loc, 1)
    #     # Set Vy boundaries at S (this must be done 2nd)
    #     if bcy_loc[ii,1] == :out
    #         Vy_loc[ii,1] = Vy_loc[ii,2] - Δ.y*bcv.∂Vy∂y_BC[ii,1]
    #     end
    #     # Set Vy boundaries at S (this must be done 2nd)
    #     if bcy_loc[ii,3] == :out
    #         Vy_loc[ii,3] = Vy_loc[ii,2] + Δ.y*bcv.∂Vy∂y_BC[ii,2]
    #     end
    # end
end

function SetRHS!(r, R, number, type, nc)

    nVx, nVy   = maximum(number.Vx), maximum(number.Vy)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            r[ind] = R.x[i,j]
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            r[ind] = R.y[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            r[ind] = R.p[i,j]
        end
    end
end

function UpdateSolution!(V, Pt, dx, number, type, nc)

    nVx, nVy   = maximum(number.Vx), maximum(number.Vy)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            V.x[i,j] += dx[ind] 
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            V.y[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            Pt[i,j] += dx[ind]
        end
    end
end

function Numbering!(N, type, nc)
    
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
    noisy ? printxy(N.Vx) : nothing

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
    noisy ? printxy(N.Vy) : nothing

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
    noisy ? printxy(N.Pt) : nothing

    neq = maximum(N.Pt)
end

