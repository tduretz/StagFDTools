struct Fields{Tx,Ty,Tp,Tpf}
    Vx::Tx
    Vy::Ty
    Pt::Tp
    Pf::Tpf
end

function Base.getindex(x::Fields, i::Int64)
    @assert 0 < i < 5 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
    i == 4 && return x.Pf
end

function Ranges(nc)     
    return (inx_Vx = 2:nc.x+2, iny_Vx = 3:nc.y+2, inx_Vy = 3:nc.x+2, iny_Vy = 2:nc.y+2, inx_c = 2:nc.x+1, iny_c = 2:nc.y+1, inx_v = 2:nc.x+2, iny_v = 2:nc.y+2, size_x = (nc.x+3, nc.y+4), size_y = (nc.x+4, nc.y+3), size_c = (nc.x+2, nc.y+2), size_v = (nc.x+3, nc.y+3))
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
        if type.Vx[i,j] == :Dirichlet_normal || (type.Vx[i,j] != :periodic && i==nc.x+3-1)
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
        if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] != :periodic && j==nc.y+3-1)
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

    ############ Numbering Pf ############

    neq_Pf                    = nc.x * nc.y
    N.Pf[2:end-1,2:end-1] .= reshape(1:neq_Pf, nc.x, nc.y)

    # Make periodic in x
    for j in axes(type.Pf,2)
        if type.Pf[1,j] === :periodic
            N.Pf[1,j] = N.Pf[end-1,j]
        end
        if type.Pf[end,j] === :periodic
            N.Pf[end,j] = N.Pf[2,j]
        end
    end

    # Make periodic in y
    for i in axes(type.Pf,1)
        if type.Pf[i,1] === :periodic
            N.Pf[i,1] = N.Pf[i,end-1]
        end
        if type.Pf[i,end] === :periodic
            N.Pf[i,end] = N.Pf[i,2]
        end
    end

end

function SetRHS!(r, R, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

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
            r[ind] = R.pt[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            r[ind] = R.pf[i,j]
        end
    end
end

function UpdateSolution!(V, P, dx, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

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
            P.t[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            P.f[i,j] += dx[ind]
        end
    end
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
#         if type.Vx[i,j] == :Dirichlet_normal || (type.Vx[i,j] != :periodic && i==nc.x+3-1)
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
#         if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] != :periodic && j==nc.y+3-1)
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

#     ############ Fields Pf ############

#     neq_Pf                    = nc.x * nc.y
#     N.Pf[2:end-1,2:end-1] .= reshape(1:neq_Pf, nc.x, nc.y)

#     # Make periodic in x
#     for j in axes(type.Pf,2)
#         if type.Pf[1,j] === :periodic
#             N.Pf[1,j] = N.Pf[end-1,j]
#         end
#         if type.Pf[end,j] === :periodic
#             N.Pf[end,j] = N.Pf[2,j]
#         end
#     end

#     # Make periodic in y
#     for i in axes(type.Pf,1)
#         if type.Pf[i,1] === :periodic
#             N.Pf[i,1] = N.Pf[i,end-1]
#         end
#         if type.Pf[i,end] === :periodic
#             N.Pf[i,end] = N.Pf[i,2]
#         end
#     end

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
        # Vx --- Pf
        Local = num.Pf[i-1:i,j-2:j] .* pattern[1][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vx[i,j]>0
                K[1][4][num.Vx[i,j], Local[ii,jj]] = 1 
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
        # Vy --- Pf
        Local = num.Pf[i-2:i,j-1:j] .* pattern[2][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Vy[i,j]>0
                K[2][4][num.Vy[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Pt ############
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
        # Pt --- Pf
        Local = num.Pf[i,j] .* pattern[3][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][4][num.Pt[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ Fields Pf ############
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Pf --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[4][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][1][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[4][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][2][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Pt
        Local = num.Pt[i,j] .* pattern[4][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][3][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
        # Pf --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[4][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][4][num.Pf[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    ############ End ############
end