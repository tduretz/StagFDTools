function RangesPoisson(nc)
    return (inx = 2:nc.x+1, iny = 2:nc.y+1)
end

function NumberingPoisson!(N, nc)
    neq                     = nc.x * nc.y
    N.num                   = zeros(Int64, nc.x+2, nc.y+2)
    N.num[2:end-1,2:end-1] .= reshape(1:neq, nc.x, nc.y)

    # Make periodic in x
    for j in axes(N.type,2)
        if N.type[1,j]==:periodic
            N.num[1,j] = N.num[end-1,j]
        end
        if N.type[end,j]==:periodic
            N.num[end,j] = N.num[2,j]
        end
    end

    # Make periodic in y
    for i in axes(N.type,1)
        if N.type[i,1]==:periodic
            N.num[i,1] = N.num[i,end-1]
        end
        if N.type[i,end]==:periodic
            N.num[i,end] = N.num[i,2]
        end
    end
end

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

@views function SparsityPatternPoisson(num, nc) 
    ndof   = maximum(num.num)
    K      = ExtendableSparseMatrix(ndof, ndof)
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Local = num.num[i-1:i+1,j-1:j+1] .* num.pattern
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) 
                K[num.num[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    return K
end

function SparsityPatternPoisson_SA(num, pattern::SMatrix{N, N, T}, nc) where {N,T}
    ndof   = maximum(num)
    K      = ExtendableSparseMatrix(ndof, ndof)
    shift  = (x=1, y=1)
    star_shift = (N >>> 1) + 1
    Local  = @MMatrix zeros(T, N, N)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        gen_local_numbering!(Local, num, pattern, star_shift, i, j)    

        for jj in axes(Local,2), ii in axes(Local,1)
            idx = Local[ii,jj]
            if idx > 0 
                K[num[i,j], idx] = 1 
            end
        end
    end
    return K
end

@inline @generated function gen_local_numbering!(Local::MMatrix{N,N}, num, pattern::SMatrix{N, N}, star_shift, i, j) where N
    quote
        Base.@nexprs $N jj -> begin
            Base.@nexprs $N ii -> begin
                @inline 
                Local[ii, jj] = num[ii - star_shift + i, jj - star_shift + j]
            end
        end
        Local .*=  pattern
        return nothing
    end
end