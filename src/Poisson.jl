function NumberingPoisson!(N, nc)
    neq                     = nc.x * nc.y
    N.Num                   = zeros(Int64, nc.x+2, nc.y+2)
    N.Num[2:end-1,2:end-1] .= reshape(1:neq, nc.x, nc.y)

    # Make periodic in x
    for j in axes(N.Type,2)
        if N.Type[1,j]==:periodic
            N.Num[1,j] = N.Num[end-1,j]
        end
        if N.Type[end,j]==:periodic
            N.Num[end,j] = N.Num[2,j]
        end
    end

    # Make periodic in y
    for i in axes(N.Type,1)
        if N.Type[i,1]==:periodic
            N.Num[i,1] = N.Num[i,end-1]
        end
        if N.Type[i,end]==:periodic
            N.Num[i,end] = N.Num[i,2]
        end
    end
end

# @views function SparsityPatternPoisson(nc, Num, Pattern::SMatrix{N, N}) where N
@views function SparsityPatternPoisson(Num, nc) 
    ndof   = maximum(Num.Num)
    K      = ExtendableSparseMatrix(ndof, ndof)
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Local = Num.Num[i-1:i+1,j-1:j+1] .* Num.Pattern
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) 
                K[Num.Num[i,j], Local[ii,jj]] = 1 
            end
        end
    end
    return K
end

function SparsityPatternPoisson_SA(Num, Pattern::SMatrix{N, N, T}, nc) where {N,T}
    ndof   = maximum(Num)
    K      = ExtendableSparseMatrix(ndof, ndof)
    shift  = (x=1, y=1)
    star_shift = (N >>> 1) + 1
    Local  = @MMatrix zeros(T, N, N)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        gen_local_numbering!(Local, Num, Pattern, star_shift, i, j)    

        for jj in axes(Local,2), ii in axes(Local,1)
            idx = Local[ii,jj]
            if idx > 0 
                K[Num[i,j], idx] = 1 
            end
        end
    end
    return K
end

@inline @generated function gen_local_numbering!(Local::MMatrix{N,N}, Num, Pattern::SMatrix{N, N}, star_shift, i, j) where N
    quote
        Base.@nexprs $N jj -> begin
            Base.@nexprs $N ii -> begin
                @inline 
                Local[ii, jj] = Num[ii - star_shift + i, jj - star_shift + j]
            end
        end
        Local .*=  Pattern
        return nothing
    end
end