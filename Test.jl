using ExtendableSparse, StaticArrays

Print_xy(x) = @show rotr90(x[:,end:-1:1])

Base.@kwdef mutable struct Physics
    Poisson         ::Bool = false
    Stokes          ::Bool = false
    NonLinearStokes ::Bool = false
    TwoPhases       ::Bool = false
    Cosserat        ::Bool = false
    Thermal         ::Bool = false
end

@views function SparsiTyPatternPoisson(nc, Num, Pattern::SMatrix{N, N}) where N
    ndof   = maximum(Num)
    K      = ExtendableSparseMatrix(ndof, ndof)
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        # Local = @SMatrix (Num[i-1:i+1,j-1:j+1]) #@SMatrix( Num[i-1:i+1,j-1:j+1]  )
        Local = Num[i-1:i+1,j-1:j+1] .* Pattern
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) 
                K[Num[i,j],Local[ii,jj]] = 1 
            end
        end
    end
    return K
end

function SparsiTyPatternPoisson_SA(nc, Num, Pattern::SMatrix{N, N, T}) where {N,T}
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

function NumberingPoisson(nc, Type)
    neq     = nc.x * nc.y
    Num     = zeros(Int64, nc.x+2, nc.y+2)
    Num[2:end-1,2:end-1] .= reshape(1:neq, nc.x, nc.y)

    # Make periodic in x
    for j in axes(Type,2)
        if Type[1,j]==:periodic
            Num[1,j] = Num[end-1,j]
        end
        if Type[end,j]==:periodic
            Num[end,j] = Num[2,j]
        end
    end

    # Make periodic in y
    for i in axes(Type,1)
        if Type[i,1]==:periodic
            Num[i,1] = Num[i,end-1]
        end
        if Type[i,end]==:periodic
            Num[i,end] = Num[i,2]
        end
    end
    return Num
end

let

    physics = Physics()
    physics.Poisson = true
    
    # Resolution
    nc = (x = 5, y= 5)
    
    # 5-point stencil
    Type = fill(:out, (nc.x+2, nc.y+2))
    Type[2:end-1,2:end-1] .= :in
    Type[1,:]     .= :periodic # make periodic
    Type[end,:]   .= :periodic 
    Type[:,1]     .= :Dirichlet
    Type[:,end]   .= :Neumann
    @info "Node types"
    Print_xy(Type) 

    if physics.Poisson
        # 5-point stencil
        Pattern = @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
        Num  = NumberingPoisson(nc, Type)
        K    = SparsiTyPatternPoisson(nc, Num, Pattern)
        K_SA = SparsiTyPatternPoisson_SA(nc, Num, Pattern)
        @assert K == K_SA
        @info "5-point stencil"
        display(K)
        display(K-K')

        # 9-point stencil
        Pattern = @SMatrix([1 1 1; 1 1 1; 1 1 1]) 
        Num = NumberingPoisson(nc, Type)
        K = SparsiTyPatternPoisson(nc, Num, Pattern)
        K_SA = SparsiTyPatternPoisson_SA(nc, Num, Pattern)
        @assert K == K_SA
        @info "9-point stencil"
        display(K)
        display(K-K')               
    end
end