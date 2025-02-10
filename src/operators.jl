for type in (:SMatrix, :MMatrix)
    @eval begin
        Base.@propagate_inbounds @inline inn(A::($type){M,N})    where {M,N} = ($type){M-2,N-2}(A[i + 1, j + 1] for i in 1:M-2, j in 1:N-2)
        Base.@propagate_inbounds @inline inn_x(A::($type){M,N})  where {M,N} = ($type){M-2,N}(  A[i + 1, j]     for i in 1:M-2, j in 1:N)
        Base.@propagate_inbounds @inline inn_y(A::($type){M,N})  where {M,N} = ($type){M,N-2}(  A[i, j + 1]     for i in 1:M,   j in 1:N-2)
        Base.@propagate_inbounds @inline av(A::($type){M,N})     where {M,N} = ($type){M-1,N-1}((A[i, j] + A[i+1, j] + A[i, j+1] + A[i+1, j+1]) / 4 for i in 1:M-1, j in 1:N-1)
        Base.@propagate_inbounds @inline harm(A::($type){M,N})   where {M,N} = ($type){M-1,N-1}(4 * inv(inv(A[i, j]) + inv(A[i+1, j]) + inv(A[i, j+1]) + inv(A[i+1, j+1]))  for i in 1:M-1, j in 1:N-1)
        Base.@propagate_inbounds @inline ∂x(A::($type){M,N})     where {M,N} = ($type){M-1,N}(A[i+1, j] - A[i, j] for i in 1:M-1, j in 1:N)
        Base.@propagate_inbounds @inline ∂x_inn(A::($type){M,N}) where {M,N} = ($type){M-1,N-2}(A[i+1, j] - A[i, j] for i in 1:M-1, j in 2:N-1)
        Base.@propagate_inbounds @inline ∂y(A::($type){M,N})     where {M,N} = ($type){M,N-1}(A[i, j+1] - A[i, j] for i in 1:M, j in 1:N-1)
        Base.@propagate_inbounds @inline ∂y_inn(A::($type){M,N}) where {M,N} = ($type){M-2,N-1}(A[i, j+1] - A[i, j] for i in 2:M-1, j in 1:N-1)
        Base.@propagate_inbounds @inline ∂kk(A::($type){M1,N1}, B::($type){M2,N2}) where {M1,N1,M2,N2} = ($type){M1, N2}(A[i, j+1] + B[i+1, j] for i in 1:M1, j in 1:N2)
    end
end