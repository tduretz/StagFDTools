
Base.@propagate_inbounds @inline inn(A::SMatrix{M,N})   where {M,N} = SMatrix{M-2,N-2}(A[i + 1, j + 1] for i in 1:M-2, j in 1:N-2)
Base.@propagate_inbounds @inline inn_x(A::SMatrix{M,N}) where {M,N} = SMatrix{M-2,N}(  A[i + 1, j]     for i in 1:M-2, j in 1:N)
Base.@propagate_inbounds @inline inn_y(A::SMatrix{M,N}) where {M,N} = SMatrix{M,N-2}(  A[i, j + 1]     for i in 1:M,   j in 1:N-2)
Base.@propagate_inbounds @inline av(A::SMatrix{M,N})    where {M,N} = SMatrix{M-1,N-1}((A[i, j] + A[i+1, j] + A[i, j+1] + A[i+1, j+1]) / 4 for i in 1:M-1, j in 1:N-1)
Base.@propagate_inbounds @inline harm(A::SMatrix{M,N})  where {M,N} = SMatrix{M-1,N-1}(4 * inv(inv(A[i, j]) + inv(A[i+1, j]) + inv(A[i, j+1]) + inv(A[i+1, j+1]))  for i in 1:M-1, j in 1:N-1)
Base.@propagate_inbounds @inline ∂x(A::SMatrix{M,N})    where {M,N} = SMatrix{M-1,N}(A[i+1, j] - A[i, j] for i in 1:M-1, j in 1:N)
Base.@propagate_inbounds @inline ∂y(A::SMatrix{M,N})    where {M,N} = SMatrix{M,N-1}(A[i, j+1] - A[i, j] for i in 1:M, j in 1:N-1)
Base.@propagate_inbounds @inline ∂kk(A::SMatrix{M1,N1}, B::SMatrix{M2,N2}) where {M1,N1,M2,N2} = SMatrix{M1, N2}(A[i, j+1] + B[i+1, j] for i in 1:M1, j in 1:N2)
