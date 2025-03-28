using Plots, Random, LinearAlgebra, SparseArrays, StaticArrays, ExtendableSparse
using StagFDTools.Poisson
# Adapted from ChatGPT

# # Function to compute Gaussian covariance between points based on distance
# @inline function gaussian_covariance(x1, y1, x2, y2, length_scale)
#     return exp(-((x1 - x2)^2 + (y1 - y2)^2) / (2 * length_scale^2))
# end

# @views function main()
#     # Parameters
#     grid_size = (100, 100)  # 100x100 grid
#     correlation_length = 1.0  # Correlation length in grid points
#     porosity_threshold = 1.9  # Threshold for fluid/solid classification (0 - fluid, 1 - solid)
#     perturbation = 1e-6  # Small value to add to the diagonal for positive definiteness

#     # Step 1: Create the covariance matrix in real space (distances between all grid points)
#     cov_matrix = zeros(Float64, grid_size[1] * grid_size[2], grid_size[1] * grid_size[2])

#     # Loop over all pairs of grid points and fill in the covariance matrix
#     for i in 1:grid_size[1]
#         for j in 1:grid_size[2]
#             for k in 1:grid_size[1]
#                 for l in 1:grid_size[2]
#                     idx1 = (i - 1) * grid_size[2] + j  # Flatten the 2D index to 1D
#                     idx2 = (k - 1) * grid_size[2] + l
#                     cov_matrix[idx1, idx2] = gaussian_covariance(i, j, k, l, correlation_length)
#                 end
#             end
#         end
#     end

#     # Step 2: Add small perturbation to the diagonal to ensure positive definiteness
#     for i in 1:(grid_size[1] * grid_size[2])
#         cov_matrix[i, i] += perturbation  # Add perturbation to diagonal
#     end

#     # Step 3: Cholesky decomposition to generate correlated random field
#     L = cholesky(cov_matrix).L

#     # Step 4: Generate uncorrelated Gaussian noise (one random value per grid point)
#     uncorrelated_noise = randn(grid_size[1] * grid_size[2])

#     # Step 5: Generate the correlated random field
#     correlated_field = L * uncorrelated_noise

#     # Step 6: Reshape the 1D correlated field into a 2D matrix
#     correlated_field_2d = reshape(correlated_field, grid_size[1], grid_size[2])

#     # Step 7: Apply a threshold to create a binary fluid/solid structure
#     binary_structure = correlated_field_2d .< porosity_threshold  # True (fluid) or False (solid)

#     # Step 8: Visualize the resulting binary porous structure
#     heatmap(binary_structure, color=:coolwarm, xlabel="X", ylabel="Y", title="Correlated Porous Media Structure")

#     ϕ = sum(binary_structure.==0) / (*(size(binary_structure)...))
#     @show (*(size(binary_structure)...))
#     @show  ϕ
# end

# @time main()

# Function to compute Gaussian covariance between points based on distance
@inline function gaussian_covariance(x1, y1, x2, y2, length_scale)
    return exp(-((x1 - x2)^2 + (y1 - y2)^2)  * length_scale)
end

function covariance_matrix!(cov_matrix, grid_size, length_scale)
    n1,n2 = grid_size
    Threads.@threads for i in 1:n1
        for j in 1:n2
            idx1 = (i - 1) * n2 + j  # Flatten the 2D index to 1D
            for k in 1:n1
                @simd for l in 1:n2
                    idx2 = (k - 1) * n2 + l
                    @inbounds cov_matrix[idx1, idx2] = gaussian_covariance(i, j, k, l, length_scale)
                end
            end
        end
    end
end

function covariance_matrix!(cov_matrix, grid_size, length_scale)
    n1,n2 = grid_size
    Threads.@threads for i in 1:n1
        for j in 1:n2
            idx1 = (i - 1) * n2 + j  # Flatten the 2D index to 1D
            for k in 1:n1
                @simd for l in 1:n2
                    idx2 = (k - 1) * n2 + l
                    @inbounds cov_matrix[idx1, idx2] = gaussian_covariance(i, j, k, l, length_scale)
                end
            end
        end
    end
end

@views function main_opt()
    # Parameters
    grid_size = (100, 100)  # 100x100 grid
    N         = prod(grid_size)
    correlation_length = 10.0  # Correlation length in grid points
    porosity_threshold = 0.9  # Threshold for fluid/solid classification (0 - fluid, 1 - solid)
    perturbation = 1e-6  # Small value to add to the diagonal for positive definiteness
    sparsity_threshold = 1e-6
    ######################

    # Resolution in FD cells
    nc = (x = grid_size[1], y = grid_size[2])
        
    # Define node types and set BC flags
    type = Fields( fill(:out, (nc.x+2, nc.y+2)) )
    type.u[2:end-1,2:end-1] .= :in
    type.u[1,:]             .= :Dirichlet 
    type.u[end,:]           .= :Dirichlet 
    type.u[:,1]             .= :Dirichlet
    type.u[:,end]           .= :Dirichlet
    
    # 5-point stencil
    pattern = Fields( Fields( @SMatrix([0 1 0; 1 1 1; 0 1 0]) ) )

    # Equation numbering
    number = Fields( fill(0, (nc.x+2, nc.y+2)) )
    Numbering!(number, type, nc)

    # Sparse matrix assembly
    nu  = maximum(number.u)
    M   = Fields( Fields( ExtendableSparseMatrix(nu, nu) ))

    @info "Assembly, ndof  = $(nu)"
    # SparsityPattern!(M, number, pattern, nc)
    # @info "5-point stencil"
    # display(M.u.u)
    # display(M.u.u - M.u.u')

    # 9-point stencil
    pattern = Fields( Fields( @SMatrix([1 1 1; 1 1 1; 1 1 1]) ) )
    SparsityPattern!(M, number, pattern, nc)
    # @info "9-point stencil"
    # display(M.u.u)
    # display(M.u.u - M.u.u') 

    ######################

    # Step 1: Create the covariance matrix in real space (distances between all grid points)
    cov_matrix = zeros(Float64, N, N)

    # Loop over all pairs of grid points and fill in the covariance matrix
    length_scale =  inv(2 * correlation_length^2)
    @time covariance_matrix!(cov_matrix, grid_size, length_scale)

    # Step 2: Add small perturbation to the diagonal to ensure positive definiteness
    @time for i in 1:N
        @inbounds cov_matrix[i, i] += perturbation  # Add perturbation to diagonal
    end



    # cov_matrix = sparse(cov_matrix .* (cov_matrix .< 0.01))
    # cov_matrix = Hermitian((cov_matrix))
    # cov_matrix = M.u.u * cov_matrix

    # Msym = SparseMatrixCSC(M.u.u .* cov_matrix)
    # Msym = 0.5*(Msym .+ Msym')
    # Msym .+= spdiagm(diag(Msym))
    # cov_matrix = Msym

    # Msym = SparseMatrixCSC(M.u.u .* cov_matrix)
    # Msym = 0.5*(Msym .+ Msym')
    # Msym .+= spdiagm(diag(Msym))
    # # cov_matrix = Msym
    # @show minimum(eigen(Matrix(Msym)).values), maximum(eigen(Matrix(Msym)).values)

    # @time Msym = Matrix(M.u.u .* cov_matrix)

    # Step 3: Cholesky decomposition to generate correlated random field
    @time L = cholesky(Matrix(cov_matrix), check=false).L

    # Step 4: Generate uncorrelated Gaussian noise (one random value per grid point)
    uncorrelated_noise = randn(N)

    # Step 5: Generate the correlated random field
    @time correlated_field = L * uncorrelated_noise

    # Step 6: Reshape the 1D correlated field into a 2D matrix
    correlated_field_2d = reshape(correlated_field, grid_size[1], grid_size[2])

    # Step 7: Apply a threshold to create a binary fluid/solid structure
    binary_structure = correlated_field_2d .< porosity_threshold  # True (fluid) or False (solid)

    ϕ = sum(iszero(x) for x in  binary_structure) / (*(size(binary_structure)...))
    @show (*(size(binary_structure)...))
    @show  ϕ

    # Step 8: Visualize the resulting binary porous structure
    heatmap(binary_structure, color=:coolwarm, xlabel="X", ylabel="Y", title="Correlated Porous Media Structure")



    # Msym = SparseMatrixCSC(M.u.u .* cov_matrix)
    # Msym = 0.5*(Msym .+ Msym')
    # Msym .+= spdiagm(diag(Msym))
    # cov_matrix = Msym
    # @show minimum(eigen(Matrix(Msym)).values), maximum(eigen(Matrix(Msym)).values)
    
    # cov_matrix
    # L = cholesky(cov_matrix).L

    # # # Step 3: Cholesky decomposition to generate correlated random field
    # # @time L = cholesky(Msym).L

    # # Step 4: Generate uncorrelated Gaussian noise (one random value per grid point)
    # uncorrelated_noise = randn(N)

    # # Step 5: Generate the correlated random field
    # correlated_field = L * uncorrelated_noise
    
end 

@time main_opt()

