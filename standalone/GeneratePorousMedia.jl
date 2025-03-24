using Random
using LinearAlgebra
using Plots

# Adapted from ChatGPT

# Function to compute Gaussian covariance between points based on distance
@inline function gaussian_covariance(x1, y1, x2, y2, length_scale)
    return exp(-((x1 - x2)^2 + (y1 - y2)^2) / (2 * length_scale^2))
end

@views function main()
    # Parameters
    grid_size = (100, 100)  # 100x100 grid
    correlation_length = 1.0  # Correlation length in grid points
    porosity_threshold = 1.9  # Threshold for fluid/solid classification (0 - fluid, 1 - solid)
    perturbation = 1e-6  # Small value to add to the diagonal for positive definiteness

    # Step 1: Create the covariance matrix in real space (distances between all grid points)
    cov_matrix = zeros(Float64, grid_size[1] * grid_size[2], grid_size[1] * grid_size[2])

    # Loop over all pairs of grid points and fill in the covariance matrix
    for i in 1:grid_size[1]
        for j in 1:grid_size[2]
            for k in 1:grid_size[1]
                for l in 1:grid_size[2]
                    idx1 = (i - 1) * grid_size[2] + j  # Flatten the 2D index to 1D
                    idx2 = (k - 1) * grid_size[2] + l
                    cov_matrix[idx1, idx2] = gaussian_covariance(i, j, k, l, correlation_length)
                end
            end
        end
    end

    # Step 2: Add small perturbation to the diagonal to ensure positive definiteness
    for i in 1:(grid_size[1] * grid_size[2])
        cov_matrix[i, i] += perturbation  # Add perturbation to diagonal
    end

    # Step 3: Cholesky decomposition to generate correlated random field
    L = cholesky(cov_matrix).L

    # Step 4: Generate uncorrelated Gaussian noise (one random value per grid point)
    uncorrelated_noise = randn(grid_size[1] * grid_size[2])

    # Step 5: Generate the correlated random field
    correlated_field = L * uncorrelated_noise

    # Step 6: Reshape the 1D correlated field into a 2D matrix
    correlated_field_2d = reshape(correlated_field, grid_size[1], grid_size[2])

    # Step 7: Apply a threshold to create a binary fluid/solid structure
    binary_structure = correlated_field_2d .< porosity_threshold  # True (fluid) or False (solid)

    # Step 8: Visualize the resulting binary porous structure
    heatmap(binary_structure, color=:coolwarm, xlabel="X", ylabel="Y", title="Correlated Porous Media Structure")

    ϕ = sum(binary_structure.==0) / (*(size(binary_structure)...))
    @show (*(size(binary_structure)...))
    @show  ϕ
end

main()
