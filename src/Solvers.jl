using SparseArrays

function DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol,  ηb=1e3, niter_l=10, ϵ_l=1e-11)
    if nnz(𝐏) == 0 # incompressible limit
        𝐏inv  = -ηb .* I(size(𝐏,1))
    else # compressible case
        𝐏inv  = spdiagm(1.0 ./diag(𝐏))
    end
    𝐊sc   = 𝐊 .- 𝐐*(𝐏inv*𝐐ᵀ)
    if fact == :chol
        𝐊fact = cholesky(Hermitian(𝐊sc), check=false)
    elseif fact == :lu
        𝐊fact = lu(𝐊sc)
    end
    u     = zeros(size(𝐊,1))
    ru    = zeros(size(𝐊,1))
    fusc  = zeros(size(𝐊,1))
    p     = zeros(size(𝐐,2))
    rp    = zeros(size(𝐐,2))
    # Iterations
    for rit=1:niter_l           
        ru   .= fu .- 𝐊*u  .- 𝐐*p
        rp   .= fp .- 𝐐ᵀ*u .- 𝐏*p
        nrmu, nrmp = norm(ru), norm(rp)
        @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
        if nrmu/sqrt(length(ru)) < ϵ_l && nrmp/sqrt(length(rp)) < ϵ_l
            break
        end
        fusc .= fu  .- 𝐐*(𝐏inv*fp .+ p)
        u    .= 𝐊fact\fusc
        p   .+= 𝐏inv*(fp .- 𝐐ᵀ*u .- 𝐏*p)
    end
    return u, p
end