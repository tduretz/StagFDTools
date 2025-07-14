using SparseArrays

function DecoupledSolver(𝐊, 𝐐, 𝐐ᵀ, 𝐏, fu, fp; fact=:chol,  ηb=1e3, niter_l=10, ϵ_l=1e-11, 𝐊_PC=I(size(𝐊,1)))
    
    if nnz(𝐏) == 0 # incompressible limit
        𝐏inv  = ηb .* I(size(𝐏,1))
    else # compressible case
        𝐏inv  = spdiagm(1.0 ./diag(𝐏))
    end
    𝐊sc      = 𝐊    .- 𝐐*(𝐏inv*𝐐ᵀ)
    𝐊sc_PC   = 𝐊_PC .- 𝐐*(𝐏inv*𝐐ᵀ)

    if fact == :chol
        L_PC  = I(size(𝐊sc,1))
        𝐊fact = cholesky(Hermitian(L_PC*𝐊sc), check=false)
    elseif fact == :symchol
        L_PC  = 𝐊sc'
        @time 𝐊fact = cholesky(Hermitian(𝐊sc_PC), check=false)
        @time Ksym = L_PC*𝐊sc
        @time 𝐊fact = cholesky(Hermitian(Ksym), check=false)
    elseif fact == :PCchol
        L_PC  = I(size(𝐊sc,1))
        @time 𝐊fact = cholesky(Hermitian(𝐊sc_PC), check=false)
    elseif fact == :lu
        L_PC  = I(size(𝐊sc,1))
        @time 𝐊fact = lu(L_PC*𝐊sc)
    end
    ru    = zeros(size(𝐊,1))
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
        u    .= 𝐊fact\(L_PC*fusc)

        # # Iterative refinement
        # ϵ_ref = 1e-7
        # for iter_ref=1:10
        #     ru .= 𝐊sc*u .- fusc
        #     @printf("  --> Iterative refinement %02d\n res.   = %2.2e\n", iter_ref, norm(ru)/sqrt(length(ru)))
        #     norm(ru)/sqrt(length(ru)) < ϵ_ref && break
        #     du  = 𝐊fact\(L_PC*ru)
        #     u  .-= du
        # end
   
        p   .+= 𝐏inv*(fp .- 𝐐ᵀ*u .- 𝐏*p)
    end
    return u, p
end