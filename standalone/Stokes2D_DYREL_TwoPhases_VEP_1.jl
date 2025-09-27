# Initialisation
using GLMakie, Printf, Statistics, LinearAlgebra, MAT
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) ) 

# can be replaced by AD
function Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ, Δx, Δy, ncx  ,ncy)
        
    ηN    = ones(ncx-1, ncy)
    ηS    = ones(ncx-1, ncy)
    ηN[:,1:end-1] .= ηv[2:end-1,2:end-1]
    ηS[:,2:end-0] .= ηv[2:end-1,2:end-1]
    ηW    = ηc[1:end-1,:]
    ηE    = ηc[2:end-0,:]
    ebW   = γ[1:end-1,:] 
    ebE   = γ[2:end-0,:] 
    Cxx   = ones(ncx-1, ncy)
    Cxy   = ones(ncx-1, ncy)
    @. Cxx = abs.(ηN ./ Δy .^ 2) + abs.(ηS ./ Δy .^ 2) + abs.(ebE ./ Δx .^ 2 + (4 // 3) * ηE ./ Δx .^ 2) + abs.(ebW ./ Δx .^ 2 + (4 // 3) * ηW ./ Δx .^ 2) + abs.(-(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx)
    @. Cxy = abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηN ./ (Δx .* Δy)) + abs.(ebE ./ (Δx .* Δy) - 2 // 3 * ηE ./ (Δx .* Δy) + ηS ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηN ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy)) + abs.(ebW ./ (Δx .* Δy) + ηS ./ (Δx .* Δy) - 2 // 3 * ηW ./ (Δx .* Δy))
    
    DVx  = ones(ncx-1, ncy)
    @. DVx .= -(-ηN ./ Δy - ηS ./ Δy) ./ Δy + (ebE ./ Δx + ebW ./ Δx) ./ Δx + ((4 // 3) * ηE ./ Δx + (4 // 3) * ηW ./ Δx) ./ Δx

    ηE    = ones(ncx, ncy-1)
    ηW    = ones(ncx, ncy-1)
    ηE[1:end-1,:] .= ηv[2:end-1,2:end-1]
    ηW[2:end-0,:] .= ηv[2:end-1,2:end-1]
    ηS    = ηc[:,1:end-1]
    ηN    = ηc[:,2:end-0]
    ebS  = γ[:,1:end-1] 
    ebN  = γ[:,2:end-0] 
    Cyy  = ones(ncx, ncy-1)
    Cyx  = ones(ncx, ncy-1)
    @. Cyy = abs.(ηE ./ Δx .^ 2) + abs.(ηW ./ Δx .^ 2) + abs.(ebN ./ Δy .^ 2 + (4 // 3) * ηN ./ Δy .^ 2) + abs.(ebS ./ Δy .^ 2 + (4 // 3) * ηS ./ Δy .^ 2) + abs.((ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx)
    @. Cyx = abs.(ebN ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy)) + abs.(ebN ./ (Δx .* Δy) - 2 // 3 * ηN ./ (Δx .* Δy) + ηW ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) + ηE ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy)) + abs.(ebS ./ (Δx .* Δy) - 2 // 3 * ηS ./ (Δx .* Δy) + ηW ./ (Δx .* Δy))

    DVy  = ones(ncx, ncy-1)
    @. DVy .= (ebN ./ Δy + ebS ./ Δy) ./ Δy + ((4 // 3) * ηN ./ Δy + (4 // 3) * ηS ./ Δy) ./ Δy - (-ηE ./ Δx - ηW ./ Δx) ./ Δx

    λmaxVx = 1.0./DVx .* (Cxx .+ Cxy)
    λmaxVy = 1.0./DVy .* (Cyx .+ Cyy)

    return DVx, DVy, λmaxVx, λmaxVy
end

# 2D Stokes routine
@views function Stokes2D(n)

    sc = (σ = 3e10, L = 1e3, t = 1e10)

    # Time steps
    nt     = 1
    Δt0    = 1e10/sc.t 

    # Physics
    Lx, Ly   = 1.0, 1.0     # domain size
    radi     = 0.2          # inclusion radius
    η0       = 1.0          # viscous viscosity
    ηi       = 1e2          # min/max inclusion viscosity
    εbg      = 1.0          # background strain-rate
    Ωη       = 2.0     
    ϕi       = 1e-3   
    k_ηf0    = 1e-3  
    # Numerics
    ncx, ncy = n*31, n*31   # numerical grid resolution
    ϵ        = 1e-6         # tolerance
    iterMax  = 1e4          # max number of iters
    nout     = 100          # check frequency
    c_fact   = 0.8          # damping factor
    dτ_local = false        # helps a little bit sometimes, sometimes not! 
    γfact    = 20           # penalty: multiplier to the arithmetic mean of η
    rel_drop = 1e-3         # relative drop of velocity residual per PH iteration
    # Preprocessing
    Δx, Δy   = Lx/ncx, Ly/ncy
    # Array initialisation
    ϕ        = ϕi.*ones(ncy  ,ncy  )
    ∇qD      = zeros(ncy  ,ncy  )
    qDx      = zeros(ncy+1,ncy  )
    qDy      = zeros(ncy  ,ncy+1)
    Pt       = zeros(ncx  ,ncy  )
    Pf       = zeros(ncy+2,ncy+2)
    ∇V       = zeros(ncx  ,ncy  )
    Vx       = zeros(ncx+1,ncy+2) 
    Vy       = zeros(ncx+2,ncy+1)
    Exx      = zeros(ncx  ,ncy  )
    Eyy      = zeros(ncx  ,ncy  )
    Exy      = zeros(ncx+1,ncy+1)
    Txx      = zeros(ncx  ,ncy  )
    Tyy      = zeros(ncx  ,ncy  )
    Txy      = zeros(ncx+1,ncy+1)
    RVx      = zeros(ncx-1,ncy  )
    RVy      = zeros(ncx  ,ncy-1)
    RPt      = zeros(ncx  ,ncy  )
    RPf      = zeros(ncx  ,ncy  )
    RVx0     = zeros(ncx-1,ncy  )
    RVy0     = zeros(ncx  ,ncy-1)
    RPf0     = zeros(ncx  ,ncy  )
    dVxdτ    = zeros(ncx-1,ncy  )
    dVydτ    = zeros(ncx  ,ncy-1)
    dPfdτ    = zeros(ncx  ,ncy  )
    βVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    βVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    βPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    cVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    cVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    cPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    αVx      = zeros(ncx-1,ncy  )  # this disappears is dτ is not local
    αVy      = zeros(ncx  ,ncy-1)  # this disappears is dτ is not local
    αPf      = zeros(ncx  ,ncy  )  # this disappears is dτ is not local
    dVx      = zeros(ncx-1,ncy  )  # Could be computed on the fly
    dVy      = zeros(ncx  ,ncy-1)  # Could be computed on the fly
    dPf      = zeros(ncx  ,ncy  )  # Could be computed on the fly
    ηb       = zeros(ncx  ,ncy  )
    ηc       = zeros(ncx  ,ncy  )
    ηv       = zeros(ncx+1,ncy+1)
    ηc_sharp = zeros(ncx  ,ncy  )
    ηv_sharp = zeros(ncx+1,ncy+1)
    # Initialisation
    xce, yce = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, ncx+2), LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, ncy+2)
    xc, yc   = LinRange(-Lx/2+Δx/2, Lx/2-Δx/2, ncx), LinRange(-Ly/2+Δy/2, Ly/2-Δy/2, ncy)
    xv, yv   = LinRange(-Lx/2, Lx/2, ncx+1), LinRange(-Ly/2, Ly/2, ncy+1)
    # Multiple circles with various viscosities
    ηi    = (w=1/ηi, s=ηi) 
    x_inc = [0.0   0.2  -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1 ] 
    y_inc = [0.0   0.4   0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4 ]
    r_inc = [radi  0.09  0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07] 
    η_inc = [ηi.s  ηi.w  ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    ηv_sharp   .= η0
    for inc in 1:1#eachindex(η_inc)
        ηv_sharp[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    ηc_sharp   .= η0
    for inc in 1:1#eachindex(η_inc)
        ηc_sharp[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end  
    # Harmonic averaging mimicking PIC interpolation
    ηc    .= av4_harm(ηv_sharp)
    ηv[2:end-1,2:end-1] .= av4_harm(ηc_sharp)
    # Bulk viscosity
    ηb    .= ηc.*Ωη
    # Select γ
    γi   = γfact*mean(ηc).*ones(size(ηc))
    # (Pseudo-)compressibility
    γ_eff = zeros(size(ηb)) 
    γ_num = γi.*ones(size(ηb)) * 1.0
    γ_phy = ηb.*(1 .- ϕ)
    γ_eff = ((γ_phy.*γ_num)./(γ_phy.+γ_num))
    # Optimal pseudo-time steps - can be replaced by AD
    DVx, DVy, λmaxVx, λmaxVy = Gershgorin_Stokes2D_SchurComplement(ηc, ηv, γ_eff, Δx, Δy, ncx ,ncy)
    DPf    =  1 ./ηb./(1 .- ϕ) .+ 2*k_ηf0*(1/Δx^2 + 1/Δy^2) 
    λmaxPf = (1 ./ηb./(1 .- ϕ) .+ 4*k_ηf0*(1/Δx^2 + 1/Δy^2)) ./ DPf
    # Select dτ
    if dτ_local
        dτVx = 2.0./sqrt.(λmaxVx)*0.99
        dτVy = 2.0./sqrt.(λmaxVy)*0.99
        dτPf = 2.0./sqrt.(λmaxPf)*0.99 
    else
        dτVx = 2.0./sqrt.(maximum(λmaxVx))*0.99 
        dτVy = 2.0./sqrt.(maximum(λmaxVy))*0.99
        dτPf = 2.0./sqrt.(maximum(λmaxPf))*0.99 
    end
    βVx   .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
    βVy   .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
    βPf   .= 2 .* dτPf ./ (2 .+ cPf.*dτPf)
    αVx   .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
    αVy   .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
    αPf   .= (2 .- cPf.*dτPf) ./ (2 .+ cPf.*dτPf)
    # Initial condition
    Vx     .=   εbg.*xv .+    0*yce'
    Vy     .=     0*xce .- εbg.*yv'
    Vx[2:end-1,:]       .= 0   # ensure non zero initial pressure residual
    Vy[:,2:end-1]       .= 0   # ensure non zero initial pressure residual
    Pf[2:end-1,2:end-1] .= 1e-3
    # Iteration loop
    errVx0 = 1.0;  errVy0 = 1.0;  errPf0 = 1.0;   errPt0 = 1.0 
    errVx00= 1.0;  errVy00= 1.0;  errPf00= 1.0 
    iter=1; err=2*ϵ; err_evo_V=[]; err_evo_Pt=[]; err_evo_Pf=[]; err_evo_it=[]
    @time for itPH = 1:50
        # Boundaries
        Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
        Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
        Pf[1,:] .= Pf[2,:]; Pf[end,:] .= Pf[end-1,:] 
        Pf[:,1] .= Pf[:,2]; Pf[:,end] .= Pf[:,end-1]
        # Darcy flux divergence
        qDx    .= -k_ηf0 .* diff(Pf[:,2:end-1], dims=1)/Δx
        qDy    .= -k_ηf0 .* diff(Pf[2:end-1,:], dims=2)/Δy
        ∇qD    .= diff(qDx, dims=1)/Δx .+ diff(qDy, dims=2)/Δy
        # Divergence
        ∇V    .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
        # Deviatoric strain rate
        Exx   .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
        Eyy   .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
        Exy   .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
        # Deviatoric stress
        Txx   .= 2.0.*ηc.*Exx
        Tyy   .= 2.0.*ηc.*Eyy
        Txy   .= 2.0.*ηv.*Exy 
        # Residuals
        RVx    .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
        RVy    .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
        RPt    .= (.-∇V  .- (Pt.-Pf[2:end-1,2:end-1])./ηb./(1.0.-ϕ))  
        RPf    .= (.-∇qD .+ (Pt.-Pf[2:end-1,2:end-1])./ηb./(1.0.-ϕ))
        # Residual check
        errVx = norm(RVx); errVy = norm(RVy); errPt = norm(RPt); errPf = norm(RPf)
        if itPH==1 errVx0=errVx; errVy0=errVy; errPt0=errPt; errPf0=errPf; end
        err = maximum([errVx/errVx0, errVy/errVy0, errPt/errPt0, errPf/errPf0])
        @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[RVx=%1.3e, RVy=%1.3e, RPt=%1.3e, RPf=%1.3e] \n", itPH, iter, iter/ncx, err, errVx/errVx0, errVy/errVy0, errPt/errPt0, errPf/errPf0)
        if (err<ϵ) break end
        # Set tolerance of velocity solve proportional to residual
        ϵ_vel = err*rel_drop
        itPT = 0.
        while (err>ϵ_vel && itPT<=iterMax)
            itPT     += 1
            itg      = iter
            # Pseudo-old dudes 
            RVx0    .= RVx
            RVy0    .= RVy
            RPf0    .= RPf
            # Boundaries
            Vx[:,1] .= Vx[:,2]; Vx[:,end] .= Vx[:,end-1]
            Vy[1,:] .= Vy[2,:]; Vy[end,:] .= Vy[end-1,:]
            Pf[1,:] .= Pf[2,:]; Pf[end,:] .= Pf[end-1,:] 
            Pf[:,1] .= Pf[:,2]; Pf[:,end] .= Pf[:,end-1]
            # Darcy flux divergence
            qDx     .= -k_ηf0 .* diff(Pf[:,2:end-1], dims=1)/Δx
            qDy     .= -k_ηf0 .* diff(Pf[2:end-1,:], dims=2)/Δy
            ∇qD     .= diff(qDx, dims=1)/Δx .+ diff(qDy, dims=2)/Δy
            # Divergence 
            ∇V      .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .+ (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy
            # Deviatoric strain rate
            Exx     .= (Vx[2:end,2:end-1] .- Vx[1:end-1,2:end-1])./Δx .- 1.0/3.0.*∇V
            Eyy     .= (Vy[2:end-1,2:end] .- Vy[2:end-1,1:end-1])./Δy .- 1.0/3.0.*∇V
            Exy     .= 0.5.*((Vx[:,2:end] .- Vx[:,1:end-1])./Δy .+ (Vy[2:end,:] .- Vy[1:end-1,:])./Δx)
            # "Deviatoric" stress
            RPt     .= .-∇V  .- (Pt.-Pf[2:end-1,2:end-1])./ηb./(1.0.-ϕ)
            Txx     .= 2.0.*ηc.*Exx .- γ_eff .* RPt  
            Tyy     .= 2.0.*ηc.*Eyy .- γ_eff .* RPt  
            Txy     .= 2.0.*ηv.*Exy 
            # Residuals
            RVx     .= (1.0./DVx).*(.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
            RVy     .= (1.0./DVy).*(.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
            RPf     .= (1.0./DPf).*(.-∇qD .+ (Pt.-Pf[2:end-1,2:end-1])./ηb./(1.0.-ϕ))
            # Damping-pong
            dVxdτ   .= αVx.*dVxdτ .+ RVx
            dVydτ   .= αVy.*dVydτ .+ RVy
            dPfdτ   .= αPf.*dPfdτ .+ RPf
            # PT updates
            Vx[2:end-1,2:end-1] .+= dVxdτ.*βVx.*dτVx 
            Vy[2:end-1,2:end-1] .+= dVydτ.*βVy.*dτVy 
            Pf[2:end-1,2:end-1] .+= dPfdτ.*βPf.*dτPf 
            # Residual check
            if mod(iter, nout)==0
                errVx = norm(DVx.*RVx); errVy = norm(DVy.*RVy); errPf = norm(DPf.*RPf)
                if iter==nout errVx00=errVx; errVy00=errVy; errPf00=errPf; end
                err = maximum([errVx./errVx00, errVy./errVy00, errPf./errPf00])
                push!(err_evo_V, errVx/errVx00); push!(err_evo_Pt, errPt/errPt0); push!(err_evo_Pf, errPf/errPf0); push!(err_evo_it, itg)
                dVx .= dVxdτ.*βVx.*dτVx
                dVy .= dVydτ.*βVy.*dτVy
                dPf .= dPfdτ.*βPf.*dτPf 
                # @printf("it = %d, iter = %d, err = %1.3e norm[RVx=%1.3e, RVy=%1.3e] \n", it, iter, err, norm_Rx, norm_Ry)
                # λminV  = abs.((sum(dVx.*(RVx .- RVx0))) + abs.((sum(dVy.*(RVy .- RVy0))) )/ ( sum(dVx.*dVx)) + sum(dVy.*dVy) ) 
                λminV  = abs( sum(dVx.*(RVx .- RVx0)) + sum(dVy.*(RVy .- RVy0))  ) / (sum(dVx.*dVx) .+ sum(dVy.*dVy))
                λminPf = abs( sum(dPf.*(RPf .- RPf0))) / sum(dPf.*dPf)
                cVx .= 2*sqrt.(λminV )*c_fact
                cVy .= 2*sqrt.(λminV )*c_fact
                cPf .= 2*sqrt.(λminPf)*c_fact
                βVx .= 2 .* dτVx ./ (2 .+ cVx.*dτVx)
                βVy .= 2 .* dτVy ./ (2 .+ cVy.*dτVy)
                βPf .= 2 .* dτPf ./ (2 .+ cPf.*dτPf)
                αVx .= (2 .- cVx.*dτVx) ./ (2 .+ cVx.*dτVx)
                αVy .= (2 .- cVy.*dτVy) ./ (2 .+ cVy.*dτVy)
                αPf .= (2 .- cPf.*dτPf) ./ (2 .+ cPf.*dτPf)
            end
            iter += 1 
        end
        Pt .+= γ_eff.*RPt
    end

      # Visualise
        function figure()
            fig  = Figure(fontsize = 20, size = (900, 600) )    
            step = 10
            ftsz = 15
            eps  = 1e-10

            # ax   = Axis(fig[1,1], aspect=DataAspect(), title=L"$$Strain", xlabel=L"x", ylabel=L"y")
            # # field = log10.((λ̇.c[inx_c,iny_c] .+ eps)/sc.t )
            # field = log10.(εp[inx_c,iny_c])
            # hm = heatmap!(ax, X.c.x, X.c.y, field, colormap=:jet, colorrange=(-3, -2.3))
            # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # hidexdecorations!(ax)
            # Colorbar(fig[2, 1], hm, label = L"$\lambda$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            # # arrows2d!(ax, X.c.x[1:step:end], X.c.y[1:step:end], Vxsc[1:step:end,1:step:end], Vysc[1:step:end,1:step:end], lengthscale=10000.4, color=:white)

            # ax    = Axis(fig[3,1], aspect=DataAspect(), title=L"$$Porosity", xlabel=L"x", ylabel=L"y")
            # field = Φ.c[inx_c,iny_c]
            # hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # hidexdecorations!(ax)
            # Colorbar(fig[4, 1], hm, label = L"$\dot\lambda$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            # ax    = Axis(fig[1,2], aspect=DataAspect(), title=L"$P^t$ [MPa]", xlabel=L"x", ylabel=L"y")
            # field = (P.t)[inx_c,iny_c].*sc.σ./1e6 
            # hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:jet, colorrange=(-6, 4))
            # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # hidexdecorations!(ax)
            # Colorbar(fig[2, 2], hm, label = L"$P^t$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            # # arrows2d!(ax, X.c.x[1:step:end], X.c.y[1:step:end], Vxsc[1:step:end,1:step:end], Vysc[1:step:end,1:step:end], lengthscale=10000.4, color=:white)

            # ax    = Axis(fig[3,2], aspect=DataAspect(), title=L"$P^e - \tau$", xlabel=L"P^e", ylabel=L"\tau")
                  
            # (materials.single_phase) ? α1 = 0.0 : α1 = 1.0 
            # Pe    = (P.t .- α1*P.f)[inx_c,iny_c].*sc.σ

            # τII       = (τ.II)[inx_c,iny_c].*sc.σ
            # P_ax      = LinRange(-5e6, 5e6, 100)
            # τ_ax_rock = materials.C[1]*sc.σ*materials.cosϕ[1] .+ P_ax.*materials.sinϕ[1]
            # lines!(ax, P_ax/1e6, τ_ax_rock/1e6, color=:black)
            # scatter!(ax, Pe[:]/1e6, τII[:]/1e6, color=:black )
            # F_post = @. τ.II - materials.C[1]*materials.cosϕ[1] - (P.t .- α1*P.f)*materials.sinϕ[1]
            # maxF   =  maximum( F_post[inx_c,iny_c] )
            # @info maxF, maxF .*sc.σ /1e6
            # @show maximum(τ.f[inx_c,iny_c]),  maximum(τ.f[inx_c,iny_c]) .*sc.σ /1e6

            # # # Previous stress states
            # # τxyc0 = av2D(τ0.xy)
            # # τII0  = sqrt.( 0.5.*(τ0.xx[inx_c,iny_c].^2 + τ0.yy[inx_c,iny_c].^2 + (-τ0.xx[inx_c,iny_c]-τ0.yy[inx_c,iny_c]).^2) .+ τxyc0[inx_c,iny_c].^2 )
            # # Pe    = (P0.t .- α1*P0.f)[inx_c,iny_c].*sc.σ
            # # τII   = τII0.*sc.σ
            # # scatter!(ax, Pe[:]/1e6, τII[:]/1e6, color=:gray )

            # # ax    = Axis(fig[1,3], aspect=DataAspect(), title=L"$\tau_\text{II}$ [MPa]", xlabel=L"x", ylabel=L"y")
            # # field = (τ.II)[inx_c,iny_c].*sc.σ./1e6
            # # hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # # hidexdecorations!(ax)
            # # Colorbar(fig[2, 3], hm, label = L"$\tau_\text{II}$", height=20, width = 200, labelsize = ftsz, ticklabelsize = ftsz, vertical=false, valign=true, flipaxis = true )
            
            # # ax  = Axis(fig[3,3], xlabel="Iterations @ step $(it) ", ylabel="log₁₀ error")
            # # scatter!(ax, 1:niter, log10.(err.x[1:niter]./err.x[1]) )
            # # scatter!(ax, 1:niter, log10.(err.y[1:niter]./err.x[1]) )
            # # scatter!(ax, 1:niter, log10.(err.pt[1:niter]./err.pt[1]) )
            # # scatter!(ax, 1:niter, log10.(err.pf[1:niter]./err.pf[1]) )
            # # ylims!(ax, -10, 1.1)

            # ax  = Axis(fig[1,3], xlabel="Strain", ylabel="Mean pressure")
            # lines!(  ax, data["strvec"][1:end], data["Pvec"][1:end] )
            # scatter!(ax, probes.str[1:2:nt], probes.Pt[1:2:nt] )

            # ax  = Axis(fig[3,3], xlabel="Strain", ylabel="Mean stress invariant")
            # lines!(  ax, data["strvec"][1:end], data["Tiivec"][1:end] )
            # scatter!(ax, probes.str[1:2:nt], probes.τ[1:2:nt] )

            # # field = P.f.*sc.σ
            # # hm    = heatmap!(ax, X.c.x, X.c.y, field, colormap=:bluesreds, colorrange=(minimum(field)-eps, maximum(field)+eps))
            # # contour!(ax, X.c.x, X.c.y,  phases.c[inx_c,iny_c], color=:black)
            # # hidexdecorations!(ax)
            # # Colorbar(fig[4, 2], hm, label = L"$P^f$", height=20, width = 200, labelsize = 20, ticklabelsize = 20, vertical=false, valign=true, flipaxis = true )
            
            display(fig) 
        end
        with_theme(figure, theme_latexfonts())
    
    return
end

n = 4
Stokes2D(n)
