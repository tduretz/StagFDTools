using Plots, LinearAlgebra

function Residual!(R, Rp, V, Pt, ε̇, τ, η, Δ)
    @. V.x[:,[1 end]] .= V.x[:,[2 end-1]]
    @. V.y[[1 end],:] .= V.y[[2 end-1],:]
    @. ε̇.kk = (V.x[2:end-0,2:end-1] - V.x[1:end-1,2:end-1]) / Δ.x + (V.y[2:end-1,2:end-0] - V.y[2:end-1,1:end-1]) / Δ.y
    @. ε̇.xx = (V.x[2:end-0,2:end-1] - V.x[1:end-1,2:end-1]) / Δ.x - 1/3*ε̇.kk 
    @. ε̇.yy = (V.y[2:end-1,2:end-0] - V.y[2:end-1,1:end-1]) / Δ.y - 1/3*ε̇.kk 
    @. ε̇.xy = 1/2 * ((V.x[:,2:end-0] - V.x[:,1:end-1]) / Δ.y + (V.y[2:end-0,:] - V.y[1:end-1,:]) / Δ.x)
    @. τ.xx = 2 * η.p  * ε̇.xx
    @. τ.yy = 2 * η.p  * ε̇.yy
    @. τ.xy = 2 * η.xy * ε̇.xy
    @. R.x[2:end-1,2:end-1] = (τ.xx[2:end,:] - τ.xx[1:end-1,:]) / Δ.x - (Pt[2:end,:] - Pt[1:end-1,:]) / Δ.x + (τ.xy[2:end-1,2:end] - τ.xy[2:end-1,1:end-1]) / Δ.y
    @. R.y[2:end-1,2:end-1] = (τ.yy[:,2:end] - τ.yy[:,1:end-1]) / Δ.y - (Pt[:,2:end] - Pt[:,1:end-1]) / Δ.y + (τ.xy[2:end,2:end-1] - τ.xy[1:end-1,2:end-1]) / Δ.x
    @. Rp = ε̇.kk
    return nothing
end
    
@views function (@main)(nc)

    size_x, size_y, size_p, size_xy = (nc.x+1, nc.y+2), (nc.x+2, nc.y+1), (nc.x, nc.y), (nc.x+1, nc.y+1)

    # Intialise field
    L   = (x=10.0, y=10.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y)
    R   = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_p...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    ε̇   = (xx=zeros(size_p...), yy=zeros(size_p...), kk=zeros(size_p...), xy=zeros(size_xy...))
    τ   = (xx=zeros(size_p...), yy=zeros(size_p...), xy=zeros(size_xy...))
    η   = (x= ones(size_x...), y= ones(size_y...), p=ones(size_p...), xy=zeros(size_xy...) )
    Rp  = zeros(size_p...)
    Pt  = zeros(size_p...)
    xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc  = LinRange(-L.x/2-Δ.x/2, L.x/2+Δ.x/2, nc.x+2)
    yc  = LinRange(-L.y/2-Δ.y/2, L.y/2+Δ.y/2, nc.y+2)

    # Initial configuration
    ε̇bg  = -1.0
    V.x .= ε̇bg*xv .+ 0*yc' 
    V.y .= 0*xc .-  ε̇bg*yv' 

    η0       = 1.0e-3
    η1       = 1.0
    ηi    = (s=min(η0,η1), w=1/min(η0,η1)) 
    x_inc = [0.0       0.2  -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1] *10
    y_inc = [0.0       0.4   0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4] *10
    r_inc = [0.2       0.09  0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07]*10
    η_inc = [ηi.s      ηi.w  ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    
    for i in eachindex(η_inc)
        η.x[((xv.-x_inc[i]).^2 .+ (yc'.-y_inc[i]).^2) .<= r_inc[i]^2] .= η_inc[i] 
        η.y[((xc.-x_inc[i]).^2 .+ (yv'.-y_inc[i]).^2) .<= r_inc[i]^2] .= η_inc[i]
    end
    η.p  .= 0.25.*(η.x[1:end-1,2:end-1].+η.x[2:end-0,2:end-1].+η.y[2:end-1,1:end-1].+η.y[2:end-1,2:end-0])
    η.xy .= 0.25.*(η.y[1:end-1,:].+η.y[2:end-0,:].+η.x[:,1:end-1].+η.x[:,2:end-0])

    # Diagonal preconditioner
    D    = (x=ones(size_x...), y=ones(size_y...), p=ones(size_p...))
    dx, dy = Δ.x, Δ.y
    etaW, etaE = η.p[1:end-1,:], η.p[2:end-0,:]
    etaS, etaN = η.xy[2:end-1,1:end-1], η.xy[2:end-1,2:end-0]
    etaS[:,1]   .= 0.0
    etaN[:,end] .= 0.0
    D.x[2:end-1,2:end-1] .= (-etaN ./ dy - etaS ./ dy) ./ dy + (-4 // 3 * etaE ./ dx - 4 // 3 * etaW ./ dx) ./ dx
    etaW, etaE = η.xy[1:end-1,2:end-1],  η.xy[2:end-0,2:end-1] 
    etaS, etaN = η.p[:,1:end-1], η.p[:,2:end-0] 
    etaW[1,:]   .= 0.0
    etaE[end,:] .= 0.0
    D.y[2:end-1,2:end-1] .= (-4 // 3 * etaN ./ dy - 4 // 3 * etaS ./ dy) ./ dy + (-etaE ./ dx - etaW ./ dx) ./ dx
    D.p .= max(nc...) ./ η.p

    # Initial residual
    Residual!(R, Rp, V, Pt, ε̇, τ, η, Δ)

    # Arrays for solver 
    dV   = (x=zeros(size_x...), y=zeros(size_y...)); dPt  = zeros(size_p...)
    Ap   = (x=zeros(size_x...), y=zeros(size_y...)); Ap_p = zeros(size_p...)
    z    = (x=zeros(size_x...), y=zeros(size_y...)); z_p  = zeros(size_p...)
    p    = (x=zeros(size_x...), y=zeros(size_y...)); p_p  = zeros(size_p...)
    
    # Initial residual and preconditioned residual
    z.x  .= (1 ./D.x).*R.x; z.y  .= (1 ./D.y).*R.y; z_p   .= (1 ./D.p).*Rp
    p.x  .= z.x;            p.y .= z.y;             p_p   .= z_p
    
    # Initialize residual and preconditioned residual
    norm_r0 = sqrt(sum(R.x.*R.x) + sum(R.y.*R.y) + sum(Rp.*Rp)) 
    @show norm_r0
    
    max_iter = 3*nc.x*nc.y 
    tol      = 1e-8
    err      = zeros(max_iter)
    iter     = 0
    α, β     = 0., 0.
    nout     = 1

    # Iteration loop
    for k in 1:max_iter

        iter+=1

        # Compute A * p
        Residual!(Ap, Ap_p, p, p_p, ε̇, τ, η, Δ)

        # Compute step size α
        if k==1 || mod(k, nout)==0
            r_dot_z = (dot(R.x, z.x) + dot(R.y, z.y) + dot(Rp, z_p))
            α   = r_dot_z / (dot(p.x, Ap.x) + dot(p.y, Ap.y) + dot(p_p, Ap_p) )
        end

        # Update the solution vector x
        dV.x .+= α .* p.x
        dV.y .+= α .* p.y
        dPt  .+= α .* p_p
        
        # Compute residual
        R.x .-= α .* Ap.x
        R.y .-= α .* Ap.y
        Rp  .-= α .* Ap_p
        norm_r_new = sqrt(sum(R.x.*R.x) + sum(R.y.*R.y) + sum(Rp.*Rp)) 
        err[iter]  = norm_r_new / norm_r0
        
        # Check for convergence
        if norm_r_new / norm_r0 < tol  #|| norm_r_new/sqrt(n) < 2*tol 
            println("Converged in $(k/max(nc...)) it/nx.")
            break
        end
        
        # Apply preconditioner
        z.x  .= (1 ./D.x).*R.x; z.y  .= (1 ./D.y).*R.y; z_p   .= (1 ./D.p).*Rp
        
        # Compute β for direction update
        if k==1 || mod(k, nout)==0
            β = (dot(R.x, z.x) + dot(R.y, z.y) + dot(Rp, z_p)) / r_dot_z
        end
        # Update direction p and residual r
        p.x .= z.x .+ β .* p.x
        p.y .= z.y .+ β .* p.y
        p_p .= z_p .+ β .* p_p
    end
    V.x .+= dV.x
    V.y .+= dV.y
    Pt  .+= dPt
    p1 = heatmap(xv, yc, V.x', aspect_ratio=1, xlim=extrema(xv), title="Vx")
    p2 = heatmap(xc, yv, V.y', aspect_ratio=1, xlim=extrema(xv), title="Vy")
    p3 = heatmap(xc[2:end-1], yc[2:end-1], Pt', aspect_ratio=1, xlim=extrema(xv), title="Pt")
    p4 = plot(1:iter, log10.(err[1:iter]), label="", title="Error")
    display(plot(p1, p2, p3, p4))
    return iter/max(nc...)
end

main( (x=1*30, y=1*32) )
main( (x=2*30, y=2*32) )
main( (x=4*30, y=4*32) )
main( (x=8*30, y=8*32) )