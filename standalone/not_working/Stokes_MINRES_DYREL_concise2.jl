using Plots, LinearAlgebra

function Residual!(R, Rp, V, P, ε̇, τ, η, Δ)
    @. V.x[:,[1 end]] .= V.x[:,[2 end-1]]
    @. V.y[[1 end],:] .= V.y[[2 end-1],:]
    @. ε̇.kk = (V.x[2:end-0,2:end-1] - V.x[1:end-1,2:end-1]) / Δ.x + (V.y[2:end-1,2:end-0] - V.y[2:end-1,1:end-1]) / Δ.y
    @. ε̇.xx = (V.x[2:end-0,2:end-1] - V.x[1:end-1,2:end-1]) / Δ.x - 1/3*ε̇.kk 
    @. ε̇.yy = (V.y[2:end-1,2:end-0] - V.y[2:end-1,1:end-1]) / Δ.y - 1/3*ε̇.kk 
    @. ε̇.xy = 1/2 * ((V.x[:,2:end-0] - V.x[:,1:end-1]) / Δ.y + (V.y[2:end-0,:] - V.y[1:end-1,:]) / Δ.x)
    @. τ.xx = 2 * η.p  * ε̇.xx
    @. τ.yy = 2 * η.p  * ε̇.yy
    @. τ.xy = 2 * η.xy * ε̇.xy
    @. R.x[2:end-1,2:end-1] = -((τ.xx[2:end,:] - τ.xx[1:end-1,:]) / Δ.x - (P[2:end,:] - P[1:end-1,:]) / Δ.x + (τ.xy[2:end-1,2:end] - τ.xy[2:end-1,1:end-1]) / Δ.y)
    @. R.y[2:end-1,2:end-1] = -((τ.yy[:,2:end] - τ.yy[:,1:end-1]) / Δ.y - (P[:,2:end] - P[:,1:end-1]) / Δ.y + (τ.xy[2:end,2:end-1] - τ.xy[1:end-1,2:end-1]) / Δ.x)
    @. Rp = -ε̇.kk
    return nothing
end
    
# @views function (@main)(nc)

#     size_x, size_y, size_c, size_xy = (nc.x+1, nc.y+2), (nc.x+2, nc.y+1), (nc.x, nc.y), (nc.x+1, nc.y+1)

#     # Intialise field
#     L   = (x=10.0, y=10.0)
#     Δ   = (x=L.x/nc.x, y=L.y/nc.y)
#     R   = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
#     V   = (x=zeros(size_x...), y=zeros(size_y...))
#     ε̇   = (xx=zeros(size_c...), yy=zeros(size_c...), kk=zeros(size_c...), xy=zeros(size_xy...))
#     τ   = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_xy...))
#     η   = (x= ones(size_x...), y= ones(size_y...), p=ones(size_c...), xy=zeros(size_xy...) )
#     Rp  = zeros(size_c...)
#     Pt  = zeros(size_c...)
#     xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
#     yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
#     xc  = LinRange(-L.x/2-Δ.x/2, L.x/2+Δ.x/2, nc.x+2)
#     yc  = LinRange(-L.y/2-Δ.y/2, L.y/2+Δ.y/2, nc.y+2)

#     # Initial configuration
#     ε̇bg  = -1.0
#     V.x .= ε̇bg*xv .+ 0*yc' 
#     V.y .= 0*xc .-  ε̇bg*yv' 

#     η0       = 1.0e-3
#     η1       = 1.0
#     ηi    = (w=min(η0,η1), s=1/min(η0,η1)) 
#     x_inc = [0.0       0.2  -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1] *10
#     y_inc = [0.0       0.4   0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4] *10
#     r_inc = [0.2       0.09  0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07]*10
#     η_inc = [ηi.s      ηi.w  ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    
#     for i in eachindex(η_inc)
#         η.x[((xv.-x_inc[i]).^2 .+ (yc'.-y_inc[i]).^2) .<= r_inc[i]^2] .= η_inc[i] 
#         η.y[((xc.-x_inc[i]).^2 .+ (yv'.-y_inc[i]).^2) .<= r_inc[i]^2] .= η_inc[i]
#     end
#     η.p  .= 0.25.*(η.x[1:end-1,2:end-1].+η.x[2:end-0,2:end-1].+η.y[2:end-1,1:end-1].+η.y[2:end-1,2:end-0])
#     η.xy .= 0.25.*(η.y[1:end-1,:].+η.y[2:end-0,:].+η.x[:,1:end-1].+η.x[:,2:end-0])

#     # Diagonal preconditioner
#     D    = (x=ones(size_x...), y=ones(size_y...), p=ones(size_c...))
#     dx, dy = Δ.x, Δ.y
#     etaW, etaE = η.p[1:end-1,:], η.p[2:end-0,:]
#     etaS, etaN = η.xy[2:end-1,1:end-1], η.xy[2:end-1,2:end-0]
#     etaS[:,1]   .= 0.0
#     etaN[:,end] .= 0.0
#     D.x[2:end-1,2:end-1] .= (-etaN ./ dy - etaS ./ dy) ./ dy + (-4 // 3 * etaE ./ dx - 4 // 3 * etaW ./ dx) ./ dx
#     etaW, etaE = η.xy[1:end-1,2:end-1],  η.xy[2:end-0,2:end-1] 
#     etaS, etaN = η.p[:,1:end-1], η.p[:,2:end-0] 
#     etaW[1,:]   .= 0.0
#     etaE[end,:] .= 0.0
#     D.y[2:end-1,2:end-1] .= (-4 // 3 * etaN ./ dy - 4 // 3 * etaS ./ dy) ./ dy + (-etaE ./ dx - etaW ./ dx) ./ dx
#     D.p .= max(nc...) ./ η.p

#     # Initial residual
#     Residual!(R, Rp, V, Pt, ε̇, τ, η, Δ)

#     # Arrays for solver 
#     A∂V∂τ   = (x=zeros(size_x...), y=zeros(size_y...)); A∂P∂τ = zeros(size_c...)
#     ∂V∂τ = (x=zeros(size_x...), y=zeros(size_y...)); ∂P∂τ  = zeros(size_c...)
    
#     # Initial residual and preconditioned residual
#     ∂V∂τ.x  .= (1 ./D.x).*R.x;  ∂V∂τ.y .= (1 ./D.y).*R.y;  ∂P∂τ   .= (1 ./D.p).*Rp
    
#     # Initialize residual and preconditioned residual
#     norm_r0 = sqrt(sum(R.x.*R.x) + sum(R.y.*R.y) + sum(Rp.*Rp)) 
#     @show norm_r0
    
#     max_iter = 3*nc.x*nc.y 
#     tol      = 1e-8
#     err      = zeros(max_iter)
#     iter     = 0
#     α, β     = 0., 0.
#     nout     = 1

#     # Iteration loop
#     for k in 1:max_iter

#         iter+=1

#         # Compute A * ∂V∂τ
#         Residual!(A∂V∂τ, A∂P∂τ, ∂V∂τ, ∂P∂τ, ε̇, τ, η, Δ)

#         # Compute pseudo time step α
#         if k==1 || mod(k, nout)==0
#             r_dot_z = (dot(R.x, (1 ./D.x).*R.x) + dot(R.y, (1 ./D.y).*R.y) + dot(Rp, (1 ./D.p).*Rp))
#             α = r_dot_z / (dot(∂V∂τ.x, A∂V∂τ.x) + dot(∂V∂τ.y, A∂V∂τ.y) + dot(∂P∂τ, A∂P∂τ) )
#         end

#         # Update coupled solution
#         V.x .+= α .* ∂V∂τ.x
#         V.y .+= α .* ∂V∂τ.y
#         Pt  .+= α .* ∂P∂τ
        
#         # Compute residual
#         Residual!(R, Rp, V, Pt, ε̇, τ, η, Δ)
#         # R.x .-= α .* A∂V∂τ.x
#         # R.y .-= α .* A∂V∂τ.y
#         # Rp  .-= α .* A∂P∂τ
#         norm_r_new = sqrt(sum(R.x.*R.x) + sum(R.y.*R.y) + sum(Rp.*Rp)) 
#         err[iter]  = norm_r_new / norm_r0
        
#         # Check for convergence
#         if norm_r_new / norm_r0 < tol  #|| norm_r_new/sqrt(n) < 2*tol 
#             println("Converged in $(k/max(nc...)) it/nx.")
#             break
#         end
        
#         # Compute damping factor β for direction update
#         if k==1 || mod(k, nout)==0
#             β = (dot(R.x, (1 ./D.x).*R.x) + dot(R.y, (1 ./D.y).*R.y) + dot(Rp, (1 ./D.p).*Rp)) / r_dot_z
#         end

#         # Update direction p and residual r
#         ∂V∂τ.x .= (1 ./D.x).*R.x .+ β .* ∂V∂τ.x
#         ∂V∂τ.y .= (1 ./D.y).*R.y .+ β .* ∂V∂τ.y
#         ∂P∂τ   .= (1 ./D.p).*Rp  .+ β .* ∂P∂τ
#     end
#     # Visualise
#     p1 = heatmap(xv, yc, V.x', aspect_ratio=1, xlim=extrema(xv), title="Vx")
#     p2 = heatmap(xc, yv, V.y', aspect_ratio=1, xlim=extrema(xv), title="Vy")
#     p3 = heatmap(xc[2:end-1], yc[2:end-1], Pt', aspect_ratio=1, xlim=extrema(xv), title="Pt", clim=(-10, 10))
#     p4 = plot(1:iter, log10.(err[1:iter]), label="", title="Error")
#     display(plot(p1, p2, p3, p4))
#     return iter/max(nc...)
# end

@views function (@main)(nc)

    size_x, size_y, size_c, size_xy = (nc.x+1, nc.y+2), (nc.x+2, nc.y+1), (nc.x, nc.y), (nc.x+1, nc.y+1)

    # Intialise field
    L   = (x=10.0, y=10.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y)
    R   = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    R0  = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    R1  = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))

    V   = (x=zeros(size_x...), y=zeros(size_y...))
    ε̇   = (xx=zeros(size_c...), yy=zeros(size_c...), kk=zeros(size_c...), xy=zeros(size_xy...))
    τ   = (xx=zeros(size_c...), yy=zeros(size_c...), xy=zeros(size_xy...))
    η   = (x= ones(size_x...), y= ones(size_y...), p=ones(size_c...), xy=zeros(size_xy...) )
    Rp  = zeros(size_c...)
    Rp0 = zeros(size_c...)
    Rp1 = zeros(size_c...)

    P   = zeros(size_c...)
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
    ηi    = (w=min(η0,η1), s=1/min(η0,η1)) 
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
    D    = (x=ones(size_x...), y=ones(size_y...), p=ones(size_c...))
    G    = (x=ones(size_x...), y=ones(size_y...), p=ones(size_c...))
    dx, dy = Δ.x, Δ.y
    etaW, etaE = η.p[1:end-1,:], η.p[2:end-0,:]
    etaS, etaN = η.xy[2:end-1,1:end-1], η.xy[2:end-1,2:end-0]
    etaS[:,1]   .= 0.0
    etaN[:,end] .= 0.0
    D.x[2:end-1,2:end-1] .= (-etaN ./ dy - etaS ./ dy) ./ dy + (-4 // 3 * etaE ./ dx - 4 // 3 * etaW ./ dx) ./ dx
    G.x[2:end-1,2:end-1] .= @. (4 // 3) * abs(etaE ./ dx .^ 2) + (4 // 3) * abs(etaW ./ dx .^ 2) + abs(etaN ./ dy .^ 2) + abs(etaS ./ dy .^ 2) + abs((3 * dx .^ 2 .* (etaN + etaS) + 4 * dy .^ 2 .* (etaE + etaW)) ./ (dx .^ 2 .* dy .^ 2)) / 3 + abs((2 * etaE - 3 * etaN) ./ (dx .* dy)) / 3 + abs((2 * etaE - 3 * etaS) ./ (dx .* dy)) / 3 + abs((3 * etaN - 2 * etaW) ./ (dx .* dy)) / 3 + abs((3 * etaS - 2 * etaW) ./ (dx .* dy)) / 3
    @show maximum(max.((G.x./D.x)) )
    @show maximum(max.(((2/dx) ./D.x)) )


    etaW, etaE = η.xy[1:end-1,2:end-1],  η.xy[2:end-0,2:end-1] 
    etaS, etaN = η.p[:,1:end-1], η.p[:,2:end-0] 
    etaW[1,:]   .= 0.0
    etaE[end,:] .= 0.0
    D.y[2:end-1,2:end-1] .= (-4 // 3 * etaN ./ dy - 4 // 3 * etaS ./ dy) ./ dy + (-etaE ./ dx - etaW ./ dx) ./ dx
    D.p .= max(nc...) ./ η.p

    # Initial residual
    Residual!(R, Rp, V, P, ε̇, τ, η, Δ)

    # Arrays for solver 
    AV̇   = (x=zeros(size_x...), y=zeros(size_y...)); AṖ = zeros(size_c...)
    V̇ = (x=zeros(size_x...), y=zeros(size_y...)); Ṗ  = zeros(size_c...)
    
    # Initial residual and preconditioned residual
    V̇.x  .= (1 ./D.x).*R.x;  V̇.y .= (1 ./D.y).*R.y;  Ṗ   .= (1 ./D.p).*Rp
    
    # Initialize residual and preconditioned residual
    norm_r0 = sqrt(sum(R.x.*R.x) + sum(R.y.*R.y) + sum(Rp.*Rp)) 
    @show norm_r0
    
    max_iter = 200000#3*nc.x*nc.y 
    tol      = 1e-6
    err      = zeros(max_iter)
    iter     = 0
    α, β     = 0., 0.
    nout     = 100

    # Update damping
    λminV = 0.
    λmaxV = 6.3
    h     = 2.0/sqrt(λmaxV)*0.99
    c     = sqrt(λminV)*0.9
    α     = 2 * h / (2 + c*h) * h
    β     = (2 - c*h) / (2 + c*h)

    # Iteration loop
    for k in 1:max_iter

        iter+=1

        R0.x .= R.x
        R0.y .= R.y
        Rp0  .= Rp

        # Compute A * V̇
        Residual!(AV̇, AṖ, V̇, Ṗ, ε̇, τ, η, Δ)
        # AṖ .-= mean(AṖ)

        # # Compute pseudo time step α
        # if k==1 || mod(k, nout)==0
            # r_dot_z = (dot(R.x, (1 ./D.x).*R.x) + dot(R.y, (1 ./D.y).*R.y) + dot(Rp, (1 ./D.p).*Rp))
        #     α = r_dot_z / (dot(V̇.x, -AV̇.x) + dot(V̇.y, -AV̇.y) + dot(Ṗ, -AṖ) )
        # end

        # Update coupled solution
        V.x .+= h .* V̇.x
        V.y .+= h .* V̇.y
        P   .+= h .* Ṗ
        
        # Compute residual
        Residual!(R, Rp, V, P, ε̇, τ, η, Δ)

        norm_r_new = sqrt(sum(R.x.*R.x) + sum(R.y.*R.y) + sum(Rp.*Rp)) 
        err[iter]  = norm_r_new / norm_r0
        # Check for convergence
        if norm_r_new / norm_r0 < tol  #|| norm_r_new/sqrt(n) < 2*tol 
            println("Converged in $(k/max(nc...)) it/nx.")
            break
        end

        # Compute damping factor β for direction update
        if  mod(k, nout)==0

            λmaxV = (dot(V.x, (1 ./D.x).*V.x) + dot(V.y, (1 ./D.y).*V.y) + dot(P, (1 ./D.p).*P)) / (dot(V.x, V.x) + dot(V.y, V.y) + dot(P, P)) 
            # @show λmaxV
            # λ1 = (dot(V.x, (-1 ./D.x).*R.x) + dot(V.y, (1 ./D.y).*R.y) + dot(P, (1 ./D.p).*Rp)) / (dot(V.x, V.x) + dot(V.y, V.y) + dot(P, P)) 
            # h    = 2.0/sqrt(λmaxV)*0.99

            # β = (dot(R.x, (1 ./D.x).*R.x) + dot(R.y, (1 ./D.y).*R.y) + dot(Rp, (1 ./D.p).*Rp)) / r_dot_z

            # Update damping
            λ2  = abs( (dot(V̇.x, -α.*AV̇.x./D.x) + dot(V̇.y, -α.*AV̇.y./D.y) + dot(Ṗ, -α.*AṖ./D.p)) / ( dot(α.*V̇.x, α.*V̇.x) + dot(α.*V̇.y, α.*V̇.y) + dot(α.*Ṗ, α.*Ṗ)) )
            # λ3  = abs( (dot(V̇.x, h.*(R.x .- R0.x)./D.x) + dot(V̇.y, h.*(R.y .- R0.y)./D.y) + dot(Ṗ, h.*(Rp .- Rp0)./D.p)) / ( dot(h.*V̇.x, h.*V̇.x) + dot(h.*V̇.y, h.*V̇.y) + dot(h.*Ṗ, h.*Ṗ)) )
            # @show λ1, λ2, λ3

            # c     = sqrt(λmaxV*λminV/(λmaxV+λminV))*2.
            c     = sqrt(λ2)*2.1

            α   = 2 * h / (2 + c*h) * h
            β   = (2 - c*h) / (2 + c*h)
            # @show α, β 

            if mod(k, 1000)==0  
                @show norm_r_new
                @show λminV
                isnan(norm_r_new) ? error("Nans") : nothing
            end
        end

        # Update direction p and residual r
        V̇.x .= α .* (1 ./D.x).*R.x .+ β .* V̇.x 
        V̇.y .= α .* (1 ./D.y).*R.y .+ β .* V̇.y 
        Ṗ   .= α .* (1 ./D.p).*Rp  .+ β .* Ṗ 
    end
    # Visualise
    p1 = heatmap(xv, yc, V.x', aspect_ratio=1, xlim=extrema(xv), title="Vx")
    p2 = heatmap(xc, yv, V.y', aspect_ratio=1, xlim=extrema(xv), title="Vy")
    p3 = heatmap(xc[2:end-1], yc[2:end-1], P', aspect_ratio=1, xlim=extrema(xv), title="P", clim=(-10, 10))
    p4 = plot(1:iter, log10.(err[1:iter]), label="", title="Error")
    display(plot(p1, p2, p3, p4))
    Residual!(R, Rp, V, P, ε̇, τ, η, Δ)
    @show norm_r_new = sqrt(sum(R.x.*R.x) + sum(R.y.*R.y) + sum(Rp.*Rp)) 
    @show norm_r_new / norm_r0 < tol 
    return iter/max(nc...)
end

main( (x=1*30, y=1*32) )
# main( (x=2*30, y=2*32) )
# main( (x=4*30, y=4*32) )
# main( (x=8*30, y=8*32) )


