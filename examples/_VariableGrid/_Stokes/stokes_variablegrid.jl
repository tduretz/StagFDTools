
using Distributions

function normal_linspace_interval(inflimit::Float64, suplimit::Float64, μ::Float64, σ::Float64, ncells::Int)
    dist = Normal(μ, σ)
    inf_cdf = cdf(dist, inflimit)
    sup_cdf = cdf(dist, suplimit)
    vec = range(inf_cdf, sup_cdf; length=ncells)
    return quantile.(dist, vec)
end


function TangentOperator_var!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η , V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)

    _ones = @SVector ones(4)

    # Loop over centroids
    for j=1:size(ε̇.xx,2)-0, i=1:size(ε̇.xx,1)-0
        
        if (i==1 && j==1) || (i==size(ε̇.xx,1) && j==1) || (i==1 && j==size(ε̇.xx,2)) || (i==size(ε̇.xx,1) && j==size(ε̇.xx,2))
            # Avoid the outer corners - nothing is well defined there ;)
        else
            Vx     = SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1,   jj in j:j+2)
            Vy     = SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2,   jj in j:j+1)
            bcx    = SMatrix{2,3}(    BC.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
            bcy    = SMatrix{3,2}(    BC.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
            typex  = SMatrix{2,3}(  type.Vx[ii,jj] for ii in i:i+1,   jj in j:j+2)
            typey  = SMatrix{3,2}(  type.Vy[ii,jj] for ii in i:i+2,   jj in j:j+1)
            τxy0   = SMatrix{2,2}(    τ0.xy[ii,jj] for ii in i:i+1,   jj in j:j+1)
            # no need of P because we are on centroids, we can directly take Pt value

            if ( j==size(ε̇.xx,2) ) || ( i==size(ε̇.xx,1) )
                Δx_loc     = zeros(2)
                Δx_loc[1] = Δ.x[i]
                Δx_loc[2] = Δx_loc[1]
                Δy_loc     = zeros(2)
                Δy_loc[1] = Δ.y[j]
                Δy_loc[2] = Δy_loc[1]
                Δy_Vx_loc  = zeros(3)
                Δy_Vx_loc[1] = Δ.y[j]
                Δy_Vx_loc[2] = Δy_Vx_loc[1]
                Δy_Vx_loc[3] = Δy_Vx_loc[2]
                Δx_Vy_loc  = zeros(3)
                Δx_Vy_loc[1] = Δ.x[j]
                Δx_Vy_loc[2] = Δx_Vy_loc[1]
                Δx_Vy_loc[3] = Δx_Vy_loc[2]
            elseif ( j==size(ε̇.xx,2)-1 ) || ( i==size(ε̇.xx,1)-1 )
                Δx_loc     = SVector{2}(Δ.x[ii] for ii in i:i+1)
                Δy_loc     = SVector{2}(Δ.y[jj] for jj in j:j+1)
                Δy_Vx_loc  = zeros(3)
                Δy_Vx_loc[1] = Δ.y[j]
                Δy_Vx_loc[2] = Δ.y[j+1]
                Δy_Vx_loc[3] = Δy_Vx_loc[2]
                Δx_Vy_loc  = zeros(3)
                Δx_Vy_loc[1] = Δ.x[j]
                Δx_Vy_loc[2] = Δ.x[j+1]
                Δx_Vy_loc[3] = Δx_Vy_loc[2]
            else
                Δx_loc     = SVector{2}(Δ.x[ii] for ii in i:i+1)
                Δy_loc     = SVector{2}(Δ.y[jj] for jj in j:j+1)
                Δy_Vx_loc     = SVector{3}(Δ.y[jj] for jj in j:j+2)
                Δx_Vy_loc     = SVector{3}(Δ.x[ii] for ii in i:i+2)
            end

            Vx     = SetBCVx1_var(Vx, typex, bcx, Δx_loc, Δy_loc)
            Vy     = SetBCVy1_var(Vy, typey, bcy, Δx_loc, Δy_loc)

            Δx = zeros(2)
            Δx .= Δx_loc
            Δy = zeros(2)
            Δy .= Δy_loc
            size_stencil_Vx_X = 2
            size_stencil_Vx_Y = 3
            size_stencil_Vy_X = 3
            size_stencil_Vy_Y = 2
            
            Dxx = zeros(size_stencil_Vx_X-1,size_stencil_Vx_Y-2)
            for i in 1:size_stencil_Vx_X-1, j in 1:size_stencil_Vx_Y-2
                Dxx[i,j] = (Vx[i+1, j+1] - Vx[i, j+1]) / Δx_loc[i]
            end
            Dyy = zeros(size_stencil_Vy_X-2,size_stencil_Vy_Y-1)
            for i in 1:size_stencil_Vy_X-2, j in 1:size_stencil_Vy_Y-1
                Dyy[i,j] = (Vy[i+1, j+1] - Vy[i+1, j]) / Δy_loc[j]
            end
            Dxy = zeros(size_stencil_Vx_X,size_stencil_Vx_Y-1)
            for i in 1:size_stencil_Vx_X, j in 1:size_stencil_Vx_Y-1
                Dxy[i,j] = (Vx[i, j+1] - Vx[i, j]) / ((Δy_Vx_loc[j]+Δy_Vx_loc[j+1])/2)
            end
            Dyx = zeros(size_stencil_Vy_X-1,size_stencil_Vy_Y)
            for i in 1:size_stencil_Vy_X-1, j in 1:size_stencil_Vy_Y
                Dyx[i,j] = (Vy[i+1, j] - Vy[i, j]) / ((Δx_Vy_loc[i]+Δx_Vy_loc[i+1])/2)
            end

            Dkk = Dxx .+ Dyy
            ε̇xx = @. Dxx - Dkk ./ 3
            ε̇yy = @. Dyy - Dkk ./ 3
            ε̇xy = @. (Dxy + Dyx) ./ 2

            ε̇̄xy = zeros(2)
            τ̄xy0 = zeros(2)
            for i in 1:2
                ε̇̄xy[i]  = mean(ε̇xy[i,:])
                τ̄xy0[i] = mean(τxy0[i,:])
            end

            # Visco-elasticity
            G     = materials.G[phases.c[i,j]]
            ε̇vec  = @SVector([ε̇xx[1]+τ0.xx[i,j]/(2*G[1]*Δ.t[1]), ε̇yy[1]+τ0.yy[i,j]/(2*G[1]*Δ.t[1]), ε̇̄xy[1]+τ̄xy0[1]/(2*G[1]*Δ.t[1]), Pt[i,j]])
            # Tangent operator used for Newton Linearisation
            Δt =  Δ.t[1]
            jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector_var!, ε̇vec, Const(Dkk[1]), Const(Pt0[i,j]), Const(materials), Const(phases.c[i,j]), Const(Δt))

            # Why the hell is enzyme breaking the Jacobian into vectors??? :D
            # jac.derivs = ∂ (output StressVector) / ∂ε̇vec -> (Tuple(4) * (Svector{4}, Float, Float), nothing, nothing, nothing)
            # jac.derivs[1] = tuple from enzyme, Tuple(4) * (Svector{4}, Float, Float), containing the 4 derivatives with respect to each variable
            # jac.derivs[1][l] = l_th item of tuple, (Svector{4}, float, float), derivative with respect to l_th variable
            # jac.derivs[1][1][l] = Svector{4}, derivative of τ with respect to ε̇[l]
            # D_ctl is the true derivative 
            @views 𝐷_ctl.c[i,j][:,1] .= jac.derivs[1][1][1] # ∂τ / ∂ε̇[1] = ∂ (2ηε̇vec_updated) / ∂ε̇[1] -> Svector{4}
            @views 𝐷_ctl.c[i,j][:,2] .= jac.derivs[1][2][1]
            @views 𝐷_ctl.c[i,j][:,3] .= jac.derivs[1][3][1]
            @views 𝐷_ctl.c[i,j][:,4] .= jac.derivs[1][4][1]

            # Tangent operator used for Picard Linearisation (we remove non linear term)
            𝐷.c[i,j] .= diagm(2*jac.val[2] * _ones) # 2 * eta updated * ones
            𝐷.c[i,j][4,4] = 1 # force pressure at 1

            # Update stress
            τ.xx[i,j]  = jac.val[1][1] # 2 eta Epsi_xx
            τ.yy[i,j]  = jac.val[1][2] # 2 eta Epsi_yy
            ε̇.xx[i,j]  = ε̇xx[1]
            ε̇.yy[i,j]  = ε̇yy[1]
            λ̇.c[i,j]   = jac.val[3] # Lambda updated
            η.c[i,j]   = jac.val[2] # eta updated
            ΔPt.c[i,j] = (jac.val[1][4] - Pt[i,j]) # P updated after stressvector - Pi,j
        end
    end

    # Loop over vertices
    for j=1:size(ε̇.xy,2)-2, i=1:size(ε̇.xy,1)-2
        #scatter!(p, [i], [j], markersize=3, color=:red)
        Vx     = SMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        Vy     = SMatrix{2,3}(      V.y[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        bcx    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        bcy    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        typex  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2,   jj in j+1:j+2)
        typey  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i+1:i+2, jj in j:j+2  )
        τxx0   = SMatrix{2,2}(    τ0.xx[ii,jj] for ii in i:i+1,   jj in j:j+1)
        τyy0   = SMatrix{2,2}(    τ0.yy[ii,jj] for ii in i:i+1,   jj in j:j+1)
        P      = SMatrix{2,2}(       Pt[ii,jj] for ii in i:i+1,   jj in j:j+1)

        if ( i == size(ε̇.xy,1)-1) || ( j == size(ε̇.xy,2)-1)
            Δx_loc = zeros(2)
            Δy_loc = zeros(2)
            Δx_loc[1] = Δ.x[end] # end because size delta.x = size(ε̇.xy,1)-1, so we take the last item
            Δx_loc[2] = Δx_loc[1]
            Δy_loc[1] = Δ.y[end]
            Δy_loc[2] = Δy_loc[1]
            Δx_Vx_loc = zeros(3)
            Δy_Vy_loc = zeros(3)
            Δx_Vx_loc[1] = Δ.x[end]
            Δx_Vx_loc[2] = Δx_Vx_loc[1]
            Δx_Vx_loc[3] = Δx_Vx_loc[2]
            Δy_Vy_loc[1] = Δ.y[end]
            Δy_Vy_loc[2] = Δy_Vy_loc[1]
            Δy_Vy_loc[3] = Δy_Vy_loc[2]
        elseif ( i == size(ε̇.xy,1)-2) || ( j == size(ε̇.xy,2)-2)
            Δx_loc = zeros(2)
            Δy_loc = zeros(2)
            Δx_loc[1] = Δ.x[i+1]
            Δx_loc[2] = Δx_loc[1]
            Δy_loc[1] = Δ.y[j+1]
            Δy_loc[2] = Δy_loc[1]
            Δx_Vx_loc = zeros(3)
            Δy_Vy_loc = zeros(3)
            Δx_Vx_loc[1] = Δ.x[i]
            Δx_Vx_loc[2] = Δ.x[i+1]
            Δx_Vx_loc[3] = Δx_Vx_loc[2]
            Δy_Vy_loc[1] = Δ.y[j]
            Δy_Vy_loc[2] = Δ.y[j+1]
            Δy_Vy_loc[3] = Δy_Vy_loc[2]
        else
            Δx_loc     = SVector{2}(Δ.x[ii] for ii in i+1:i+2)
            Δy_loc     = SVector{2}(Δ.y[jj] for jj in j+1:j+2)
            Δx_Vx_loc     = SVector{3}(Δ.x[ii] for ii in i:i+2)
            Δy_Vy_loc     = SVector{3}(Δ.y[jj] for jj in j:j+2)
        end
           
        Vx     = SetBCVx1_var(Vx, typex, bcx, Δx_loc, Δy_loc)
        Vy     = SetBCVy1_var(Vy, typey, bcy, Δx_loc, Δy_loc)

        Δx = zeros(2)
        Δx .= Δx_loc
        Δy = zeros(2)
        Δy .= Δy_loc
        size_stencil_Vx_X = 3
        size_stencil_Vx_Y = 2
        size_stencil_Vy_X = 2
        size_stencil_Vy_Y = 3
        Dxx = zeros(size_stencil_Vx_X-1,size_stencil_Vx_Y)
        for i in 1:size_stencil_Vx_X-1, j in 1:size_stencil_Vx_Y
            Dxx[i,j] = (Vx[i+1, j] - Vx[i, j]) / ((Δx_Vx_loc[i]+Δx_Vx_loc[i+1])/2)
        end
        Dyy = zeros(size_stencil_Vy_X,size_stencil_Vy_Y-1)
        for i in 1:size_stencil_Vy_X, j in 1:size_stencil_Vy_Y-1
            Dyy[i,j] = (Vy[i, j+1] - Vy[i, j]) / ((Δy_Vy_loc[j]+Δy_Vy_loc[j+1])/2)
        end
        Dxy = zeros(size_stencil_Vx_X-2,size_stencil_Vx_Y-1)
        for i in 1:size_stencil_Vx_X-2, j in 1:size_stencil_Vx_Y-1
            Dxy[i,j] = (Vx[i+1, j+1] - Vx[i+1, j]) / Δy_loc[j]
        end
        Dyx = zeros(size_stencil_Vy_X-1,size_stencil_Vy_Y-2)
        for i in 1:size_stencil_Vy_X-1, j in 1:size_stencil_Vy_Y-2
            Dyx[i,j] = (Vy[i+1, j+1] - Vy[i, j+1]) / Δx_loc[i]
        end

        Dkk   = @. Dxx + Dyy
        ε̇xx   = @. Dxx - Dkk / 3
        ε̇yy   = @. Dyy - Dkk / 3
        ε̇xy   = @. (Dxy + Dyx) /2

        ε̇̄xx = zeros(2)
        ε̇̄yy = zeros(2)
        P̄ = zeros(2) # we are looping over vertices, but Pt is defined over centroids, thus we take the mean
        τ̄xx0 = zeros(2)
        τ̄yy0 = zeros(2)
        for i in 1:2
            ε̇̄xx[i]  = mean(ε̇xx[i,:])
            ε̇̄yy[i]  = mean(ε̇yy[i,:])
            P̄[i]    = mean(P[i,:])
            τ̄xx0[i] = mean(τxx0[i,:])
            τ̄yy0[i] = mean(τyy0[i,:])
        end
        
        # Visco-elasticity
        G     = materials.G[phases.v[i+1,j+1]]

        ε̇vec  = @SVector([ε̇̄xx[1]+τ̄xx0[1]/(2*G[1]*Δ.t[1]), ε̇̄yy[1]+τ̄yy0[1]/(2*G[1]*Δ.t[1]), ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t[1]), P̄[1]])
        
        # Tangent operator used for Newton Linearisation
        Δt = Δ.t[1]
        jac   = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVector_var!, ε̇vec, Const(Dkk[1]), Const(Pt0[i,j]),Const(materials), Const(phases.v[i+1,j+1]), Const(Δt))

        # Why the hell is enzyme breaking the Jacobian into vectors??? :D 
        @views 𝐷_ctl.v[i+1,j+1][:,1] .= jac.derivs[1][1][1]
        @views 𝐷_ctl.v[i+1,j+1][:,2] .= jac.derivs[1][2][1]
        @views 𝐷_ctl.v[i+1,j+1][:,3] .= jac.derivs[1][3][1]
        @views 𝐷_ctl.v[i+1,j+1][:,4] .= jac.derivs[1][4][1]

        # Tangent operator used for Picard Linearisation
        𝐷.v[i+1,j+1] .= diagm(2*jac.val[2] * _ones)
        𝐷.v[i+1,j+1][4,4] = 1

        # Update stress
        τ.xy[i+1,j+1] = jac.val[1][3]
        ε̇.xy[i+1,j+1] = ε̇xy[1]
        λ̇.v[i+1,j+1]  = jac.val[3]
        η.v[i+1,j+1]  = jac.val[2]
        # τ.xy[i+1,j+1] = 2*jac.val[2]*(ε̇xy[1]+τ0.xy[i+1,j+1]/(2*G[1]*Δ.t))
    end

end



function SetBCVx1_var(Vx, typex, bcx, Δx, Δy)

    MVx = MMatrix(Vx)
    # Vx follows vertices axes for North and South orientations
    # N/S
    for ii in axes(typex, 1)
        if typex[ii,1] == :Dirichlet_tangent
            MVx[ii,1] = fma(2, bcx[ii,1], -Vx[ii,2])
        elseif typex[ii,1] == :Neumann_tangent
            coeff = ((Δy[1]+Δy[2])/2)
            MVx[ii,1] = fma(coeff, bcx[ii,1], Vx[ii,2])
        end

        if typex[ii,end] == :Dirichlet_tangent
            MVx[ii,end] = fma(2, bcx[ii,end], -Vx[ii,end-1])
        elseif typex[ii,end] == :Neumann_tangent
            coeff = ((Δy[end]+Δy[end-1])/2)
            MVx[ii,end] = fma(coeff, bcx[ii,end], Vx[ii,end-1])
        end
    end
    
    # E/W
    # Vx follows centers axes for East and West orientations
    for jj in axes(typex, 2)
        if typex[1,jj] == :Neumann_normal
            MVx[1,jj] = fma(2, Δx[1]*bcx[1,jj], Vx[2,jj])
        end
        if typex[end,jj] == :Neumann_normal
            MVx[end,jj] = fma(2,-Δx[end]*bcx[end,jj], Vx[end-1,jj])
        end
    end

    return SMatrix(MVx)
end


function SetBCVy1_var(Vy, typey, bcy, Δx, Δy)
    MVy = MMatrix(Vy)
    # E/W
    # Vy follows vertices axes for East and West orientations
    for jj in axes(typey, 2)
        if typey[1,jj] == :Dirichlet_tangent
            MVy[1,jj] = fma(2, bcy[1,jj], -Vy[2,jj])
        elseif typey[1,jj] == :Neumann_tangent
            coeff = ((Δx[1]+Δx[2])/2)
            MVy[1,jj] = fma(coeff, bcy[1,jj], Vy[2,jj])
        end

        if typey[end,jj] == :Dirichlet_tangent
            MVy[end,jj] = fma(2, bcy[end,jj], -Vy[end-1,jj])
        elseif typey[end,jj] == :Neumann_tangent
            coeff = ((Δx[end]+Δx[end-1])/2)
            MVy[end,jj] = fma(coeff, bcy[end,jj], Vy[end-1,jj])
        end
    end
    # N/S
    # Vy follows centers axes for North and South orientations
    for ii in axes(typey, 1)
        if typey[ii,1] == :Neumann_normal
            MVy[ii,1] = fma(2, Δy[1]*bcy[ii,1], Vy[ii,2])
        end
        if typey[ii,end] == :Neumann_normal
            MVy[ii,end] = fma(2,-Δy[end]*bcy[ii,end], Vy[ii,end-1])
        end
    end
    return SMatrix(MVy) 
end


function ResidualContinuity2D_var!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
                
    # loop on centroids
    for j in 2:size(R.p,2)-1, i in 2:size(R.p,1)-1
        if type.Pt[i,j] !== :constant 
            Vx_loc     = SMatrix{2,2}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+1)
            Vy_loc     = SMatrix{2,2}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+1)

            Δx_loc     = SVector{1}(Δ.x[ii] for ii in i:i)
            Δy_loc     = SVector{1}(Δ.y[jj] for jj in j:j)

            Δt_loc = Δ.t[1]

            bcv_loc    = (;)
            type_loc   = (;)
            D          = (;)
            R.p[i,j]   = Continuity_var(Vx_loc, Vy_loc, P[i,j], P0[i,j], D, phases.c[i,j], materials, type_loc, bcv_loc, Δx_loc, Δy_loc, Δt_loc)
        end
    end
    return nothing
end


function Continuity_var(Vx, Vy, Pt, Pt0, D, phase, materials, type_loc, bcv_loc, Δx, Δy, Δt)
    invΔx = 1 / Δx[1]
    invΔy = 1 / Δy[1]

    invΔt = 1 / Δt
    β     = materials.β[phase]
    η     = materials.β[phase]
    comp  = materials.compressible

    f     = ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy) + comp * β * (Pt[1] - Pt0) * invΔt #+ 1/(1000*η)*Pt[1]
    return f
end



function ResidualMomentum2D_x_var!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        if type.Vx[i,j] == :in
            Vx_loc     = SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc     = SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
            P_loc      = SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            ΔP_loc     = SMatrix{2,1}(     ΔP.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-1:i+1, jj in j-1:j  )

            Dc         = SMatrix{2,1}(      𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            Dv         = SMatrix{1,2}(      𝐷.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

            if (i == nc.x+shift.x+1)
                Δx_loc     = zeros(2)
                Δx_loc[1] = Δ.x[i-1]
                Δx_loc[2] = Δ.x[i]
            else
                Δx_loc     = SVector{2}(Δ.x[ii] for ii in i-1:i)
            end
            if (j == nc.y+shift.y)
                Δy_loc     = zeros(4)
                Δy_loc[1] = Δ.y[j-2]
                Δy_loc[2] = Δ.y[j-1]
                Δy_loc[3] = Δ.y[j]
                Δy_loc[4] = Δy_loc[3]
            else
                Δy_loc     = SVector{4}(Δ.y[jj] for jj in j-2:j+1)
            end

            R.x[i,j]   = SMomentum_x_Generic_var(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δx_loc, Δy_loc, Δ.t[1], i, j)
        end
    end
    return nothing
end


function SMomentum_x_Generic_var(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, phases, materials, type, bcv, Δx, Δy, Δt, i, j)

    # BC
    Vx = SetBCVx1_var(Vx_loc, type.x, bcv.x, Δx, Δy)
    Vy = SetBCVy1_var(Vy_loc, type.y, bcv.y, Δx, Δy)

    # Velocity gradient
    size_stencil_X = 3
    size_stencil_Y = 4
    Dxx = zeros(size_stencil_X-1,size_stencil_X)
    for i in 1:size_stencil_X-1, j in 1:size_stencil_X
        Dxx[i,j] = (Vx[i+1, j] - Vx[i, j]) / Δx[i]
    end
    Dyy = zeros(size_stencil_Y-2,size_stencil_Y-1)
    for i in 1:size_stencil_Y-2, j in 1:size_stencil_Y-1
        Dyy[i,j] = (Vy[i+1, j+1] - Vy[i+1, j]) / Δy[j]
    end
    Dxy = zeros(size_stencil_X,size_stencil_X-1) # we get values at yellow pointa
    for i in 1:size_stencil_X, j in 1:size_stencil_X-1
        Dxy[i,j] = (Vx[i, j+1] - Vx[i, j]) / ((Δy[j]+Δy[j+1])/2) # Centre + south et centre + north
    end
    Dyx = zeros(size_stencil_Y-1,size_stencil_Y-2) # we get values at yellow points
    for i in 1:size_stencil_Y-1, j in 1:size_stencil_Y-2
        Dyx[i,j] = (Vy[i+1, j+1] - Vy[i, j+1]) / ((Δx[1]+Δx[2])/2) # fixed
    end

    # Strain rate
    ε̇kk = @. Dxx + Dyy
    # normal strain rate xx
    ε̇xx = @. Dxx - 1/3*ε̇kk
    # normal strain rate yy
    ε̇yy = @. Dyy - 1/3*ε̇kk
    # shear strain rate 
    ε̇xy = @. 1/2 * ( Dxy + Dyx ) # on the vertical face, to get the shear strain rate, I take the mean

    ε̇̄xy = zeros(2)
    ε̇̄xx = zeros(2)
    ε̇̄yy = zeros(2)
    P̄t = zeros(2)
    τ̄0xx = zeros(2)
    τ̄0yy = zeros(2)
    τ̄0xy = zeros(2)
    for i in 1:2
        # Average vertex to centroid
        ε̇̄xy[i]  = mean(ε̇xy[i,:])
        # Average centroid to vertex
        ε̇̄xx[i]  = mean(ε̇xx[i,:])
        ε̇̄yy[i]  = mean(ε̇yy[i,:])
        P̄t[i]   = mean(Pt[i,:])
        τ̄0xx[i] = mean(τ0.xx[i,:])
        τ̄0yy[i] = mean(τ0.yy[i,:])
        τ̄0xy[i] = mean(τ0.xy[i,:])
    end

    # Effective strain rate
    Gc   = SVector{2, Float64}( materials.G[phases.c] )
    Gv   = SVector{2, Float64}( materials.G[phases.v] )
    tmpc = @. inv(2 * Gc * Δt)
    tmpv = @. inv(2 * Gv * Δt)
    ϵ̇xx  = @. ε̇xx[:,2] + τ0.xx[:,2] * tmpc # over centroids
    ϵ̇yy  = @. ε̇yy[:,2] + τ0.yy[:,2] * tmpc
    ϵ̇̄xy  = @. ε̇̄xy[:]   + τ̄0xy[:]    * tmpc
    ϵ̇̄xx  = @. ε̇̄xx[:]   + τ̄0xx[:]    * tmpv # over vertices
    ϵ̇̄yy  = @. ε̇̄yy[:]   + τ̄0yy[:]    * tmpv
    ϵ̇xy  = @. ε̇xy[2,:] + τ0.xy[2,:] * tmpv

    # Corrected pressure
    comp = materials.compressible
    Ptc  = SVector{2, Float64}( @. Pt[:,2] + comp * ΔP[:] )

    # Stress
    # normal stress gradient, wrt vertices
    τxx = @MVector zeros(2) # left and right cells
    # shear stress gradient, wrt centroids
    τxy = @MVector zeros(2) # left and right cells with respect to the current vertical face on which Vx is defined
    for i=1:2
        # Txx = (partial T / partial ϵ̇)_xx - 0 (P component)  𝐷.c[i][1,:] = [(partial T / partial ϵ̇)_xx   0    0    0]
        # Last line of tangent operator 𝐷.c[i][1,:] = [ 0   0   0   P]
        τxx[i] = (𝐷.c[i][1,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][1,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][1,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][1,4] - (𝐷.c[i][4,4] - 1)) * Pt[i,2]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                       * P̄t[i]
    end

    # Residual
    fx  = ( τxx[2]  - τxx[1] ) * (1/Δx[2]) # east - west
    fx += ( τxy[2]  - τxy[1] ) * (2/(Δy[2]+Δy[1])) # north - south
    fx -= ( Ptc[2]  - Ptc[1] ) * (1/Δx[2]) # east - west
    fx *= -1 * Δx[2] * ((Δy[1]+Δy[2])/2)

    return fx
end


function ResidualMomentum2D_y_var!(R, V, P, P0, ΔP, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        if type.Vy[i,j] == :in
            Vx_loc     = SMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            Vy_loc     = SMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-1:i-0, jj in j-0:j-0) 
            P_loc      = SMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            ΔP_loc     = SMatrix{1,2}(     ΔP.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            τxx0       = SMatrix{3,2}(    τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τyy0       = SMatrix{3,2}(    τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τxy0       = SMatrix{2,3}(    τ0.xy[ii,jj] for ii in i-1:i,   jj in j-1:j+1)
            Dc         = SMatrix{1,2}(      𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
            Dv         = SMatrix{2,1}(      𝐷.v[ii,jj] for ii in i-1:i-0,   jj in j-0:j-0)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

            if (j == nc.y+shift.y+1)
                Δy_loc     = zeros(2)
                Δy_loc[1] = Δ.y[j-1]
                Δy_loc[2] = Δ.y[j]
            else
                Δy_loc     = SVector{2}(Δ.y[jj] for jj in j-1:j)
            end

            if (i == nc.x+shift.x)
                Δx_loc     = zeros(4)
                Δx_loc[1] = Δ.x[i-2]
                Δx_loc[2] = Δ.x[i-1]
                Δx_loc[3] = Δ.x[i]
                Δx_loc[4] = Δx_loc[3]
            else
                Δx_loc     = SVector{4}(Δ.x[ii] for ii in i-2:i+1)
            end


            R.y[i,j]   = SMomentum_y_Generic_var(Vx_loc, Vy_loc, P_loc, ΔP_loc, τ0_loc, D, ph_loc, materials, type_loc, bcv_loc, Δx_loc, Δy_loc, Δ.t[1])
        end
    end
    return nothing
end


function SMomentum_y_Generic_var(Vx_loc, Vy_loc, Pt, ΔP, τ0, 𝐷, phases, materials, type, bcv, Δx, Δy, Δt)
    
    # BC
    Vx = SetBCVx1_var(Vx_loc, type.x, bcv.x, Δx, Δy)
    Vy = SetBCVy1_var(Vy_loc, type.y, bcv.y, Δx, Δy)

    # Velocity gradient
    size_stencil_X = 4
    size_stencil_Y = 3
    
    Dxx = zeros(size_stencil_X-1,size_stencil_X-2)
    for i in 1:size_stencil_X-1, j in 1:size_stencil_X-2
        Dxx[i,j] = (Vx[i+1, j+1] - Vx[i, j+1]) / Δx[i]
    end
    Dyy = zeros(size_stencil_Y,size_stencil_Y-1)
    for i in 1:size_stencil_Y, j in 1:size_stencil_Y-1
        Dyy[i,j] = (Vy[i, j+1] - Vy[i, j]) / Δy[j]
    end
    Dxy = zeros(size_stencil_X-2,size_stencil_X-1)
    for i in 1:size_stencil_X-2, j in 1:size_stencil_X-1
        Dxy[i,j] = (Vx[i+1, j+1] - Vx[i+1, j]) / ((Δy[1]+Δy[2])/2)
    end
    Dyx = zeros(size_stencil_Y-1,size_stencil_Y)
    for i in 1:size_stencil_Y-1, j in 1:size_stencil_Y
        Dyx[i,j] = (Vy[i+1, j] - Vy[i, j]) / ((Δx[i]+Δx[i+1])/2)
    end

    # Strain rate
    ε̇kk = @. Dxx + Dyy
    ε̇xx = @. Dxx - 1/3*ε̇kk
    ε̇yy = @. Dyy - 1/3*ε̇kk
    ε̇xy = @. 1/2 * (Dxy + Dyx)

    ε̇̄xy = zeros(2)
    ε̇̄xx = zeros(2)
    ε̇̄yy = zeros(2)
    P̄t = zeros(2)
    τ̄0xx = zeros(2)
    τ̄0yy = zeros(2)
    τ̄0xy = zeros(2)
    for i in 1:2
        # Average vertex to centroid
        ε̇̄xy[i]  = mean(ε̇xy[i,:])
        # Average centroid to vertex
        ε̇̄xx[i]  = mean(ε̇xx[i,:])
        ε̇̄yy[i]  = mean(ε̇yy[i,:])
        P̄t[i]   = mean(Pt[i,:])
        τ̄0xx[i] = mean(τ0.xx[i,:])
        τ̄0yy[i] = mean(τ0.yy[i,:])
        τ̄0xy[i] = mean(τ0.xy[i,:])
    end
    
    # Effective strain rate
    Gc   = SVector{2, Float64}( materials.G[phases.c])
    Gv   = SVector{2, Float64}( materials.G[phases.v])
    tmpc = (2*Gc.*Δt)
    tmpv = (2*Gv.*Δt)
    ϵ̇xx  = @. ε̇xx[2,:] + τ0.xx[2,:] / tmpc
    ϵ̇yy  = @. ε̇yy[2,:] + τ0.yy[2,:] / tmpc
    ϵ̇̄xy  = @. ε̇̄xy[:]   + τ̄0xy[:]    / tmpc
    ϵ̇̄xx  = @. ε̇̄xx[:]   + τ̄0xx[:]    / tmpv
    ϵ̇̄yy  = @. ε̇̄yy[:]   + τ̄0yy[:]    / tmpv
    ϵ̇xy  = @. ε̇xy[:,2] + τ0.xy[:,2] / tmpv

    # Corrected pressure
    comp = materials.compressible
    Ptc  = SVector{2, Float64}( @. Pt[2,:] + comp * ΔP[:] )

    # Stress
    τyy = @MVector zeros(2)
    τxy = @MVector zeros(2)
    for i=1:2
        τyy[i] = (𝐷.c[i][2,1] - 𝐷.c[i][4,1]) * ϵ̇xx[i] + (𝐷.c[i][2,2] - 𝐷.c[i][4,2]) * ϵ̇yy[i] + (𝐷.c[i][2,3] - 𝐷.c[i][4,3]) * ϵ̇̄xy[i] + (𝐷.c[i][2,4] - (𝐷.c[i][4,4] - 1.)) * Pt[2,i]
        τxy[i] = 𝐷.v[i][3,1]                 * ϵ̇̄xx[i] + 𝐷.v[i][3,2]                 * ϵ̇̄yy[i] + 𝐷.v[i][3,3]                  * ϵ̇xy[i] + 𝐷.v[i][3,4]                        * P̄t[i]
    end

    # Residual
    fy  = ( τyy[2]  -  τyy[1] ) * (1/Δy[2])
    fy += ( τxy[2]  -  τxy[1] ) * (2/(Δx[2]+Δx[1]))
    fy -= ( Ptc[2]  -  Ptc[1])  * (1/Δy[2])
    fy *= -1 * ((Δx[1]+Δx[2])/2) * Δy[2]
    
    return fy
end



#----------------------------------
#           Assembling
#----------------------------------

function AssembleContinuity2D_var!(K, V, P, Pt0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(2,3)
    ∂R∂Vy = @MMatrix zeros(3,2)
    ∂R∂P  = @MMatrix zeros(1,1)
    
    Vx_loc= @MMatrix zeros(2,3)
    Vy_loc= @MMatrix zeros(3,2)
    P_loc = @MMatrix zeros(1,1)

    for j in 2:size(P, 2)-1, i in 2:size(P, 1)-1
        Vx_loc    .= SMatrix{2,3}(      V.x[ii,jj] for ii in i:i+1, jj in j:j+2)
        Vy_loc    .= SMatrix{3,2}(      V.y[ii,jj] for ii in i:i+2, jj in j:j+1)
        P_loc     .= SMatrix{1,1}(        P[ii,jj] for ii in i:i,   jj in j:j  )
        bcv_loc    = (;)
        type_loc   = (;)
        D          = (;)
        
        fill!(∂R∂Vx, 0e0)
        fill!(∂R∂Vy, 0e0)
        fill!(∂R∂P , 0e0)
        
        Δx_loc     = SVector{1}(Δ.x[ii] for ii in i:i)
        Δy_loc     = SVector{1}(Δ.y[jj] for jj in j:j)

        Δt_loc        = Δ.t[1]

        autodiff(Enzyme.Reverse, Continuity_var, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂P), Const(Pt0[i,j]), Const(D), Const(phases.c[i,j]), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δx_loc), Const(Δy_loc), Const(Δt_loc))

        # Pt --- Vx
        Local = SMatrix{2,3}(num.Vx[ii,jj] for ii in i:i+1, jj in j:j+2)# .* pattern[3][1]        
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] / Δx_loc[1]
            end
        end
        # Pt --- Vy
        Local = SMatrix{3,2}(num.Vy[ii,jj] for ii in i:i+2, jj in j:j+1) #.* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] / Δy_loc[1]
            end
        end

        # Pt --- Pt
        if num.Pt[i,j]>0
            K[3][3][num.Pt[i,j], num.Pt[i,j]] = ∂R∂P[1,1]
        end
    end
    return nothing
end




function AssembleMomentum2D_x_var!(K, V, P, P0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,3)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
                
    Vx_loc = @MMatrix zeros(3,3)
    Vy_loc = @MMatrix zeros(4,4)
    P_loc  = @MMatrix zeros(2,3)
    ΔP_loc = @MMatrix zeros(2,1)

    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        
        if type.Vx[i,j] == :in

            bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0) 
            
            Vx_loc    .= SMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc    .= SMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            P_loc     .= SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            ΔP_loc    .= SMatrix{2,1}(     ΔP.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)

            τxx0       = SMatrix{2,3}(    τ0.xx[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τyy0       = SMatrix{2,3}(    τ0.yy[ii,jj] for ii in i-1:i,   jj in j-2:j  )
            τxy0       = SMatrix{3,2}(    τ0.xy[ii,jj] for ii in i-1:i+1, jj in j-1:j  )
            
            Dc         = SMatrix{2,1}(      𝐷.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            Dv         = SMatrix{1,2}(      𝐷.v[ii,jj] for ii in i-0:i-0, jj in j-1:j-0)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

            if (i == nc.x+shift.x+1)
                Δx_loc     = zeros(2)
                Δx_loc[1] = Δ.x[i-1]
                Δx_loc[2] = Δ.x[i]
            else
                Δx_loc     = SVector{2}(Δ.x[ii] for ii in i-1:i)
            end

            if (j == nc.y+shift.y)
                Δy_loc     = zeros(4)
                Δy_loc[1] = Δ.y[j-2]
                Δy_loc[2] = Δ.y[j-1]
                Δy_loc[3] = Δ.y[j]
                Δy_loc[4] = Δy_loc[3]
            else
                Δy_loc     = SVector{4}(Δ.y[jj] for jj in j-2:j+1)
            end
            Δt_loc        = Δ.t[1]

            fill!(∂R∂Vx, 0e0)
            fill!(∂R∂Vy, 0e0)
            fill!(∂R∂Pt, 0e0)
            
            autodiff(Enzyme.Reverse, SMomentum_x_Generic_var, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ΔP_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δx_loc), Const(Δy_loc), Const(Δt_loc), Const(i), Const(j))
            # Vx --- Vx
            Local = SMatrix{3,3}(num.Vx[ii, jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][1][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj]
                end
            end
            # Vx --- Vy
            Local = SMatrix{4,4}(num.Vy[ii, jj] for ii in i-1:i+2, jj in j-2:j+1) .* pattern[1][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][2][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]
                end
            end
            # Vx --- Pt
            Local = SMatrix{2,3}(num.Pt[ii, jj] for ii in i-1:i, jj in j-2:j) .* pattern[1][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][3][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
        end
    end
    return nothing
end



function AssembleMomentum2D_y_var!(K, V, P, P0, ΔP, τ0, 𝐷, phases, materials, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(3,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    
    Vx_loc = @MMatrix zeros(4,4)
    Vy_loc = @MMatrix zeros(3,3)
    P_loc  = @MMatrix zeros(3,2)
    ΔP_loc = @MMatrix zeros(1,2)
       
    shift    = (x=2, y=1)
    K21 = K[2][1]
    K22 = K[2][2]
    K23 = K[2][3]

    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x

        if type.Vy[i,j] === :in

            Vx_loc    .= @inline SMatrix{4,4}(@inbounds       V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            Vy_loc    .= @inline SMatrix{3,3}(@inbounds       V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcx_loc    = @inline SMatrix{4,4}(@inbounds     BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            bcy_loc    = @inline SMatrix{3,3}(@inbounds     BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typex_loc  = @inline SMatrix{4,4}(@inbounds   type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
            typey_loc  = @inline SMatrix{3,3}(@inbounds   type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            phc_loc    = @inline SMatrix{1,2}(@inbounds  phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            phv_loc    = @inline SMatrix{2,1}(@inbounds  phases.v[ii,jj] for ii in i-1:i-0, jj in j-0:j-0) 
            P_loc     .= @inline SMatrix{3,2}(@inbounds         P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            ΔP_loc    .= @inline SMatrix{1,2}(@inbounds      ΔP.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
            τxx0       = @inline SMatrix{3,2}(@inbounds     τ0.xx[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τyy0       = @inline SMatrix{3,2}(@inbounds     τ0.yy[ii,jj] for ii in i-2:i,   jj in j-1:j  )
            τxy0       = @inline SMatrix{2,3}(@inbounds     τ0.xy[ii,jj] for ii in i-1:i,   jj in j-1:j+1)
            Dc         = @inline SMatrix{1,2}(@inbounds       𝐷.c[ii,jj] for ii in i-1:i-1,   jj in j-1:j)
            Dv         = @inline SMatrix{2,1}(@inbounds       𝐷.v[ii,jj] for ii in i-1:i-0,   jj in j-0:j-0)
            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)
            D          = (c=Dc, v=Dv)
            τ0_loc     = (xx=τxx0, yy=τyy0, xy=τxy0)

            if (j == nc.y+shift.y+1)
                Δy_loc     = zeros(2)
                Δy_loc[1] = Δ.y[j-1]
                Δy_loc[2] = Δ.y[j]
            else
                Δy_loc     = SVector{2}(Δ.y[jj] for jj in j-1:j)
            end

            if (i == nc.x+shift.x)
                Δx_loc     = zeros(4)
                Δx_loc[1] = Δ.x[i-2]
                Δx_loc[2] = Δ.x[i-1]
                Δx_loc[3] = Δ.x[i]
                Δx_loc[4] = Δx_loc[3]
            else
                Δx_loc     = SVector{4}(Δ.x[ii] for ii in i-2:i+1)
            end

            Δt_loc        = Δ.t[1]

            fill!(∂R∂Vx, 0.0)
            fill!(∂R∂Vy, 0.0)
            fill!(∂R∂Pt, 0.0)
            
            autodiff(Enzyme.Reverse, SMomentum_y_Generic_var, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ΔP_loc), Const(τ0_loc), Const(D), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δx_loc), Const(Δy_loc), Const(Δt_loc))
            
            num_Vy = @inbounds num.Vy[i,j]
            bounds_Vy = num_Vy > 0
            # Vy --- Vx
            Local1 = SMatrix{4,4}(num.Vx[ii, jj] for ii in i-2:i+1, jj in j-1:j+2) .* pattern[2][1]
            # for jj in axes(Local1,2), ii in axes(Local1,1)
            #     if (Local1[ii,jj]>0) && bounds_Vy
            #         @inbounds K21[num_Vy, Local1[ii,jj]] = ∂R∂Vx[ii,jj] 
            #     end
            # end
            # Vy --- Vy
            Local2 = SMatrix{3,3}(num.Vy[ii, jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern[2][2]
            # for jj in axes(Local2,2), ii in axes(Local2,1)
            #     if (Local2[ii,jj]>0) && bounds_Vy
            #         @inbounds K22[num_Vy, Local2[ii,jj]] = ∂R∂Vy[ii,jj]  
            #     end
            # end
            # Vy --- Pt
            Local3 = SMatrix{3,2}(num.Pt[ii, jj] for ii in i-2:i, jj in j-1:j) .* pattern[2][3]
            # for jj in axes(Local3,2), ii in axes(Local3,1)
            #     if (Local3[ii,jj]>0) && bounds_Vy
            #         @inbounds K23[num_Vy, Local3[ii,jj]] = ∂R∂Pt[ii,jj]  
            #     end
            # end 

            Base.@nexprs 4 jj -> begin
                Base.@nexprs 4 ii -> begin
                    bounds_Vy && (Local1[ii,jj]>0) && 
                        (@inbounds K21[num_Vy, Local1[ii,jj]] = ∂R∂Vx[ii,jj])
                    
                    bounds_Vy && ii<4 && jj<4 && (Local2[ii,jj]>0) &&
                        (@inbounds K22[num_Vy, Local2[ii,jj]] = ∂R∂Vy[ii,jj])

                    bounds_Vy && ii<4 && jj<3 && (Local3[ii,jj]>0) && 
                        (@inbounds K23[num_Vy, Local3[ii,jj]] = ∂R∂Pt[ii,jj])
                end
            end
        end
    end 
    return nothing
end


#----------------------------
#      Line search
#----------------------------
function LineSearch_var!(rvec, α, dx, R, V, Pt, ε̇, τ, Vi, Pti, ΔPt, Pt0, τ0, λ̇,  η, 𝐷, 𝐷_ctl, number, type, BC, materials, phases, nc, Δ)
    
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, inx_v, iny_v, size_x, size_y, size_c, size_v = Ranges(nc)

    Vi.x .= V.x 
    Vi.y .= V.y 
    Pti  .= Pt
    for i in eachindex(α)
        V.x .= Vi.x 
        V.y .= Vi.y
        Pt  .= Pti
        UpdateSolution_var!(V, Pt, α[i].*dx, number, type, nc)
        TangentOperator_var!(𝐷, 𝐷_ctl, τ, τ0, ε̇, λ̇, η, V, Pt, Pt0, ΔPt, type, BC, materials, phases, Δ)
        ResidualContinuity2D_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ) 
        ResidualMomentum2D_x_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y_var!(R, V, Pt, Pt0, ΔPt, τ0, 𝐷, phases, materials, number, type, BC, nc, Δ)
        rvec[i] = @views norm(R.x[inx_Vx,iny_Vx])/length(R.x[inx_Vx,iny_Vx]) + norm(R.y[inx_Vy,iny_Vy])/length(R.y[inx_Vy,iny_Vy]) + 0*norm(R.p[inx_c,iny_c])/length(R.p[inx_c,iny_c])  
    end
    imin = argmin(rvec)
    V.x .= Vi.x 
    V.y .= Vi.y
    Pt  .= Pti
    return imin
end


function UpdateSolution_var!(V, Pt, dx, number, type, nc)

    nVx, nVy   = maximum(number.Vx), maximum(number.Vy)

    for j=1:size(V.x,2), i=1:size(V.x,1)
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            V.x[i,j] += dx[ind]
        end
    end
 
    for j=1:size(V.y,2), i=1:size(V.y,1)
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            V.y[i,j] += dx[ind]
        end
    end
    
    for I in eachindex(Pt)
        if type.Pt[I] == :in
            ind = number.Pt[I] + nVx + nVy
            Pt[I] += dx[ind]
        end
    end

    # Set E/W periodicity
    for j=2:nc.y+3-1
        if type.Vx[nc.x+3-1,j] == :periodic
            V.x[nc.x+3-1,j] = V.x[2,j]
            V.x[nc.x+3-0,j] = V.x[3,j]
            V.x[       1,j] = V.x[nc.x+3-2,j]
        end
        if type.Vy[nc.x+3,j] == :periodic
            V.y[nc.x+3-0,j] = V.y[3,j]
            V.y[nc.x+3+1,j] = V.y[4,j]
            V.y[1,j]        = V.y[nc.x+3-2,j]
            V.y[2,j]        = V.y[nc.x+3-1,j]
        end
        if j<=nc.y+2
            if type.Pt[nc.x+2,j] == :periodic
                Pt[nc.x+2,j] = Pt[2,j]
                Pt[1,j]      = Pt[nc.x+1,j]
            end
        end
    end 

    # Set S/N periodicity
    for i=2:nc.x+3-1
        if type.Vx[i,nc.y+3] == :periodic
            V.x[i,nc.y+3-0] = V.x[i,3]
            V.x[i,nc.y+3+1] = V.x[i,4]
            V.x[i,1]        = V.x[i,nc.y+3-2]
            V.x[i,2]        = V.x[i,nc.y+3-1]
        end
        if type.Vy[i,nc.y+3-1] == :periodic
            V.y[i,nc.y+3-1] = V.y[i,2]
            V.y[i,nc.y+3-0] = V.y[i,3]
            V.y[i,       1] = V.y[i,nc.y+3-2]
        end
        if i<=nc.x+2
            if type.Pt[i,nc.y+2] == :periodic
                Pt[i,nc.y+2] = Pt[i,2]
                Pt[i,1]      = Pt[i,nc.y+1]
            end
        end
    end

end
