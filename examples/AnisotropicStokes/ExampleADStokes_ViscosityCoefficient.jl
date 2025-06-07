using Plots, Enzyme, StaticArrays

function StressVectorCartesian!(ε̇, η, θ, δ)
    # Transformation from cartesian to material coordinates
        Q         = @SMatrix([cos(θ) sin(θ); -sin(θ) cos(θ)])
        ε̇_tensor  = @SMatrix([ε̇[1] ε̇[3]; ε̇[3] ε̇[2]])
        ε̇_mat     = Q * ε̇_tensor * Q'

        # calculate stress in material coordinates
        τ_mat_vec = @SVector([2 * η   * ε̇_mat[1,1],
                     2 * η   * ε̇_mat[2,2],
                     2 * η/δ * ε̇_mat[1,2]])

        # convert stress to cartesian coordinates
        τ_mat   = @SMatrix([τ_mat_vec[1] τ_mat_vec[3]; τ_mat_vec[3] τ_mat_vec[2]])
        τ_cart  = Q' * τ_mat * Q
        τ_cart_vec = @SVector([τ_cart[1,1], τ_cart[2,2], τ_cart[1,2]])
        return τ_cart_vec
end

function ViscousRheology(θ)
    #  Material parameters
    η_n = 1.0  # normal viscosity
    δ   = 1    # anisotropy factor

    #= define velocity gradient components and resulting deviatoric strain rate components
    pure shear ε̇ = [ε̇xx 0; 0 -ε̇xx]
    simple shear ε̇ = [0 ε̇xy; ε̇xy 0]
    =#
    pureshear = 1 # = 0 for simple shear
    Dxx = pureshear * 0.5
    Dyy = -Dxx
    Dxy = (1-pureshear) * 3.0
    Dkk = Dxx + Dyy

    ε̇	= @SVector([Dxx - Dkk / 3, Dyy - Dkk / 3, (Dxy + Dxy) / 2])
    
    # θ = LinRange(0.0, π, 180)       # Angle of foliation

    D_clt = zeros(3,3)

    jac = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVectorCartesian!, ε̇, Const(η_n), Const(θ), Const(δ))

    D_clt[:,:] .= jac.derivs[1]
    @show D_clt
    # why is it not [2η 0 0; 0 2η 0; 0 0 2η] for all θ at δ = 1.0?
end

ViscousRheology(0)