using Plots, Enzyme, StaticArrays

function StressVectorCartesian!(ε̇, η_n, θ, δ)
    # Transformation from cartesian to material coordinates
    Q         = @SMatrix([cos(θ) sin(θ); -sin(θ) cos(θ)])
    ε̇_tensor  = @SMatrix([ε̇[1] ε̇[3]; ε̇[3] ε̇[2]])
    ε̇_mat     = Q * ε̇_tensor * Q'

    # calculate stress in material coordinates
    τ_mat_vec = @SVector([2 * η_n   * ε̇_mat[1,1],
                          2 * η_n   * ε̇_mat[2,2],
                          2 * η_n/δ * ε̇_mat[1,2]])

    # convert stress to cartesian coordinates
    τ_mat   = @SMatrix([τ_mat_vec[1] τ_mat_vec[3]; τ_mat_vec[3] τ_mat_vec[2]])
    τ_cart  = Q' * τ_mat * Q
    τ_cart_vec = @SVector([τ_cart[1,1], τ_cart[2,2], τ_cart[1,2]])
    return τ_cart_vec
end

function ViscousRheology(θ, η_n, δ, D)
    #= define velocity gradient components and resulting deviatoric strain rate components
    pure shear ε̇ = [ε̇xx 0; 0 -ε̇xx]
    simple shear ε̇ = [0 ε̇xy; ε̇xy 0]
    =#
    Dxx = D[1,1]
    Dyy = - Dxx
    Dxy = D[1,2]
    Dkk = Dxx + Dyy

    ε̇	= @SVector([Dxx - Dkk/3, Dyy - Dkk/3, Dxy])

    D_clt = zeros(3,3)

    jac = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVectorCartesian!, ε̇, Const(η_n), Const(θ), Const(δ))

    D_clt[:,:] .= jac.derivs[1]

    τxx  = jac.val[1]
    τyy  = jac.val[2]
    τxy  = jac.val[3]

    τ_II = sqrt(0.5 * (τxx^2 + τyy^2 + (-τxx - τyy)^2) + τxy^2)
    return τ_II
end

function Analytical(θ, η, δ, D)
    #= define velocity gradient components and resulting deviatoric strain rate components
    pure shear   ε̇ = [ε̇xx  0 ;  0  -ε̇xx]
    simple shear ε̇ = [ 0  ε̇xy; ε̇xy   0 ] =#
    Dxx = D[1,1]
    Dyy = - Dxx
    Dxy = D[1,2]
    Dkk = Dxx + Dyy

    ε̇	= @SVector([Dxx - Dkk/3, Dyy - Dkk/3, Dxy])

    # Normal vector of anisotropic direction
    n1 = -cos(θ)
    n2 = sin(θ)

    # compute isotropic and layered components for 𝐷
    Δ0 = 2 * n1^2 * n2^2
    Δ1 = n1 * n2^3 - n2 * n1^3
    Δ = @SMatrix([ Δ0 -Δ0 2*Δ1; -Δ0 Δ0 -2*Δ1; Δ1 -Δ1 1-2*Δ0])
    A = @SMatrix([ 1 0 0; 0 1 0; 0 0 1] )

    # compute 𝐷
    𝐷 = 2 * η * A - 2 * (η - η/δ) * Δ

    τ = 𝐷 * ε̇

    τ_II = sqrt(0.5 * (τ[1]^2 + τ[2]^2 + (-τ[1] - τ[2])^2) + τ[3]^2)
    return τ_II
end

let
    D = @SMatrix( [1 0; 0 -1] )

    # Anisotropy parameters
    θ = 45
    δ = 10

    ti  = ViscousRheology(θ, ηn, δ, D)
    ana = Analytical(θ, ηn, δ, D)
end