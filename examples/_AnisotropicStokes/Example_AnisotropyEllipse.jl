using Plots

let
    #  Material parameters
    η_n = 1.0  # normal viscosity
    δ   = 2    # anisotropy factor

    # define velocity gradient components and resulting deviatoric strain rate components
    #= 
    pure shear ε̇ = [ε̇xx 0; 0 -ε̇xx]
    simple shear ε̇ = [0 ε̇xy; ε̇xy 0]
    =#
    pureshear = 1 # simple shear = 0
    Dxx = pureshear * 0.5
    Dyy = -Dxx
    Dxy = (1-pureshear) * 2
    Dkk = Dxx + Dyy

    ε̇xx = Dxx - Dkk / 3.0
    ε̇yy = Dyy - Dkk / 3.0
    ε̇xy = (Dxy  + Dxy) / 2.0
    
    θ = LinRange(0.0, π, 180)       # Angle of foliation
    # Initialize stress components
    τxx  = zero(θ)
    τyy  = zero(θ)
    τxy  = zero(θ)
    τ_II = zero(θ)

    τxx_cartesian  = zero(θ)
    τyy_cartesian  = zero(θ)
    τxy_cartesian  = zero(θ)
    τ_II_cartesian = zero(θ)

    for i in eachindex(θ)
        # Transformation from cartesian to material coordinates
        Q         = [cos(θ[i]) sin(θ[i]); -sin(θ[i]) cos(θ[i])]
        ε̇_tensor  = [ε̇xx ε̇xy; ε̇xy ε̇yy]
        ε̇_mat     = Q * ε̇_tensor * Q'
        ε̇_mat_vec = [ε̇_mat[1, 1], ε̇_mat[2, 2], ε̇_mat[1, 2]]  # Voigt notation

        # calculate stress in material coordinates
        τ_mat_vec = [2 * η_n * ε̇_mat_vec[1],
                 2 * η_n * ε̇_mat_vec[2],
                 2 * η_n / δ * ε̇_mat_vec[3]]

        # convert stress to cartesian coordinates
        τ_mat       = [τ_mat_vec[1] τ_mat_vec[3]; τ_mat_vec[3] τ_mat_vec[2]]
        τ_cartesian = Q' * τ_mat * Q

        τxx[i]            = τ_mat_vec[1]
        τyy[i]            = τ_mat_vec[2]
        τxy[i]            = τ_mat_vec[3]
        τxx_cartesian[i]  = τ_cartesian[1, 1]
        τyy_cartesian[i]  = τ_cartesian[2, 2]
        τxy_cartesian[i]  = τ_cartesian[1, 2]

        τ_II[i]           = sqrt(0.5 * (τxx[i]^2 + τyy[i]^2 + (-τxx[i] - τyy[i])^2) + δ^2 * τxy[i]^2)
        τ_II_cartesian[i] = sqrt(0.5 * (τxx_cartesian[i]^2 + τyy_cartesian[i]^2 + (-τxx_cartesian[i] - τyy_cartesian[i])^2) + τxy_cartesian[i]^2)
    end
    p1 = plot( xlabel="τₓₓ'", ylabel="τₓᵧ'", aspect_ratio=1.0, legend=false, title="Deviatoric stress components (δ = $δ)")
    p1 = plot!(τxx,τxy, label="τII'", color=:red)
    p2 = plot(xlabel="θ [deg.]", ylabel="τᵢᵢ, τᵢᵢ' [-]", title="τᵢᵢ and stress invariant τᵢᵢ'")
    p2 = plot!(θ * 180 / π, τ_II_cartesian, label="τII (δ = $δ)")
    p2 = plot!(θ * 180 / π, τ_II, label="τII' (δ = $δ)")
    display(plot(p2, p1, layout = (2, 1)))
end
