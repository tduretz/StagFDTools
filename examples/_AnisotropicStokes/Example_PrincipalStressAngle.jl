using Enzyme, StaticArrays, LinearAlgebra
import CairoMakie as cm

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
    Pt   = 0

    ε̇vec	= @SVector([Dxx - Dkk/3, Dyy - Dkk/3, Dxy])

    D_clt = zeros(3,3)
    σ1    = zeros(2)
    ε̇1   = zeros(2)

    jac = Enzyme.jacobian(Enzyme.ForwardWithPrimal, StressVectorCartesian!, ε̇vec, Const(η_n), Const(θ), Const(δ))

    D_clt[:,:] .= jac.derivs[1]

    τxx  = jac.val[1]
    τyy  = jac.val[2]
    τxy  = jac.val[3]

    τ_II   = sqrt(0.5 * (τxx^2 + τyy^2 + (-τxx - τyy)^2) + τxy^2)

    # compute principal directions of stress and strain rate tensors
    σ         = @SMatrix[-Pt+τxx τxy 0.; τxy -Pt+τyy 0.; 0. 0. -Pt+(-τxx-τyy)]
    v         = eigvecs(σ)
    scale     = sqrt(v[1,1]^2 + v[2,1]^2)
    σ1[1] = v[1,1]/scale
    σ1[2] = v[2,1]/scale

    ε̇         = @SMatrix[ε̇vec[1] ε̇vec[3] 0.; ε̇vec[3] ε̇vec[2] 0.; 0. 0. (-ε̇vec[1]-ε̇vec[2])]
    ve        = eigvecs(ε̇)
    scalee    = sqrt(ve[1,1]^2 + ve[2,1]^2)
    ε̇1[1] = ve[1,1]/scalee
    ε̇1[2] = ve[2,1]/scalee

    εxx = ε̇vec[1]
    εyy = ε̇vec[2]
    εxy = ε̇vec[3]
    

    # compute angle between σ1 and ε̇1 (in radians)
    α = atan(ε̇1[2], ε̇1[1]) - atan(σ1[2], σ1[1])
    # mathematical 
    α2 = atan(2 * εxy ./ (εxx - εyy)) / 2 - atan((2 * εxy .* (δ + (1 - δ) .* cos(4 * θ) + 1) + (δ - 1) .* (εxx - εyy) .* sin(4 * θ)) ./ (2 * εxy .* (δ - 1) .* sin(4 * θ) + (εxx - εyy) .* (δ + (δ - 1) .* cos(4 * θ) + 1))) / 2
    # Visualize σ1 and ε̇1 directions
    # fig = cm.Figure()
    # ax  = cm.Axis(fig[1,1], aspect=1)
    # cm.xlims!(ax, -0.5, 0.5)
    # cm.ylims!(ax, -0.5, 0.5)
    # cm.arrows2d!(ax, [0], [0], [σ1[1]], [σ1[2]], align = :center, tiplength = 0, lengthscale=0.7, tipwidth=1, color=:black)
    # cm.arrows2d!(ax, [0], [0], [ε̇1[1]], [ε̇1[2]], align = :center, tiplength = 0, lengthscale=0.5, tipwidth=1, color=:red)
    # display(fig)

    return σ1, ε̇1, τ_II, α, α2
end

let
    D = @SMatrix( [1 0; 0 -1] )

    # Anisotropy parameters
    ηn = 2.0
    θ  = LinRange(0, π, 181)
    δ  = LinRange(1, 100, 100)

    σ1    = [@MVector(zeros(2)) for _ in eachindex(θ), _ in eachindex(δ)]
    ε̇1    = [@MVector(zeros(2)) for _ in eachindex(θ), _ in eachindex(δ)]
    τ_II  = zeros(length(θ), length(δ))
    α = zeros(length(θ), length(δ))
    α2 = zeros(length(θ), length(δ))

    for iθ in eachindex(θ) , iδ in eachindex(δ)
        σ1[iθ, iδ], ε̇1[iθ, iδ], τ_II[iθ, iδ], α[iθ, iδ], α2[iθ,iδ] = ViscousRheology(θ[iθ], ηn, δ[iδ], D)
    end
    
    fig = cm.Figure()
    ax  = cm.Axis(fig[1,1], aspect=cm.DataAspect(), xlabel="θ [°]", ylabel="α [°]")
    δind = 10
    cm.lines!(ax, θ*180/π, α[:,δind]*180/π, color=:blue, label="δ=$(δ[δind])")
    cm.scatter!(ax, θ[1:5:end]*180/π, α2[1:5:end,δind]*180/π, color=:orange, label="δ=$(δ[δind])")
    cm.axislegend(ax)
    display(fig)
    fig = cm.Figure()
    ax  = cm.Axis(fig[1,1], aspect=cm.DataAspect(), xlabel="δ", ylabel="α [°]")
    θind = 45
    cm.lines!(ax, δ, α[θind,:]*180/π, color=:red, label="θ=$(θ[θind]*180/π)°")
    cm.scatter!(ax, δ[1:5:end], α2[θind,1:5:end]*180/π, color=:green, label="θ=$(θ[θind]*180/π)°")
    cm.axislegend(ax)
    display(fig)

    fig2 = cm.Figure()
    ax2  = cm.Axis(fig2[1,1], aspect=1, xlabel="θ [°]", ylabel="δ")
    hm   = cm.heatmap!(ax2, θ*180/π, δ, α*180/π, colormap=:bluesreds)
    # hm2  = cm.heatmap!(ax2, θ*180/π, δ, (α-α2)*180/π, colormap=:bluesreds)
    cm.Colorbar(fig2[1,2], hm, label="α [°]")
    display(fig2)
end