using Plots

function main()

    ncx, ncy = 1000, 1000

    xmin, xmax = -0.5, .5
    ymin, ymax = -0.5, .5
    dx = (xmax-xmin)/ncx
    dy = (ymax-ymin)/ncy
    xc = LinRange(xmin+dx/2, xmax-dx/2, ncx)
    yc = LinRange(ymin+dy/2, ymax-dy/2, ncy)

    phase = ones(ncx, ncy)


    X = xc .+ 0 .* yc'
    Y = 0 .* xc .+ yc'

    # Ellipse 1
    x0, y0 = 0., 0.
    α  = 30.0
    ar = 100.0
    r  = 0.007
    𝑋 = cosd(α)*X .- sind(α).*Y
    𝑌 = sind(α)*X .+ cosd(α).*Y
    phase[ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 0.0

    # Ellipse 1
    x0, y0 = 0.25, 0.
    α  = -80.0
    ar = 150.0
    r  = 0.005
    𝑋 = cosd(α)*X .- sind(α).*Y
    𝑌 = sind(α)*X .+ cosd(α).*Y
    phase[ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 0.0

    # Ellipse 3
    x0, y0 = -0.15, 0.
    α  = -30.0
    ar = 100.0
    r  = 0.005
    𝑋 = cosd(α)*X .- sind(α).*Y
    𝑌 = sind(α)*X .+ cosd(α).*Y
    phase[ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 0.0

    # Ellipse 4
    x0, y0 = 0.35, -0.3
    α  = 86.0
    ar = 200.0
    r  = 0.005
    𝑋 = cosd(α)*X .- sind(α).*Y
    𝑌 = sind(α)*X .+ cosd(α).*Y
    phase[ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 0.0
 
    # Ellipse 5
    x0, y0 = 0.35, -0.3
    α  = -20.0
    ar = 250.0
    r  = 0.01
    𝑋 = cosd(α)*X .- sind(α).*Y
    𝑌 = sind(α)*X .+ cosd(α).*Y
    phase[ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 0.0
 
    # Ellipse 5
    x0, y0 = -0.35, -0.3
    α  = 15.0
    ar = 200.0
    r  = 0.004
    𝑋 = cosd(α)*X .- sind(α).*Y
    𝑌 = sind(α)*X .+ cosd(α).*Y
    phase[ ((𝑋 .- x0).^2 .+ (𝑌  .- y0).^2/(ar)^2) .< r^2] .= 0.0
  
    ϕ = sum(phase.==0)/ *(size(phase)...)
    @show ϕ

    heatmap(phase')

end

main()