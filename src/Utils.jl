printxy(x) = display( rotr90(x[end:-1:1,end:-1:1]) )
av2D(x) = @views @. 0.25*(x[1:end-1,1:end-1] + x[2:end-0,1:end-1,] + x[1:end-1,2:end-0] + x[2:end-0,2:end-0])


@views function GenerateGrid(x, y, Δ, nc)

    X = (
        v = (
            x = LinRange(x.min, x.max, nc.x+1),
            y = LinRange(y.min, y.max, nc.y+1),
        ),
        # With ghost vertices
        v_e = (
            x = LinRange(x.min-Δ.x, x.max+Δ.x, nc.x+3),
            y = LinRange(y.min-Δ.y, y.max+Δ.y, nc.y+3),
        ),
        c = (
            x = LinRange(x.min+Δ.x/2, x.max-Δ.x/2, nc.x),
            y = LinRange(y.min+Δ.y/2, y.max-Δ.y/2, nc.y),
        ),
        # With ghost centroids
        c_e = (
            x = LinRange(x.min-Δ.x/2, x.max+Δ.x/2, nc.x+2),
            y = LinRange(y.min-Δ.y/2, y.max+Δ.y/2, nc.y+2),
        ),
        vx = (
            x = LinRange(x.min, x.max, nc.x+1),
            y = LinRange(y.min+Δ.y/2, y.max-Δ.y/2, nc.y),
        ),
        vy = (
            x = LinRange(x.min+Δ.x/2, x.max-Δ.x/2, nc.x),
            y = LinRange(y.min, y.max, nc.y+1),

        )
    )

    return X
end
