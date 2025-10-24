printxy(x) = display( rotr90(x[end:-1:1,end:-1:1]) )
av2D(x) = @views @. 0.25*(x[1:end-1,1:end-1] + x[2:end-0,1:end-1,] + x[1:end-1,2:end-0] + x[2:end-0,2:end-0])


@views function GenerateGrid(x, y, Δ, nc)

    X = (
        # in = (
            v = (
                x = LinRange(x.min, x.max, nc.x+1),
                y = LinRange(y.min, y.max, nc.y+1),
            ),
            c = (
                x = LinRange(x.min+Δ.x/2, x.max-Δ.x/2, nc.x),
                y = LinRange(y.min+Δ.y/2, y.max-Δ.y/2, nc.y),
            ),
            vx = (
                x = LinRange(x.min, x.max, nc.x+1),
                y = LinRange(y.min+Δ.y/2, y.max-Δ.y/2, nc.y),
            ),
            vy = (
                x = LinRange(x.min+Δ.x/2, x.max-Δ.x/2, nc.x),
                y = LinRange(y.min, y.max, nc.y+1),

            )
        # ),

        # ex = (
        #     xvx = LinRange(x.min-Δ.x, x.max+Δ.x, nc.x+3),
        #     xvy = LinRange(x.min-3Δ.x/2, x.max+3Δ.x/2, nc.x+4),
        #     yvy = LinRange(y.min-Δ.y, y.max+Δ.y, nc.y+3),
        #     yvx = LinRange(y.min-3Δ.y/2, y.max+3Δ.y/2, nc.y+4),
        # )
    )

    return X
end
