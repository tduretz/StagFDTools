
function Stress( ε̇, η )
    return [2*η*ε̇[1], 2*η*ε̇[2], 2*η*ε̇[3]]
end

function Stress!( τ, ε̇, η )
    
    τ .= [2*η1*ε̇[1], 2*η1*ε̇[2], 2*η1*ε̇[3]]
end

f(x) = [2*x[1], 2*x[2], 1*x[3]]

let 
    η = 1.0
    ε̇ = [1., 1., 1.]
    @show Stress( ε̇, η )

    @show Enzyme.jacobian(Enzyme.Forward, Stress, ε̇, 1.0)

    # τ = [0., 0., 0.]
    # @show Enzyme.jacobian(Enzyme.Forward, Stress!, Val(τ), Const(ε̇), Const(1.0))

end