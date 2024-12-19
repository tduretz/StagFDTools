Base.@kwdef mutable struct Physics
    Poisson         ::Bool = false
    Stokes          ::Bool = false
    TwoPhases       ::Bool = false
    Cosserat        ::Bool = false
    Thermal         ::Bool = false
end

# Base.@kwdef mutable struct NumberingPoisson
#     Num     ::Union{Matrix{Int64},  Missing} = missing
#     Type    ::Union{Matrix{Symbol}, Missing} = missing
#     Pattern ::Union{SMatrix,        Missing} = missing
# end

Base.@kwdef mutable struct NumberingPoisson{N}
    num     ::Union{Matrix{Int64},   Missing} = missing
    type    ::Union{Matrix{Symbol},  Missing} = missing
    bc_val  ::Union{Matrix{Float64}, Missing} = missing
    pattern ::Union{SMatrix{N, N, Int64},  Missing} = missing
end