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
    pattern ::Union{SMatrix{N, N, Int64},  Missing} = missing
end

Base.@kwdef mutable struct StokesPattern
    num     ::Union{Matrix{Int64},   Missing} = missing
    type    ::Union{Matrix{Symbol},  Missing} = missing
    patternVx ::Union{SMatrix,  Missing} = missing  # these could be contained in same object - but they have different sizes
    patternVy ::Union{SMatrix,  Missing} = missing  # these could be contained in same object - but they have different sizes
    patternPt ::Union{SMatrix,  Missing} = missing  # these could be contained in same object - but they have different sizes
    # ideally we would like other fields also part of it (fluid pressure, microrotation)
end

Base.@kwdef mutable struct NumberingStokes
    Vx ::Union{StokesPattern, Missing} = missing
    Vy ::Union{StokesPattern, Missing} = missing
    Pt ::Union{StokesPattern, Missing} = missing
end