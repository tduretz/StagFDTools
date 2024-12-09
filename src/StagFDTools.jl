module StagFDTools

using StaticArrays, ExtendableSparse

include("Structures.jl")
export Physics, NumberingPoisson, numberingStokes

include("Poisson.jl")
export NumberingPoisson!, SparsityPatternPoisson, SparsityPatternPoisson_SA

include("Utils.jl")
export Print_xy

end # module StagFDTools
