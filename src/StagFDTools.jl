module StagFDTools

using StaticArrays, ExtendableSparse

include("Structures.jl")
export Physics, NumberingPoisson, NumberingPoisson2 #, NumberingStokes, StokesPattern

include("Poisson.jl")
export RangesPoisson, NumberingPoisson!, SparsityPatternPoisson, SparsityPatternPoisson_SA

include("Utils.jl")
export printxy

end # module StagFDTools