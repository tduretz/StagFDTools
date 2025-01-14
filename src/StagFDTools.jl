module StagFDTools

using StaticArrays, ExtendableSparse

include("Structures.jl")
export Physics, NumberingPoisson, NumberingPoisson2 #, NumberingStokes, StokesPattern

include("Poisson.jl")
export RangesPoisson, NumberingPoisson!, SparsityPatternPoisson, SparsityPatternPoisson_SA

module Stokes
    include("Stokes.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!
end

module TwoPhases
    include("TwoPhases.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!
end

include("Utils.jl")
export printxy

end # module StagFDTools