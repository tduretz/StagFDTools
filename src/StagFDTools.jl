module StagFDTools

using StaticArrays, ExtendableSparse, StaticArrays, Printf, LinearAlgebra

include("operators.jl")
export inn, inn_x, inn_y, av, harm, ∂x, ∂y, ∂x_inn, ∂y_inn, ∂kk

include("Utils.jl")
export printxy

include("Solvers.jl")
export DecoupledSolver

module Poisson
    using StaticArrays, ExtendableSparse, StaticArrays
    include("Poisson.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!
end
module Stokes
    using StaticArrays, ExtendableSparse, StaticArrays
    include("Stokes.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!
end

module StokesFSG
    using StaticArrays, ExtendableSparse, StaticArrays, Enzyme
    include("StokesFSG.jl")
    export FSG_Array, Fields, Ranges, Numbering!#, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!
    export AllocateSparseMatrix, Patterns
    export AssembleContinuity2D_1!, AssembleContinuity2D_2!, ResidualContinuity2D_1!, ResidualContinuity2D_2!
    export SetRHS!, UpdateSolution!, SetRHSSG1!, UpdateSolutionSG1!, SetRHSSG2!, UpdateSolutionSG2!
end

module TwoPhases
    using StaticArrays, ExtendableSparse, StaticArrays
    include("TwoPhases.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!
end

end # module StagFDTools