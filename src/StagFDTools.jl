module StagFDTools

using StaticArrays, ExtendableSparse, StaticArrays, Printf, LinearAlgebra

include("operators.jl")
export inn, inn_x, inn_y, av, harm, ∂x, ∂y, ∂x_inn, ∂y_inn, ∂kk

include("Utils.jl")
export printxy, av2D

include("Solvers.jl")
export DecoupledSolver

module Poisson
    using StaticArrays, ExtendableSparse, StaticArrays
    include("Poisson.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!
end
module Stokes
    using StaticArrays, ExtendableSparse, StaticArrays, Enzyme, StagFDTools
    include("Stokes.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!, set_boundaries_template!, SetBCVx1, SetBCVy1
    export Continuity, SMomentum_x_Generic, SMomentum_y_Generic
    export ResidualContinuity2D!, ResidualMomentum2D_x!, ResidualMomentum2D_y!
    export AssembleContinuity2D!, AssembleMomentum2D_x!, AssembleMomentum2D_y!
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
    using StaticArrays, ExtendableSparse, StaticArrays, Enzyme
    include("TwoPhases.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution! #, SetBCVx!, SetBCVy!
    export AssembleFluidContinuity2D!, ResidualFluidContinuity2D!, FluidContinuity
    export AssembleContinuity2D!, ResidualContinuity2D!, Continuity
    export AssembleMomentum2D_y!, ResidualMomentum2D_y!, Momentum_y
    export AssembleMomentum2D_x!, ResidualMomentum2D_x!, Momentum_x
    include("TwoPhases_VE.jl")
    export AssembleFluidContinuity2D_VE!, ResidualFluidContinuity2D_VE!, FluidContinuity_VE
    export AssembleContinuity2D_VE!, ResidualContinuity2D_VE!, Continuity_VE
end

module Rheology
    using StaticArrays, Enzyme, StagFDTools.Stokes, StagFDTools, LinearAlgebra
    include("Rheology.jl")
    export LocalRheology, StressVector!, TangentOperator!, LineSearch!
    export Kiss2023
end

end # module StagFDTools