module StagFDTools

using StaticArrays, ExtendableSparse, StaticArrays, Printf, LinearAlgebra

include("operators.jl")
export inn, inn_x, inn_y, av, avx, avy, harm, ∂x, ∂y, ∂x_inn, ∂y_inn, ∂kk

include("Utils.jl")
export printxy, av2D

include("Solvers.jl")
export DecoupledSolver
module Rheology
    using StaticArrays, Enzyme, StagFDTools, LinearAlgebra
    include("Rheology.jl")
    export LocalRheology, StressVector!
    export LocalRheology_phase_ratios, StressVector_phase_ratios! 
    export Kiss2023
end

module Poisson
    using StaticArrays, ExtendableSparse, StaticArrays
    include("Poisson.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!
end
module Stokes
    using LinearAlgebra, StaticArrays, ExtendableSparse, StaticArrays, Enzyme, StagFDTools, StagFDTools.Rheology
    include("Stokes.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!, set_boundaries_template!, SetBCVx1, SetBCVy1
    export Continuity, SMomentum_x_Generic, SMomentum_y_Generic
    export ResidualContinuity2D!, ResidualMomentum2D_x!, ResidualMomentum2D_y!
    export AssembleContinuity2D!, AssembleMomentum2D_x!, AssembleMomentum2D_y!
    export TangentOperator!
    export LineSearch!
end

module StokesJustPIC
    using LinearAlgebra, StaticArrays, ExtendableSparse, StaticArrays, Enzyme, StagFDTools, StagFDTools.Rheology
    using JustPIC, JustPIC._2D
    import JustPIC.@index
    include("StokesJustPIC.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!, set_boundaries_template!, SetBCVx1, SetBCVy1
    export Continuity, SMomentum_x_Generic, SMomentum_y_Generic
    export ResidualContinuity2D!, ResidualMomentum2D_x!, ResidualMomentum2D_y!
    export AssembleContinuity2D!, AssembleMomentum2D_x!, AssembleMomentum2D_y!
    export TangentOperator!
    export LineSearch!
end

module StokesDeformed
    using LinearAlgebra, StaticArrays, ExtendableSparse, StaticArrays, Enzyme, StagFDTools, StagFDTools.Rheology
    include("StokesDeformed.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!, set_boundaries_template!, SetBCVx1, SetBCVy1
    export Continuity, SMomentum_x_Generic, SMomentum_y_Generic
    export ResidualContinuity2D!, ResidualMomentum2D_x!, ResidualMomentum2D_y!
    export AssembleContinuity2D!, AssembleMomentum2D_x!, AssembleMomentum2D_y!
    export TangentOperator!
    export LineSearch!
end

module StokesFSG
    using StaticArrays, ExtendableSparse, StaticArrays, Enzyme
    include("StokesFSG.jl")
    export FSG_Array, Fields, Ranges, Numbering!#, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx!, SetBCVy!
    export AllocateSparseMatrix, Patterns
    export AssembleContinuity2D_1!, AssembleContinuity2D_2!, ResidualContinuity2D_1!, ResidualContinuity2D_2!
    export SetRHS!, UpdateSolution!, SetRHSSG1!, UpdateSolutionSG1!, SetRHSSG2!, UpdateSolutionSG2!
end

module ThermoMechanics
    using StagFDTools, StaticArrays, ExtendableSparse, StaticArrays, LinearAlgebra, Enzyme
    include("ThermoMechanics/ThermoMechanics.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx1, SetBCVy1
    export AssembleHeatDiffusion2D!, ResidualHeatDiffusion2D!, HeatDiffusion
    export AssembleContinuity2D!, ResidualContinuity2D!, Continuity
    export AssembleMomentum2D_y!, ResidualMomentum2D_y!, Momentum_y
    export AssembleMomentum2D_x!, ResidualMomentum2D_x!, Momentum_x
    include("ThermoMechanics/ThermoMechanics_Rheology.jl")
    export LocalRheology, StressVector!, TangentOperator!
end

module TwoPhases
    using StagFDTools, StaticArrays, ExtendableSparse, StaticArrays, LinearAlgebra, Enzyme
    include("TwoPhases/TwoPhases_v2.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx1, SetBCVy1
    export AssembleFluidContinuity2D!, ResidualFluidContinuity2D!, FluidContinuity
    export AssembleContinuity2D!, ResidualContinuity2D!, Continuity
    export AssembleMomentum2D_y!, ResidualMomentum2D_y!, Momentum_y
    export AssembleMomentum2D_x!, ResidualMomentum2D_x!, Momentum_x
    # export AssembleFluidContinuity2D_VE!, ResidualFluidContinuity2D_VE!, FluidContinuity_VE
    # export AssembleContinuity2D_VE!, ResidualContinuity2D_VE!, Continuity_VE
    # include("TwoPhases.jl")
    # export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx1, SetBCVy1
    # export AssembleFluidContinuity2D!, ResidualFluidContinuity2D!, FluidContinuity
    # export AssembleContinuity2D!, ResidualContinuity2D!, Continuity
    # export AssembleMomentum2D_y!, ResidualMomentum2D_y!, Momentum_y
    # export AssembleMomentum2D_x!, ResidualMomentum2D_x!, Momentum_x
    # include("TwoPhases_VE.jl")
    # export AssembleFluidContinuity2D_VE!, ResidualFluidContinuity2D_VE!, FluidContinuity_VE
    # export AssembleContinuity2D_VE!, ResidualContinuity2D_VE!, Continuity_VE
    include("TwoPhases/TwoPhases_Rheology.jl")
    export LocalRheology, StressVector!, TangentOperator!
end

module TwoPhases_v1
    using StagFDTools, StaticArrays, ExtendableSparse, StaticArrays, Enzyme
    include("TwoPhases/TwoPhases_v1.jl")
    export Fields, Ranges, Numbering!, SparsityPattern!, SetRHS!, UpdateSolution!, SetBCVx1, SetBCVy1
    export AssembleFluidContinuity2D!, ResidualFluidContinuity2D!, FluidContinuity
    export AssembleContinuity2D!, ResidualContinuity2D!, Continuity
    export AssembleMomentum2D_y!, ResidualMomentum2D_y!, Momentum_y
    export AssembleMomentum2D_x!, ResidualMomentum2D_x!, Momentum_x
    include("TwoPhases/TwoPhases_VE.jl")
    export AssembleFluidContinuity2D_VE!, ResidualFluidContinuity2D_VE!, FluidContinuity_VE
    export AssembleContinuity2D_VE!, ResidualContinuity2D_VE!, Continuity_VE
end

end # module StagFDTools