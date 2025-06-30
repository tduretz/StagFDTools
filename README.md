# StagFDTools

This package aims at generating flexible FD stencils that can interoperate with AD tools for Jacobian generation. In particular, the aims are:
- generic treatment of BCs
- generation of equation numbering
- generation of sparsity patterns
- flexible treatment of multi-physics

# Get started

0) Make sure you have Julia and `VSCode` or `VSCodium` working together. *Recommended way: (1) Install Julia using [`juliaup`](https://github.com/JuliaLang/juliaup), (2) install [`VSCodium`](https://vscodium.com), (3) install `Julia Language Support` in `VSCode` extensions.*

1) In your general Julia environment, make sure you have plotting packages installed.
*How: (1) In Julia's terminal (REPL), type: `]` to go to package mode (`(@v1.11) pkg>`). (2) type `add Plots` and then `add Makie`.*

2) Clone this repository somewhere on your computer.

3) In `VSCodium`, click on `File`, then `New Window`. Click on `Open...` and select the `StagFDTools` folder. *Achtung: At this point, you should agree to select `StagFDTools`'s environment*.

4) Go to package mode: type `]`. It should clearly indicate that you're working in the correct environment (`(StagFDTools) pkg>`). If not, repeat step 3. If yes, type `instantiate`.

5) Now, you should be able to run the scripts in the `example/` folder.


# Poisson sparsity examples

##  Flag boundary nodes and constant nodes (e.g. inner BC or free surface)
```julia
[ Info: Node types
6×5 Matrix{Symbol}:
 :Neumannn    :Neumannn    :Neumannn    :Neumannn    :Neumannn
 :periodic   :in         :in         :in         :periodic
 :periodic   :in         :in         :in         :periodic
 :periodic   :in         :in         :in         :periodic
 :periodic   :in         :in         :in         :periodic
 :Dirichlet  :Dirichlet  :Dirichlet  :Dirichlet  :Dirichlet
```

## Generation of a 5-point stencil including a symmetry test
```julia 
[ Info: 5-point stencil
12×12 ExtendableSparseMatrixCSC{Float64, Int64} with 54 stored entries:
 1.0  1.0  1.0  1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0  1.0  1.0   ⋅   1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0  1.0  1.0   ⋅    ⋅   1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0   ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅   1.0   ⋅   1.0  1.0  1.0   ⋅   1.0   ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅   1.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅   1.0   ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅   1.0  1.0  1.0   ⋅   1.0   ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅   1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0   ⋅    ⋅   1.0  1.0  1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0   ⋅   1.0  1.0  1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0  1.0  1.0  1.0
12×12 SparseArrays.SparseMatrixCSC{Float64, Int64} with 0 stored entries:
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
```

## Generation of a 9-point stencil including a symmetry test
```julia
[ Info: 9-point stencil
12×12 ExtendableSparseMatrixCSC{Float64, Int64} with 54 stored entries:
 1.0  1.0  1.0  1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0  1.0  1.0   ⋅   1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0  1.0  1.0   ⋅    ⋅   1.0   ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
 1.0   ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅   1.0   ⋅   1.0  1.0  1.0   ⋅   1.0   ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅   1.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅   1.0   ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅   1.0  1.0  1.0   ⋅   1.0   ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0  1.0  1.0  1.0   ⋅    ⋅   1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0   ⋅    ⋅   1.0  1.0  1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0   ⋅   1.0  1.0  1.0
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅   1.0  1.0  1.0  1.0
12×12 SparseArrays.SparseMatrixCSC{Float64, Int64} with 0 stored entries:
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅    ⋅ 
```
## Power-law Stokes (standard staggered grid - asymmetric operator)

![image](./results/PowerLaw.gif)
<!-- 
![image](https://github.com/user-attachments/assets/e29d72a5-93cf-4cc5-84a1-c353a05a4edb) -->

## Shear banding with compressibility and dilatancy
![image](./results/ShearBanding.gif)

## Pressurized holed with mixed-mode plasticity (see Kiss et al., 2023)
![image](./results/PressurizedHole.gif)

## Host-inclusion decompression with Drucker-Prager plasticity
![image](./results/HostInclusion_DruckerPrager.gif)

## Host-inclusion decompression with tensile plasticity
![image](./results/HostInclusion_tensile.gif)

## Power-law Stokes (full staggered grid - symmetric operator)
![image](https://github.com/user-attachments/assets/9c1e02d5-6b7f-4764-a99d-12a87e28ea21)

## Anisotropic Stokes (full staggered grid - symmetric operator)
![image](https://github.com/user-attachments/assets/3df8215a-0eca-4e3e-b01a-85a501a4bacb)

## Diffusion (backward Euler)
![image](results/Diffusion_BackwardEuler.svg)

## Diffusion (Crank-Nicolson)
![image](results/Diffusion_CrankNicolson.svg)

# Two-phase flow
![image](https://github.com/user-attachments/assets/e5606f59-1a56-43e8-84d9-25381318eb0c)

