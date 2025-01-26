# StagFDTools

This package aims at generating flexible FD stencils that can interoperate with AD tools for Jacobian generation. In particular, the aims are:
- generic treatment of BCs
- generation of equation numbering
- generation of sparsity patterns
- flexible treatment of multi-physics

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

# Power-law Stokes (standard stagerred grid - asymmetric operator)
![image](https://github.com/user-attachments/assets/e29d72a5-93cf-4cc5-84a1-c353a05a4edb)

# Power-law Stokes (full stagerred grid - symmetric operator)
![image](https://github.com/user-attachments/assets/9c1e02d5-6b7f-4764-a99d-12a87e28ea21)

# Anisotropic Stokes (full stagerred grid - symmetric operator)
![image](https://github.com/user-attachments/assets/3df8215a-0eca-4e3e-b01a-85a501a4bacb)

# Two-phase flow
![image](https://github.com/user-attachments/assets/e5606f59-1a56-43e8-84d9-25381318eb0c)

