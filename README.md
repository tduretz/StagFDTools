# Poisson sparsity examples

##  Flag boundary nodes and constant nodes (e.g. inner BC or free surface)
```julia
[ Info: Node types
6×5 Matrix{Symbol}:
 :Neumann    :Neumann    :Neumann    :Neumann    :Neumann
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