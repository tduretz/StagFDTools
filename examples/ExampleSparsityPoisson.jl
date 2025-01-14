using StagFDTools.Poisson, ExtendableSparse, StaticArrays

let

    # Resolution in FD cells
    nc = (x = 3, y = 4)
        
    # Define node types and set BC flags
    type = Fields( fill(:out, (nc.x+2, nc.y+2)) )
    type.u[2:end-1,2:end-1] .= :in
    type.u[1,:]             .= :periodic 
    type.u[end,:]           .= :periodic 
    type.u[:,1]             .= :Dirichlet
    type.u[:,end]           .= :Neumannn
    
    # 5-point stencil
    pattern = Fields( Fields( @SMatrix([0 1 0; 1 1 1; 0 1 0]) ) )

    # Equation numbering
    number = Fields( fill(0, (nc.x+2, nc.y+2)) )
    Numbering!(number, type, nc)

    # Sparse matrix assembly
    nu  = maximum(number.u)
    M   = Fields( Fields( ExtendableSparseMatrix(nu, nu) ))

    @info "Assembly, ndof  = $(nu)"
    SparsityPattern!(M, number, pattern, nc)
    @info "5-point stencil"
    display(M.u.u)
    display(M.u.u - M.u.u')

    # 9-point stencil
    pattern = Fields( Fields( @SMatrix([1 1 1; 1 1 1; 1 1 1]) ) )
    SparsityPattern!(M, number, pattern, nc)
    @info "9-point stencil"
    display(M.u.u)
    display(M.u.u - M.u.u')             
end