using StagFDTools.Stokes, ExtendableSparse, StaticArrays

let

    # Resolution
    nc = (x = 4, y = 3)
    
    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    # -------- Vx -------- #
    type.Vx[2:end-1,3:end-2] .= :in       
    type.Vx[2,2:end-1]       .= :Dirichlet_normal 
    type.Vx[end-1,2:1:end-1] .= :Dirichlet_normal 
    type.Vx[2:end-1,2]       .= :Dirichlet
    type.Vx[2:end-1,end-1]   .= :Dirichlet
    # -------- Vy -------- #
    type.Vy[2:end-2,2:end-1] .= :in       
    type.Vy[2,2:end-1]       .= :Dirichlet
    type.Vy[end-1,2:end-1]   .= :Dirichlet
    type.Vy[2:end-1,2]       .= :Dirichlet_normal 
    type.Vy[2:end-1,end-1]   .= :Dirichlet_normal 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in
    
    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([0 1 0; 1 1 1; 0 1 0]),                 @SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]), @SMatrix([0 1 0;  0 1 0])), 
        Fields(@SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]),  @SMatrix([0 1 0; 1 1 1; 0 1 0]),                @SMatrix([0 0; 1 1; 0 0])), 
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
    )

    # Equation numbering
    number = Fields(
        fill(0, (nc.x+3, nc.y+4)),
        fill(0, (nc.x+4, nc.y+3)),
        fill(0, (nc.x+2, nc.y+2)),
    )
    Numbering!(number, type, nc)

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )

    @info "Assembly, ndof  = $(nVx + nVy + nPt)"
    SparsityPattern!(M, number, pattern, nc)

    # Stokes operator as block matrices
    K  = [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
    Q  = [M.Vx.Pt; M.Vy.Pt]
    Qᵀ = [M.Pt.Vx M.Pt.Vy]

    @info "Velocity block symmetry"
    display(K - K')

    @info "Grad-Div symmetry"
    display(Q' - Qᵀ)

end