using StagFDTools, ExtendableSparse, StaticArrays

let
    # Generates an empty numbering structure
    numbering = NumberingPoisson{3}()
    
    # Resolution in FD cells
    nc = (x = 3, y = 4)
    
    # Define node types and set BC flags
    numbering.type = fill(:out, (nc.x+2, nc.y+2))
    numbering.type[2:end-1,2:end-1] .= :in
    numbering.type[1,:]             .= :periodic 
    numbering.type[end,:]           .= :periodic 
    numbering.type[:,1]             .= :Dirichlet
    numbering.type[:,end]           .= :Neumannn
    
    @info "Node types"
    Print_xy(numbering.type) 

    # 5-point stencil
    numbering.pattern = @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
    NumberingPoisson!(numbering, nc)
    @time K    = SparsityPatternPoisson(numbering, nc)
    @time K_SA = SparsityPatternPoisson_SA(numbering.num, numbering.pattern, nc)
    @assert K == K_SA
    @info "5-point stencil"
    display(K)
    display(K-K')

    # 9-point stencil
    pattern = @SMatrix([1 1 1; 1 1 1; 1 1 1]) 
    NumberingPoisson!(numbering, nc)
    K    = SparsityPatternPoisson(numbering, nc)
    K_SA = SparsityPatternPoisson_SA(numbering.num, numbering.pattern, nc)
    @assert K == K_SA
    @info "9-point stencil"
    display(K)
    display(K-K')               
end