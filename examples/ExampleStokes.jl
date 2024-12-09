using StagFDTools, ExtendableSparse, StaticArrays

let

    physics = Physics()
    physics.Stokes = true
    
    # Resolution
    nc = (x = 10, y = 9)

    Numbering = NumberingStokes()
    
    # 5-point stencil
    # Type = fill(:out, (nc.x+2, nc.y+2))
    # Type[2:end-1,2:end-1] .= :in
    # Type[1,:]     .= :periodic # make periodic
    # Type[end,:]   .= :periodic 
    # Type[:,1]     .= :Dirichlet
    # Type[:,end]   .= :Neumann
    # @info "Node types"
    # Print_xy(Type) 

    # if physics.Poisson
    #     # 5-point stencil
    #     Pattern = @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
    #     Num  = NumberingPoisson(nc, Type)
    #     @time K    = SparsiTyPatternPoisson(nc, Num, Pattern)
    #     @time K_SA = SparsiTyPatternPoisson_SA(nc, Num, Pattern)
    #     @assert K == K_SA
    #     @info "5-point stencil"
    #     display(K)
    #     display(K-K')

    #     # 9-point stencil
    #     Pattern = @SMatrix([1 1 1; 1 1 1; 1 1 1]) 
    #     Num = NumberingPoisson(nc, Type)
    #     K = SparsiTyPatternPoisson(nc, Num, Pattern)
    #     K_SA = SparsiTyPatternPoisson_SA(nc, Num, Pattern)
    #     @assert K == K_SA
    #     @info "9-point stencil"
    #     display(K)
    #     display(K-K')               
    # end
end