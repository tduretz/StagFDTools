using StagFDTools, ExtendableSparse, StaticArrays

let

    physics = Physics()
    physics.Stokes = true
    
    # Resolution
    nc = (x = 10, y = 9)

    numbering = numberingStokes()
    
    # Define node types and set BC flags
    numbering.Vx      = NumberingPoisson{3}()
    numbering.Vx.type = fill(:out, (nc.x+3, nc.y+4))
    numbering.Vx.type[2:end-1,2:end-2] .= :in
    numbering.Vx.type[1,:]     .= :periodic # make periodic
    numbering.Vx.type[end,:]   .= :periodic 
    numbering.Vx.type[:,1]     .= :Dirichlet
    numbering.Vx.type[:,end]   .= :Neumann
    @info "Node types"
    Print_xy(numbering.Vx.type) 

    # if physics.Poisson
    #     # 5-point stencil
    #     pattern = @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
    #     num  = NumberingPoisson(nc, type)
    #     @time K    = SparsiTypatternPoisson(nc, num, pattern)
    #     @time K_SA = SparsiTypatternPoisson_SA(nc, num, pattern)
    #     @assert K == K_SA
    #     @info "5-point stencil"
    #     display(K)
    #     display(K-K')

    #     # 9-point stencil
    #     pattern = @SMatrix([1 1 1; 1 1 1; 1 1 1]) 
    #     num = NumberingPoisson(nc, type)
    #     K = SparsiTypatternPoisson(nc, num, pattern)
    #     K_SA = SparsiTypatternPoisson_SA(nc, num, pattern)
    #     @assert K == K_SA
    #     @info "9-point stencil"
    #     display(K)
    #     display(K-K')               
    # end
end