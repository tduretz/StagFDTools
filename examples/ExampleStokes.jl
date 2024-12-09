using StagFDTools, ExtendableSparse, StaticArrays

let

    physics = Physics()
    physics.Stokes = true
    
    # Resolution
    nc = (x = 10, y = 9)

    numbering = NumberingStokes()
    
    # Define node types and set BC flags
    numbering.Vx      = NumberingPoisson{3}()
    numbering.Vx.type = fill(:out, (nc.x+3, nc.y+4))
    numbering.Vx.type[2:end-1,2:end-2] .= :in
    numbering.Vx.type[1,:]     .= :periodic # make periodic
    numbering.Vx.type[end,:]   .= :periodic 
    numbering.Vx.type[:,1]     .= :Dirichlet
    numbering.Vx.type[:,end]   .= :Neumann
    @info "Vx Node types"
    Print_xy(numbering.Vx.type) 

    numbering.Vy      = NumberingPoisson{3}()
    numbering.Vy.type = fill(:out, (nc.x+4, nc.y+3))
    numbering.Vy.type[2:end-1,2:end-2] .= :in
    numbering.Vy.type[1,:]     .= :periodic # make periodic
    numbering.Vy.type[end,:]   .= :periodic 
    numbering.Vy.type[:,1]     .= :Dirichlet
    numbering.Vy.type[:,end]   .= :Neumann
    @info "Vy Node types"
    Print_xy(numbering.Vy.type) 

    numbering.Pt      = NumberingPoisson{3}()
    numbering.Pt.type = fill(:out, (nc.x+2, nc.y+2))
    numbering.Pt.type[2:end-1,2:end-2] .= :in
    numbering.Pt.type[1,:]     .= :periodic # make periodic
    numbering.Pt.type[end,:]   .= :periodic 
    numbering.Pt.type[:,1]     .= :Dirichlet
    numbering.Pt.type[:,end]   .= :Neumann
    @info "Pt Node types"
    Print_xy(numbering.Pt.type) 

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