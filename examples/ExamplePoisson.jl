using StagFDTools, ExtendableSparse, StaticArrays

let

    physics = Physics()
    physics.Poisson = true

    numbering = NumberingPoisson{3}()
    
    # Resolution
    nc = (x = 10, y = 9)
    
    # Define node types and set BC flags
    numbering.type = fill(:out, (nc.x+2, nc.y+2))
    numbering.type[2:end-1,2:end-1] .= :in
    numbering.type[1,:]     .= :periodic # make periodic
    numbering.type[end,:]   .= :periodic 
    numbering.type[:,1]     .= :Dirichlet
    numbering.type[:,end]   .= :Neumann
    @info "Node types"
    Print_xy(numbering.type) 

    if physics.Poisson
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
end