using StagFDTools, ExtendableSparse, StaticArrays

let

    physics = Physics()
    physics.Poisson = true

    Numbering = NumberingPoisson()
    
    # Resolution
    nc = (x = 10, y = 9)
    
    # 5-point stencil
    Numbering.Type = fill(:out, (nc.x+2, nc.y+2))
    Numbering.Type[2:end-1,2:end-1] .= :in
    Numbering.Type[1,:]     .= :periodic # make periodic
    Numbering.Type[end,:]   .= :periodic 
    Numbering.Type[:,1]     .= :Dirichlet
    Numbering.Type[:,end]   .= :Neumann
    @info "Node types"
    Print_xy(Numbering.Type) 

    if physics.Poisson
        # 5-point stencil
        Numbering.Pattern = @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
        NumberingPoisson!(Numbering, nc)
        @time K    = SparsityPatternPoisson(Numbering, nc)
        @time K_SA = SparsityPatternPoisson_SA(Numbering.Num, Numbering.Pattern, nc)
        @assert K == K_SA
        @info "5-point stencil"
        display(K)
        display(K-K')

        # 9-point stencil
        Pattern = @SMatrix([1 1 1; 1 1 1; 1 1 1]) 
        NumberingPoisson!(Numbering, nc)
        K    = SparsityPatternPoisson(Numbering, nc)
        K_SA = SparsityPatternPoisson_SA(Numbering.Num, Numbering.Pattern, nc)
        @assert K == K_SA
        @info "9-point stencil"
        display(K)
        display(K-K')               
    end
end