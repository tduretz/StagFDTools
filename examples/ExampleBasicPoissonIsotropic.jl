using StagFDTools, ExtendableSparse, StaticArrays, LinearAlgebra, UnPack, Plots

@views function AssemblyBasicPoisson(num, nc, Δ) 
    Kloc   = zeros(3,3)
    ndof   = maximum(num.num)
    K      = ExtendableSparseMatrix(ndof, ndof)
    shift  = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Local = num.num[i-1:i+1,j-1:j+1] .* num.pattern
        W_E = [-1; 2; -1]
        if i==2
            W_E = [0; 3; -1]
        elseif i==nc.x+1
            W_E = [-1; 3; 0]
        end
        S_N = [-1; 2; -1]
        if j==2
            S_N = [0; 3; -1]
        elseif j==nc.y+1
            S_N = [-1; 3; 0]
        end

        Kloc .= 0
        Kloc[:,2]  .= W_E./Δ.x^2
        Kloc[2,:] .+= S_N./Δ.y^2

        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) 
                K[num.num[i,j], Local[ii,jj]] = Kloc[ii,jj] 
            end
        end
    end
    return K
end

let
    # Generates an empty numbering structure
    numbering = NumberingPoisson{3}()
    
    # Resolution in FD cells
    nc = (x = 30, y = 40)

    ranges = RangesPoisson(nc)
    @unpack inx, iny = ranges
    
    # Define node types and set BC flags
    numbering.type = fill(:out, (nc.x+2, nc.y+2))
    numbering.type[inx,iny] .= :in
    numbering.type[1,:]     .= :Dirichlet 
    numbering.type[end,:]   .= :Dirichlet 
    numbering.type[:,1]     .= :Dirichlet
    numbering.type[:,end]   .= :Dirichlet
    
    @info "Node types"
    printxy(numbering.type) 

    # 5-point stencil
    numbering.pattern = @SMatrix([0 1 0; 1 1 1; 0 1 0]) 
    NumberingPoisson!(numbering, nc)
    # Parameters
    L     = 1.
    u_dir = 1.
    # Arrays
    r   = zeros(nc.x+2, nc.y+2)
    s   = zeros(nc.x+2, nc.y+2)
    u   = zeros(nc.x+2, nc.y+2)
    Δ   = (x=L/nc.x, y=L/nc.y)
    xc  = LinRange(-L/2-Δ.x/2, L/2+Δ.x/2, nc.x+2)
    yc  = LinRange(-L/2-Δ.y/2, L/2+Δ.y/2, nc.y+2)
    s  .= 50*exp.(-(xc.^2 .+ (yc').^2)./0.4^2)
    # Residual check
    u[[1,end],iny] .= 2*u_dir .- u[[2,end-1],iny]
    u[inx,[1,end]] .= 2*u_dir .- u[inx,[2,end-1]]
    r[inx,iny]     .= (u[1:end-2,iny] .+ u[3:end-0,iny] - 2*u[inx,iny])/Δ.x^2 + (u[inx,1:end-2] .+ u[inx,3:end-0] - 2*u[inx,iny])/Δ.y^2 .+ s[inx,iny]
    @info norm(r)/sqrt(length(r))
    # Assembly
    K  = AssemblyBasicPoisson(numbering, nc, Δ)
    @show norm(K-K')
    b  = r[inx,iny][:]
    # Solve
    du           = K\b
    u[inx,iny] .+= reshape(du, nc...)
    # Residual check
    u[[1,end],iny] .= 2*u_dir .- u[[2,end-1],iny]
    u[inx,[1,end]] .= 2*u_dir .- u[inx,[2,end-1]]
    r[inx,iny]     .= (u[1:end-2,iny] .+ u[3:end-0,iny] - 2*u[inx,iny])/Δ.x^2 + (u[inx,1:end-2] .+ u[inx,3:end-0] - 2*u[inx,iny])/Δ.y^2 .+ s[inx,iny]
    @info norm(r)/sqrt(length(r))
    # Visualization
    heatmap(xc[inx], yc[iny], u[inx,iny]')
end