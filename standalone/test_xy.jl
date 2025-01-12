let  
    #--------------------------------------------#
    # Resolution
    nc = (x = 10, y = 10)

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_Pt, iny_Pt, size_x, size_y, size_p = Ranges_Stokes(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = BoundaryConditions(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
        fill(:out, (nc.x+1, nc.y+1)),
    )
    BC = BoundaryConditions(
        fill(0., (nc.x+3, nc.y+4)),
        fill(0., (nc.x+4, nc.y+3)),
        fill(0., (nc.x+2, nc.y+2)),
        fill(0., (nc.x+1, nc.y+1)),
    )

    type.xy                  .= :τxy 
    type.xy[2:end-1,2:end-1] .= :in 


    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx] .= :in       
    type.Vx[2,iny_Vx]       .= :constant 
    type.Vx[end-1,iny_Vx]   .= :constant 
    type.Vx[inx_Vx,2]       .= :Neumann
    type.Vx[inx_Vx,end-1]   .= :Neumann
    # type.Vx[:,2]       .= :Neumann
    # type.Vx[:,end-1]   .= :Neumann
    type.Vx[inx_Vx,2]       .= :Dirichlet
    type.Vx[inx_Vx,end-1]   .= :Dirichlet
    BC.Vx[2,iny_Vx]         .= 0.0
    BC.Vx[end-1,iny_Vx]     .= 0.0
    BC.Vx[inx_Vx,2]         .= 0.0
    BC.Vx[inx_Vx,end-1]     .= 0.0
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy] .= :in       
    type.Vy[2,iny_Vy]       .= :Neumann
    type.Vy[end-1,iny_Vy]   .= :Neumann
    type.Vy[inx_Vy,2]       .= :constant 
    type.Vy[inx_Vy,end-1]   .= :constant 
    BC.Vy[2,iny_Vy]         .= 0.0
    BC.Vy[end-1,iny_Vy]     .= 0.0
    BC.Vy[inx_Vy,2]         .= 0.0
    BC.Vy[inx_Vy,end-1]     .= 0.0
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in

    #--------------------------------------------#
    # Intialise field
    L  = (x=1.0, y=1.0)
    Δ  = (x=L.x/nc.x, y=L.y/nc.y)
    R  = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_p...))
    V  = (x=zeros(size_x...), y=zeros(size_y...))
    η  = (x= ones(size_x...), y= ones(size_y...), p=ones(size_p...) )
    Rp = zeros(size_p...)
    Pt = zeros(size_p...)
    xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)
    xvy = LinRange(-L.x/2-3Δ.x/2, L.x/2+3Δ.x/2, nc.x+4)
    yvy = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    yvx = LinRange(-L.y/2-3Δ.y/2, L.y/2+3Δ.y/2, nc.y+4)

    # Initial configuration
    ε̇  = -1.0
    V.x[inx_Vx,iny_Vx] .=  ε̇*xv .+ 0*yc' 
    V.y[inx_Vy,iny_Vy] .= 0*xc .-  ε̇*yv'
    
    ymin, ymax = -L.y/2, L.y/2
    V.x[inx_Vx,iny_Vx]      .= 0*ε̇*xv .+ ε̇*yc' 

    ε̇xy     = zeros(nc.x+1, nc.y+1)
    ε̇x̄ȳ_loc = zeros(2,2)
    ε̇xy_loc = zeros(3,3)
    Vx_loc  = zeros(3,4)
    Vy_loc  = zeros(4,3)
    bcx_loc = fill(:out, (3,4))
    bcy_loc = fill(:out, (4,3))

    for i in 1:nc.x+1, j in 1:nc.y+1

        bcx_loc .= type.Vx[i:i+2,j:j+3]
        bcy_loc .= type.Vy[i:i+3,j:j+2]
        Vx_loc  .= V.x[i:i+2,j:j+3]
        Vy_loc  .= V.y[i:i+3,j:j+2]

        # if i==nc.x+1
        #     printxy(bcy_loc)
        # end

        # ########################
        for jj=1:3
            if bcy_loc[1,jj] == :Neumann 
                Vy_loc[1,jj] = Vy_loc[2,jj] 
            end
            if bcy_loc[1,jj] == :out
                Vy_loc[2,jj] = Vy_loc[3,jj] #- Δ.y*ε̇
                Vy_loc[1,jj] = Vy_loc[4,jj] # simplification
            end
            if bcy_loc[4,jj] == :Neumann 
                Vy_loc[4,jj] = Vy_loc[3,jj] 
            end
            if bcy_loc[4,jj] == :out
                Vy_loc[3,jj] = Vy_loc[2,jj] #- Δ.y*ε̇
                Vy_loc[4,jj] = Vy_loc[1,jj] # simplification
            end
        end
    
        # ########################


        for ii=1:3

            if bcx_loc[ii,4] == :Neumann 
                Vx_loc[ii,4] = Vx_loc[ii,3] + Δ.y*ε̇
            elseif bcx_loc[ii,4] == :Dirichlet 
                Vx_loc[ii,4] = -Vx_loc[ii,3] + 2*ε̇*ymax
            end
            if bcx_loc[ii,4] == :out
                if bcx_loc[ii,3] == :Neumann
                    Vx_loc[ii,3] =  Vx_loc[ii,2] + Δ.y*ε̇
                    Vx_loc[ii,4] =  Vx_loc[ii,1] + 3*Δ.y*ε̇ # simplification 
                elseif bcx_loc[ii,3] == :Dirichlet
                    Vx_loc[ii,3] = -Vx_loc[ii,2] + 2*ε̇*ymax 
                    Vx_loc[ii,4] = -Vx_loc[ii,1] + 2*ε̇*ymax # simplification 
                end
            end

            if bcx_loc[ii,1] == :Neumann 
                Vx_loc[ii,1] =  Vx_loc[ii,2] - Δ.y*ε̇
            elseif bcx_loc[ii,1] == :Dirichlet 
                Vx_loc[ii,1] = -Vx_loc[ii,2] + 2*ε̇*ymin
            end
            if bcx_loc[ii,1] == :out 
                if bcx_loc[ii,2] == :Neumann
                    Vx_loc[ii,2] =  Vx_loc[ii,3] - Δ.y*ε̇
                    Vx_loc[ii,1] =  Vx_loc[ii,4] - 3*Δ.y*ε̇ # simplification
                elseif bcx_loc[ii,2] == :Dirichlet
                    Vx_loc[ii,2] = -Vx_loc[ii,3] + 2*ε̇*ymin
                    Vx_loc[ii,1] = -Vx_loc[ii,4] + 2*ε̇*ymin # simplification
                end

            end

            for jj=1:4
                if bcx_loc[1,jj] == :out
                    Vx_loc[1,jj] = Vx_loc[2,jj]
                end
                if bcx_loc[3,jj] == :out
                    Vx_loc[3,jj] = Vx_loc[2,jj]
                end
            end
    
         
        end

        # if j == nc.y+1
        # @show Vx_loc1
        # end

        ε̇xy_loc .= 1/2* ( diff(Vx_loc, dims=2)/Δ.y + diff(Vy_loc, dims=1)/Δ.x ) 

        ε̇x̄ȳ_loc .= 1/4*(ε̇xy_loc[1:end-1,1:end-1] + ε̇xy_loc[2:end-0,1:end-1] + ε̇xy_loc[1:end-1,2:end-0] + ε̇xy_loc[2:end-0,2:end-0])
        ε̇xy[i,j] = 1/4*(ε̇x̄ȳ_loc[1:end-1,1:end-1] + ε̇x̄ȳ_loc[2:end-0,1:end-1] + ε̇x̄ȳ_loc[1:end-1,2:end-0] + ε̇x̄ȳ_loc[2:end-0,2:end-0])[1]

    end
     
    #--------------------------------------------#
    p1 = heatmap(xv, yv, ε̇xy', aspect_ratio=1, xlim=extrema(xc))
    display(plot(p1))

    printxy(type.Vx)

    printxy(ε̇xy)
    
end