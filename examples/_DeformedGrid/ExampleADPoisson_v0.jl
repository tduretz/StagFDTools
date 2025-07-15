using StagFDTools, StagFDTools.Poisson, ExtendableSparse, StaticArrays, LinearAlgebra, Statistics, UnPack
using TimerOutputs
# using Enzyme
using ForwardDiff, Enzyme  # AD backends you want to use 
import CairoMakie as cm
import Makie.GeometryBasics as geom

###############################################################
########################## NEW STUFF ##########################
###############################################################

function TransformCoordinates(ξ, params)
    h = params.Amp*exp(-(ξ[1] - params.x0)^2 / params.σx^2)
    return @SVector([ξ[1], (ξ[2]/params.ymin0)*(params.m-h)+h])
end

###############################################################
########################## NEW STUFF ##########################
###############################################################

# This function computes the residuals of the 2D Poisson equation using a 5-point stencil
# qx = -k*du_dx
# qy = -k*du_dy
# dqx_dx + dqy_dy = s
# where qx and qy are components of a flux vector (q) in 2D and s is a source term
# The 5-point stencil looks like this:
#      |             |
#      |     uN      |
#      |             |
#----- x --- qyN --- x -----
#      |             |
# uW  qxW    uC     qxE  uE
#      |             |
#----- x --- qyS --- x -----
#      |             |
#      |     uS      |
#      |             |
# dx[1]| -- dx[2] -- | dx[3]

# vectuerus dx et dy
function Poisson2D(u_loc, k, s, type_loc, bcv_loc, Δ)

    # u_loc is 3*3 matrix containing the current values of u for the whole stencil
    #             0   uN  0
    #     u_loc = uW  uC  uE
    #             0   uS  0
    # Therefore u_loc[2,2] is the current value of uC
    uC       = u_loc[2,2]

    # Boundary conditions need to be applied on every boundaries :D
    # Here we define the values of the ghost nodes. For example at the west side uW needs to be defined  
    # For example, to set a Dirichlet values, we say: 1/2*(uW + uC) = u_BC, hence uW = 2*u_BC - uC
    # West
    if type_loc[1,2] === :Dirichlet           #          uBC = 1/2*(uW + uC) # grille constant  # Variable Grid        
        uW = 2*bcv_loc[1,2] - u_loc[2,2]      #   uW ---- | ---- uC ---- |
    elseif type_loc[1,2] === :Neumann
        uW = Δ.x*bcv_loc[1,2] + u_loc[2,2]    #   qBC = - k(uW - uC) / dx  --> uW = dx*qBC + uC   # Variable Grid 
    elseif type_loc[1,2] === :periodic || type_loc[1,2] === :in
        uW = u_loc[1,2]                       #   rien 
    end

    # East
    if type_loc[3,2] === :Dirichlet
        uE = 2*bcv_loc[3,2] - u_loc[2,2]
    elseif type_loc[3,2] === :Neumann
        uE = -Δ.x*bcv_loc[3,2] + u_loc[2,2]
    elseif type_loc[3,2] === :periodic || type_loc[3,2] === :in
        uE = u_loc[3,2] 
    end

    # South
    if type_loc[2,1] === :Dirichlet
        uS = 2*bcv_loc[2,1] - u_loc[2,2]
    elseif type_loc[2,1] === :Neumann
        uS = Δ.y*bcv_loc[2,1] + u_loc[2,2]
    elseif type_loc[2,1] === :periodic || type_loc[2,1] === :in
        uS = u_loc[2,1] 
    end

    # North
    if type_loc[2,3] === :Dirichlet
        uN = 2*bcv_loc[2,3] - u_loc[2,2]
    elseif type_loc[2,3] === :Neumann
        uN = -Δ.y*bcv_loc[2,3] + u_loc[2,2]
    elseif type_loc[2,3] === :periodic || type_loc[2,3] === :in
        uN = u_loc[2,3] 
    end

    # Heat flux for each face based on finite differences
    qxW = -k.xx[1]*(uC - uW)/Δ.x  # West  # Achtung: dx ---> 1/2(dx[1]+dx[2])
    qxE = -k.xx[2]*(uE - uC)/Δ.x  # East
    qyS = -k.yy[1]*(uC - uS)/Δ.y  # South
    qyN = -k.yy[2]*(uN - uC)/Δ.y  # North

    # Return the residual function based on finite differences
    return -(-(qxE - qxW)/Δ.x - (qyN - qyS)/Δ.y + s)  # Achtung: dx ---> dx[2]
end

# This function loop over the whole set of 2D cells and computes the residual in each cells
function ResidualPoisson2D!(R, u, k, s, num, type, bc_val, nc, Δ)  # u_loc, s, type_loc, Δ

    # This is just a vector of zeros (off-diagonal terms of the conductivity tensor are 0 in the isotropic case)
    # Here StaticArrays are being used a bit everywhere. This is a Julia Library which is used for achieving good performance.
    # Have a look there: https://github.com/JuliaArrays/StaticArrays.jl
    # Basically all small arrays should be contained within such objects (e.g. heat flux vector, matrix containing the stencil values)
    k_loc_shear = @SVector(zeros(2))
                
    shift    = (x=1, y=1)
    
    # Loop over the cells in 2D
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x

        # vecteur dx
        # vecteur dy

        # This is the 3*3 matrix that contains the current stencil values (see residual function above) 
        u_loc     =  SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        # Conductivity is also stored into Static arrays - can be simplified
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]] # W and E values
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]] # S and N values
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)

        # This stores the VALUE of the boundary condition for the stencils (3*3 matrix)
        bcv_loc   = SMatrix{3,3}(bc_val.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        # This stores the TYPE of the boundary condition for the stencils (3*3 matrix)
        type_loc  = SMatrix{3,3}(  type.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        # This calls the residual function
        # ---> vecteurs dx et dy
        R[i,j]    = Poisson2D(u_loc, k_loc, s[i,j], type_loc, bcv_loc, Δ)  
    end
    return nothing
end

# The 2 functions below do the same thing. They compute the partial derivatives of the residual function and puts the coefficient in the system matrix.
# One function does it with ForwardDiff: https://github.com/JuliaDiff/ForwardDiff.jl
# The other does it with Enzyme: https://github.com/EnzymeAD/Enzyme.jl

# This function you can skip since we're going to use the second option  :D
function AssemblyPoisson_ForwardDiff!(K, u, k, s, number, type, pattern, bc_val, nc, Δ)

    shift    = (x=1, y=1)

    k_loc_shear = @SVector(zeros(2))

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        
        num_loc   = SMatrix{3,3}(number.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern.u.u
        u_loc     = SMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)
        bcv_loc   = SMatrix{3,3}(bc_val.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc  = SMatrix{3,3}(  type.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
 
        ∂R∂u = ForwardDiff.gradient(
            x -> Poisson2D(x, k_loc, s[i,j], type_loc, bcv_loc, Δ), 
            u_loc
        )

        num_ij = number.u[i,j]
        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if num_loc[ii,jj] > 0
                K.u.u[num_ij, num_loc[ii,jj]] = ∂R∂u[ii,jj] 
            end
        end
    end
    return nothing
end

# Computation of the partial derivatives of the residual function. Sets the coefficient in the system matrix.
function AssemblyPoisson_Enzyme!(K, u, k, s, number, type, pattern, bc_val, nc, Δ)

    # This is a local matrix that stores the partial derivatives.
     
    # Remember:     #             0    uN    0
                    #     u_loc = uW   uC    uE
                    #             0    uS    0 
                    
    # Now we have   #                0     df_duN      0 
                    #     ∂R∂u  = df_duW   df_duC   df_duE
                    #                0     df_duS      0 
    # These cofficient are automatically computed by Enzyme and are then collected in the sparse matrix that defines the linear system of equations
    ∂R∂u     = @MMatrix zeros(3,3) 
    shift    = (x=1, y=1)

    # Same as above 
    k_loc_shear = @SVector(zeros(2))

    # Loop over the cells in 2D
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x

        # This is a 3*3 matrix that contains the number of each equation in the stencil (each cell has one corresponding value of u, which has a unique equation number)
        num_loc   = SMatrix{3,3}(number.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern.u.u

        # This is the 3*3 matrix that contains the current stencil values (see residual function above) 
        u_loc     = MMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        # Same as above: conductivity tensor
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]]
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)

        # Same as above: boundary condition VALUES
        bcv_loc   = SMatrix{3,3}(bc_val.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        # Same as above: boundary condition TYPES
        type_loc  = SMatrix{3,3}(  type.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        # Initialise partial derivatives to 0
        ∂R∂u     .= 0e0

        # Here the magic happens: we call a function from Enzyme that computes all, partial derivatives for the current stencil block 
        autodiff(Enzyme.Reverse, Poisson2D, Duplicated(u_loc, ∂R∂u), Const(k_loc), Const(s[i,j]), Const(type_loc), Const(bcv_loc), Const(Δ))

        # This loops through the 2*2 stencil block and sets the coefficient ∂R∂u into the sparse matrix K.u.u
        num_ij = number.u[i,j]
        for jj in axes(num_loc,2), ii in axes(num_loc,1)
            if num_loc[ii,jj] > 0 # only if equation exists, then put a value
                # For the given equation number (num_ij) enter all values on the line of the sparse matrix K.u.u
                K.u.u[num_ij, num_loc[ii,jj]] = ∂R∂u[ii,jj]  
            end
        end
    end
    return nothing
end

let
    # This is the main code that calls the above functions!
    to = TimerOutput()

    # Resolution in FD cells
    nc = (x = 30, y = 40)

    # Get ranges
    ranges = Ranges(nc)
    (; inx, iny) = ranges

    # Define node types 
    type = Fields( fill(:out, (nc.x+2, nc.y+2)) )   # Achtung: geist nodes
    type.u[2:end-1,2:end-1].= :in                   # inside nodes are all type :in
    type.u[1,:]            .= :Dirichlet            # one BC type is :Dirichlet # West
    type.u[end,:]          .= :Dirichlet            # East
    type.u[:,1]            .= :Dirichlet            # South
    type.u[:,end]          .= :Dirichlet            # North

    # Define values of the boundary conditions
    bc_val = Fields( fill(0., (nc.x+2, nc.y+2)) )   # Achtung: geist nodes
    bc_val.u        .= zeros(nc.x+2, nc.y+2)        # useless ?!
    bc_val.u[1,:]   .= 1.0                          # Boundary value is 1.0 and this will be a Dirichlet (u at West = 1) # West
    bc_val.u[end,:] .= 1.0                          # East
    bc_val.u[:,1]   .= 1.0                          # South
    bc_val.u[:,end] .= 1.0                          # North

    # 5-point stencil, this is the definition of the stencil block. 
    # It basically states which points are being included in the stencil
    pattern = Fields( Fields( @SMatrix([0 1 0; 
                                        1 1 1; 
                                        0 1 0]) ) )

    # Equation number: a 2D table containing the equation numbers 
    number = Fields( fill(0, (nc.x+2, nc.y+2)) )  # Achtung: geist nodes  
    Numbering!(number, type, nc)                  # There is a StagFD function that does it
    
    # Parameters
    L     = 1.                   # Domain extent
    # Arrays
    # div( q ) - s = 0
    # q = - k * grad(u) --> k is a tensor               k = kxx kxy            grad(u) = dudx
    #                                                       kyx kyy                      dudy
    # q = -( kxx*dudx + kxy*dudy )
    #     -( kyx*dudx + kyy*dudy )

    #      |                  |
    #      |        uN        |
    #      |                  |
    #----- x -- kyyN, kyxN -- x -----
    #      |                  |
    # uW  kxxW, kxyW  uC kxxE, kxyE  uE
    #      |                  |
    #----- x -- kyyS, kyxS -- x -----
    #      |                  |
    #      |        uS        |
    #      |                  |
    
    r   = zeros(nc.x+2, nc.y+2)  # residual of the equation (right-hand side)
    s   = zeros(nc.x+2, nc.y+2)  # forcing term
    u   = zeros(nc.x+2, nc.y+2)  # solution
    k   = (x = (xx= ones(nc.x+1,nc.y), xy=zeros(nc.x+1,nc.y)), # conductivity tensor (can be simplidied)
           y = (yx=zeros(nc.x,nc.y+1), yy= ones(nc.x,nc.y+1)))
    
    ###############################################################
    ########################## NEW STUFF ##########################
    ###############################################################

    # Reference domain
    x = (min=-L/2, max=L/2)
    y = (min=  -L, max=0.0)
    Δ = (x=L/nc.x, y=L/nc.y)
    xv = LinRange(x.min-Δ.x,   x.max+Δ.x,   nc.x+3)
    yv = LinRange(y.min-Δ.y,   y.max+Δ.y,   nc.y+3)
    xc = LinRange(x.min-Δ.x/2, x.max+Δ.x/2, nc.x+2)
    yc = LinRange(y.min-Δ.y/2, y.max+Δ.y/2, nc.y+2)

    # Reference coordinates ξ
    ξ = (
        v =  [@MVector(zeros(2)) for _ in axes(xv,1), _ in axes(yv,1)],
        c =  [@MVector(zeros(2)) for _ in axes(xc,1), _ in axes(yc,1)],
    )
    for I in CartesianIndices(ξ.v)
        i, j = I[1], I[2]
        ξ.v[I] .= @SVector([xv[i], yv[j]]) 
    end
    for I in CartesianIndices(ξ.c)
        i, j = I[1], I[2]
        ξ.c[I] .= @SVector([xc[i], yc[j]]) 
    end

    # Physical coordinates X 
    X = (
        v =  [@MVector(zeros(2)) for _ in axes(xv,1), _ in axes(yv,1)],
        c =  [@MVector(zeros(2)) for _ in axes(xc,1), _ in axes(yc,1)],
    )
    for I in CartesianIndices(ξ.v)
        i, j = I[1], I[2]
        X.v[I] .= @SVector([xv[i], yv[j]]) 
    end
    for I in CartesianIndices(ξ.c)
        i, j = I[1], I[2]
        X.c[I] .= @SVector([xc[i], yc[j]]) 
    end

    # Deform mesh along y
    params = (
        m      = -1,
        Amp    = 0.25,
        σx     = 0.1,
        ymin0  = -1,
        ymax0  = 0.5,
        y0     = 0.5,
        x0     = 0.0,
    )
    for I in CartesianIndices(X.v)
        X.v[I] .=  TransformCoordinates(ξ.v[I], params)
    end
    for I in CartesianIndices(X.c)
        X.c[I] .= TransformCoordinates(ξ.v[I], params)
    end

    # Model configuration: source term
    for I in CartesianIndices(ξ.c)
        x = X.c[I][1]
        y = X.c[I][2]
        s[I]  = 50*exp(-(x^2 + (y + L/2)^2)/0.4^2)
    end

    ###############################################################
    ########################## NEW STUFF ##########################
    ###############################################################

    # Residual check: div( q ) - s = r
    @timeit to "Residual" ResidualPoisson2D!(r, u, k, s, number, type, bc_val, nc, Δ) 
    @info norm(r)/sqrt(length(r))
    
    # Sparse matrix assembly
    nu  = maximum(number.u)
    M   = Fields( Fields( ExtendableSparseMatrix(nu, nu) )) 
  
    @timeit to "Assembly Enzyme" begin
        AssemblyPoisson_Enzyme!(M, u, k, s, number, type, pattern, bc_val, nc, Δ)
    end
    @timeit to "Assembly ForwardDiff" begin
        AssemblyPoisson_ForwardDiff!(M, u, k, s, number, type, pattern, bc_val, nc, Δ)
    end

    @info "Symmetry"
    @show norm(M.u.u - M.u.u')
    # A one-step Newton iteration - the problem is linear: only one step is needed to reach maximum accurracy
    b  = r[inx,iny][:]                  # creates a 1D rhight hand side vector (whitout ghosts), values are the current residual
    # Solve
    du           = .-M.u.u\b              # apply inverse of matrix M.u.u to residual vector 
    u[inx,iny] .+= reshape(du, nc...)   # update the solution u using the correction du
    # Residual check
    ResidualPoisson2D!(r, u, k, s, number, type, bc_val, nc, Δ)     
    @info norm(r)/sqrt(length(r))
    
    # Visualization

    ###############################################################
    ########################## NEW STUFF ##########################
    ###############################################################

    # Post-process
    cells = (
        x = zeros((nc.x+2)*(nc.y+2), 4),
        y = zeros((nc.x+2)*(nc.y+2), 4)
    )
    for I in CartesianIndices(X.c)
        i, j = I[1], I[2]
        c = i + (j-1)*(nc.x+2)
        cells.x[c,:] .= @SVector([X.v[i,j][1], X.v[i+1,j][1], X.v[i+1,j+1][1], X.v[i,j+1][1] ]) 
        cells.y[c,:] .= @SVector([X.v[i,j][2], X.v[i+1,j][2], X.v[i+1,j+1][2], X.v[i,j+1][2] ]) 
    end

    ###############################################################
    ########################## NEW STUFF ##########################
    ###############################################################

    pc = [cm.Polygon( geom.Point2f[ (cells.x[i,j], cells.y[i,j]) for j=1:4] ) for i in 1:(nc.x+2)*(nc.y+2)]
    # Visu
    res = 800
    fig = cm.Figure(size = (res, res), fontsize=25)
    # ----
    ax  = cm.Axis(fig[1, 1], title = "u - centroids", xlabel = "x", ylabel = "y", aspect=1.0)
    cm.poly!(ax, pc, color = u[:], colormap = :turbo, strokewidth = 0, strokecolormap = :white, colorrange=extrema(u[2:end-1,2:end-1]))#, colorrange=limits
    cm.Colorbar(fig[1, 2], colormap = :turbo, flipaxis = true, size = 10 )    
    display(fig)
end


