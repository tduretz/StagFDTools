using StagFDTools, StagFDTools.Poisson, ExtendableSparse, StaticArrays, LinearAlgebra, Statistics, UnPack
using TimerOutputs
using ForwardDiff
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian
import CairoMakie as cm
import CairoMakie.Makie.GeometryBasics as geom

###############################################################
########################## NEW STUFF ##########################
###############################################################

function TransformCoordinates(ξ, params)
    h = params.Amp*exp(-(ξ[1] - params.x0)^2 / params.σx^2)
    # return @SVector([ξ[1], (ξ[2]/params.ymin0)*(params.m-h)+h])
    return @SVector([ξ[1], ξ[2]])

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
function Poisson2D(u_loc, k, s, Jc, Jv, type_loc, bcv_loc, Δ)

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

    # BC later
    uSW = u_loc[1,1]
    uSE = u_loc[3,1]
    uNW = u_loc[1,3]
    uNE = u_loc[3,3]

    # # We need to interpolate u values to vertices using the 9 points
    # u_v_SW = 0.25*(uC + uSW + uS + uW) 
    # u_v_SE = 0.25*(uC + uSE + uS + uE) 
    # u_v_NW = 0.25*(uC + uNW + uN + uW) 
    # u_v_NE = 0.25*(uC + uNE + uN + uE)

    # # This could all be precomputed 
    # ∂ξ∂x_W = 0.5* (Jv[1,1][1,1] + Jv[1,2][1,1])
    # ∂ξ∂y_W = 0.5* (Jv[1,1][1,2] + Jv[1,2][1,2])
    # ∂ξ∂x_E = 0.5* (Jv[2,1][1,1] + Jv[2,2][1,1])
    # ∂ξ∂y_E = 0.5* (Jv[2,1][1,2] + Jv[2,2][1,2])
    # ∂η∂y_S = 0.5* (Jv[1,1][2,2] + Jv[2,1][2,2])
    # ∂η∂x_S = 0.5* (Jv[1,1][1,2] + Jv[2,1][1,2])
    # ∂η∂y_N = 0.5* (Jv[1,2][2,2] + Jv[2,2][2,2])
    # ∂η∂x_N = 0.5* (Jv[1,2][1,2] + Jv[2,2][1,2])

    # # Maybe this can be made nicer?
    # ∂u∂x_W = (uC - uW)/Δ.ξ * ∂ξ∂x_W + (u_v_NW - u_v_SW)/Δ.η * ∂ξ∂y_W
    # ∂u∂x_E = (uE - uC)/Δ.ξ * ∂ξ∂x_E + (u_v_NE - u_v_SE)/Δ.η * ∂ξ∂y_E
    # ∂u∂y_S = (uC - uS)/Δ.η * ∂η∂y_S + (u_v_SE - u_v_SW)/Δ.ξ * ∂η∂x_S
    # ∂u∂y_N = (uN - uC)/Δ.η * ∂η∂y_N + (u_v_NE - u_v_NW)/Δ.ξ * ∂η∂x_N

    # # Heat flux for each face based on finite differences
    # qxW = -k.xx[1]*∂u∂x_W  
    # qxE = -k.xx[2]*∂u∂x_E  
    # qyS = -k.yy[1]*∂u∂y_S
    # qyN = -k.yy[2]*∂u∂y_N

    # qxSW = 0.
    # qxSE = 0.
    # qxNW = 0.
    # qxNE = 0.
    # qySW = 0.
    # qySE = 0.
    # qyNW = 0.
    # qyNE = 0.

    # # We need to interpolate the horizontal flux component to y points
    # qxS = 0.25*(qxW + qxE + qxSW + qxSE)
    # qxN = 0.25*(qxW + qxE + qxNW + qxNE)

    # # We need to interpolate the vertical flux component to x points
    # qyW = 0.25*(qyS + qyN + qySW + qyNW)
    # qyE = 0.25*(qyS + qyN + qySE + qyNE)

    # # Return the residual function based on finite differences
    # ∂q∂x = (qxE - qxW)/Δ.ξ * Jc[1,1][1,1] + (qxN - qxS)/Δ.η * Jc[1,1][1,2]
    # ∂q∂y = (qyN - qyS)/Δ.η * Jc[1,1][2,2] + (qyE - qyW)/Δ.ξ * Jc[1,1][2,1]
    # f    = ∂q∂x + ∂q∂y - s

    ##############################################

    # Interpolate u at y locations
    u_y_SW = 0.5*(uW + uSW)
    u_y_S  = 0.5*(uC + uS)
    u_y_SE = 0.5*(uE + uSE)
    u_y_NW = 0.5*(uW + uNW)
    u_y_N  = 0.5*(uC + uN)
    u_y_NE = 0.5*(uE + uNE)

    # Interpolate u at x locations
    u_x_SW = 0.5*(uS + uSW)
    u_x_W  = 0.5*(uC + uW)
    u_x_NW = 0.5*(uN + uNW)
    u_x_SE = 0.5*(uS + uSE)
    u_x_E  = 0.5*(uC + uE)
    u_x_NE = 0.5*(uN + uNE)

    @show Jc[1,1][2,2]
    @show Jc[1,1][2,2]
    @show Jc[1,1][2,2]
    @show Jc[1,1][2,2]


    # x flux component at each vertices
    qxSW = -k.xx[1] * ( (u_y_S - u_y_SW)/Δ.ξ * Jv[1,1][1,1] + 0*(u_x_W - u_x_SW)/Δ.η * Jv[1,1][1,2])
    qxSE = -k.xx[1] * ( (u_y_SE - u_y_S)/Δ.ξ * Jv[2,1][1,1] + 0*(u_x_E - u_x_SE)/Δ.η * Jv[2,1][1,2])
    qxNW = -k.xx[1] * ( (u_y_N - u_y_NW)/Δ.ξ * Jv[1,2][1,1] + 0*(u_x_NW - u_x_W)/Δ.η * Jv[1,2][1,2])
    qxNE = -k.xx[1] * ( (u_y_NE - u_y_N)/Δ.ξ * Jv[2,2][1,1] + 0*(u_x_NE - u_x_E)/Δ.η * Jv[2,2][1,2])

    # y flux component at each vertices
    qySW = -k.yy[1] * ( 0*(u_y_S - u_y_SW)/Δ.ξ * Jv[1,1][2,1] + (u_x_W - u_x_SW)/Δ.η * Jv[1,1][2,2])
    qySE = -k.yy[1] * ( 0*(u_y_SE - u_y_S)/Δ.ξ * Jv[2,1][2,1] + (u_x_E - u_x_SE)/Δ.η * Jv[2,1][2,2])
    qyNW = -k.yy[1] * ( 0*(u_y_N - u_y_NW)/Δ.ξ * Jv[1,2][2,1] + (u_x_NW - u_x_W)/Δ.η * Jv[1,2][2,2])
    qyNE = -k.yy[1] * ( 0*(u_y_NE - u_y_N)/Δ.ξ * Jv[2,2][2,1] + (u_x_NE - u_x_E)/Δ.η * Jv[2,2][2,2])

    # Average flux components to x and y locations
    qxW  = 1/2*(qxSW + qxNW)
    qxE  = 1/2*(qxSE + qxNE) 
    qxS  = 1/2*(qxSW + qxSE)
    qxN  = 1/2*(qxNW + qxNE) 
    qyW  = 1/2*(qySW + qyNW)
    qyE  = 1/2*(qySE + qyNE) 
    qyS  = 1/2*(qySW + qySE)
    qyN  = 1/2*(qyNW + qyNE)

    # Return the residual function based on finite differences
    ∂qx∂x = (qxE - qxW)/Δ.ξ * Jc[1,1][1,1] + 0*(qxN - qxS)/Δ.η * Jc[1,1][1,2]
    ∂qy∂y = 0*(qyE - qyW)/Δ.ξ * Jc[1,1][2,1] + (qyN - qyS)/Δ.η * Jc[1,1][2,2]
    f    = ∂qx∂x + ∂qy∂y - s
    return f  
end

# This function loop over the whole set of 2D cells and computes the residual in each cells
function ResidualPoisson2D!(R, u, k, s, Jinv, num, type, bc_val, nc, Δ)  # u_loc, s, type_loc, Δ

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

        Jinv_c    = SMatrix{1,1}(  Jinv.c[ii,jj] for ii in i:i,   jj in j:j  )
        Jinv_v    = SMatrix{2,2}(  Jinv.v[ii,jj] for ii in i:i+1, jj in j:j+1)

        # This calls the residual function
        # ---> vecteurs dx et dy
        R[i,j]    = Poisson2D(u_loc, k_loc, s[i,j], Jinv_c, Jinv_v, type_loc, bcv_loc, Δ)  
    end
    return nothing
end

# The 2 functions below do the same thing. They compute the partial derivatives of the residual function and puts the coefficient in the system matrix.
# One function does it with ForwardDiff: https://github.com/JuliaDiff/ForwardDiff.jl
# The other does it with Enzyme: https://github.com/EnzymeAD/Enzyme.jl

# Computation of the partial derivatives of the residual function. Sets the coefficient in the system matrix.
function AssemblyPoisson_Enzyme!(K, u, k, s, Jinv, number, type, pattern, bc_val, nc, Δ)

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

        Jinv_c    = SMatrix{1,1}(  Jinv.c[ii,jj] for ii in i:i,   jj in j:j  )
        Jinv_v    = SMatrix{2,2}(  Jinv.v[ii,jj] for ii in i:i+1, jj in j:j+1)

        # Initialise partial derivatives to 0
        ∂R∂u     .= 0e0

        # Here the magic happens: we call a function from Enzyme that computes all, partial derivatives for the current stencil block 
        forwarddiff_gradients!(Poisson2D, Duplicated(u_loc, ∂R∂u), Const(k_loc), Const(s[i,j]), Const(Jinv_c), Const(Jinv_v), Const(type_loc), Const(bcv_loc), Const(Δ))

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
    pattern = Fields( Fields( @SMatrix([1 1 1; 
                                        1 1 1; 
                                        1 1 1]) ) )

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
    ξ = (min=-L/2, max=L/2)
    η = (min=  -L, max=0.0)
    Δ = (ξ=L/nc.x, η=L/nc.y)
    ξv = LinRange(ξ.min-Δ.ξ,   ξ.max+Δ.ξ,   nc.x+3)
    ηv = LinRange(η.min-Δ.η,   η.max+Δ.η,   nc.y+3)
    ξc = LinRange(ξ.min-Δ.ξ/2, ξ.max+Δ.ξ/2, nc.x+2)
    ηc = LinRange(η.min-Δ.η/2, η.max+Δ.η/2, nc.y+2)

    # Reference coordinates ξ
    ξ = (
        v =  [@MVector(zeros(2)) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c =  [@MVector(zeros(2)) for _ in axes(ξc,1), _ in axes(ηc,1)],
    )
    for I in CartesianIndices(ξ.v)
        i, j = I[1], I[2]
        ξ.v[I] .= @SVector([ξv[i], ηv[j]]) 
    end
    for I in CartesianIndices(ξ.c)
        i, j = I[1], I[2]
        ξ.c[I] .= @SVector([ξc[i], ηc[j]]) 
    end

    # Physical coordinates X 
    X = (
        v =  [@MVector(zeros(2)) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c =  [@MVector(zeros(2)) for _ in axes(ξc,1), _ in axes(ηc,1)],
    )

    # Mesh deformation parameters
    params = (
        m      = -1,
        Amp    = 0.25,
        σx     = 0.1,
        ymin0  = -1,
        ymax0  = 0.5,
        y0     = 0.5,
        x0     = 0.0,
    )
   
    # Deform mesh and determine the inverse Jacobian  
    Jinv = (
        v =  [@MMatrix(zeros(2,2)) for _ in axes(ξv,1), _ in axes(ηv,1)],
        c =  [@MMatrix(zeros(2,2)) for _ in axes(ξc,1), _ in axes(ηc,1)],
    )
    
    I2  = LinearAlgebra.I(2)

    for I in CartesianIndices(X.v)
        J          = forwarddiff_jacobian(TransformCoordinates, ξ.v[I], Const(params))
        Jinv.v[I] .= J.derivs[1] \ I2
        X.v[I]    .= J.val
    end

    for I in CartesianIndices(X.c)
        J          = forwarddiff_jacobian(TransformCoordinates, ξ.c[I], Const(params))
        Jinv.c[I] .= J.derivs[1] \ I2
        X.c[I]    .= J.val
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
    @timeit to "Residual" ResidualPoisson2D!(r, u, k, s, Jinv, number, type, bc_val, nc, Δ) 
    @info norm(r)/sqrt(length(r))
    
    # Sparse matrix assembly
    nu  = maximum(number.u)
    M   = Fields( Fields( ExtendableSparseMatrix(nu, nu) )) 
  
    @timeit to "Assembly
" begin
        AssemblyPoisson_Enzyme!(M, u, k, s, Jinv, number, type, pattern, bc_val, nc, Δ)
    end

    @info "Symmetry"
    @show norm(M.u.u - M.u.u')
    # A one-step Newton iteration - the problem is linear: only one step is needed to reach maximum accurracy
    b  = r[inx,iny][:]                  # creates a 1D rhight hand side vector (whitout ghosts), values are the current residual
    # Solve
    du           = .-M.u.u\b              # apply inverse of matrix M.u.u to residual vector 
    u[inx,iny] .+= reshape(du, nc...)   # update the solution u using the correction du
    # Residual check
    ResidualPoisson2D!(r, u, k, s, Jinv, number, type, bc_val, nc, Δ)     
    @info norm(r)/sqrt(length(r))
    
    # Visualization

    ###############################################################
    ########################## NEW STUFF ##########################
    ###############################################################

    # Node list
    vertices = zeros((nc.x+3)*(nc.y+3), 2)
    for I in CartesianIndices(X.c)
        i, j = I[1], I[2]
        v = i + (j-1)*(nc.x+3)
        vertices[v, :] .= X.v[I]
    end

    # Face list
    faces = zeros(Int64, (nc.x+2)*(nc.y+2), 4)
    for I in CartesianIndices(X.c)
        i, j = I[1], I[2]
        c  = i + (j-1)*(nc.x+2)
        v1 = i + (j-1)*(nc.x+3)
        v2 = i + (j-1)*(nc.x+3) + 1
        v3 = i + (j  )*(nc.x+3) + 1
        v4 = i + (j  )*(nc.x+3) 
        faces[c, :] .= [v1, v2, v3, v4]
    end

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
    # cm.mesh!(ax, vertices, faces, color=u[:])
    cm.Colorbar(fig[1, 2], colormap = :turbo, flipaxis = true, size = 10 )    
    display(fig)

    display(to)
end


