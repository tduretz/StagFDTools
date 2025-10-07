using StagFDTools, StagFDTools.Poisson, ExtendableSparse, StaticArrays, LinearAlgebra, Statistics, UnPack, Plots
using TimerOutputs
# using Enzyme
using ForwardDiff, Enzyme  # AD backends you want to use 
using Distributions
using JLD2
######

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



function Poisson2D(u_loc, k, s, type_loc, bcv_loc, Δxv, Δyv)

    # u_loc is 3*3 matrix containing the current values of u for the whole stencil
    #             0   uN  0
    #     u_loc = uW  uC  uE
    #             0   uS  0
    # Therefore u_loc[2,2] is the current value of uC
    uC       = u_loc[2,2]
    
    # Boundary conditions need to be applied on every boundaries
    # Here we define the values of the ghost nodes. For example at the west side uW needs to be defined  
    # For example, to set a Dirichlet values, we say: 1/2*(uW + uC) = u_BC, hence uW = 2*u_BC - uC
    # West
    if type_loc[1,2] === :Dirichlet 
        uW = (1/Δxv[1])*((Δxv[1]+Δxv[2])*bcv_loc[1,2]-Δxv[2]*uC)
    elseif type_loc[1,2] === :Neumann
        uW = uC + (1/2)*bcv_loc[1,2]*(Δxv[1]+Δxv[2])
    elseif type_loc[1,2] === :periodic || type_loc[1,2] === :in
        uW = u_loc[1,2]
    end

    # East
    if type_loc[3,2] === :Dirichlet
        uE = (1/Δxv[3])*((Δxv[3]+Δxv[2])*bcv_loc[3,2]-Δxv[2]*uC)
    elseif type_loc[3,2] === :Neumann
        uE = uC - (1/2)*bcv_loc[3,2]*(Δxv[2]+Δxv[3])
    elseif type_loc[3,2] === :periodic || type_loc[3,2] === :in
        uE = u_loc[3,2]
    end

    # South
    if type_loc[2,1] === :Dirichlet
        uS = (1/Δyv[1])*((Δyv[1]+Δyv[2])*bcv_loc[2,1]-Δyv[2]*uC)
    elseif type_loc[2,1] === :Neumann
        uS = uC + (1/2)*bcv_loc[2,1]*(Δyv[1]+Δyv[2])
    elseif type_loc[2,1] === :periodic || type_loc[2,1] === :in
        uS = u_loc[2,1]
    end

    # North
    if type_loc[2,3] === :Dirichlet
        uN = (1/Δyv[3])*((Δyv[3]+Δyv[2])*bcv_loc[2,3]-Δyv[2]*uC)
    elseif type_loc[2,3] === :Neumann
        uN = uC - (1/2)*bcv_loc[2,3]*(Δyv[3]+Δyv[2])
    elseif type_loc[2,3] === :periodic || type_loc[2,3] === :in
        uN = u_loc[2,3]
    end

    # Heat flux for each face based on finite differences
    qxW = -k.xx[1]*( (uC - uW) / ( (1/2)*(Δxv[1]+Δxv[2])) )
    qxE = -k.xx[2]*( (uE - uC) / ( (1/2)*(Δxv[2]+Δxv[3])) )
    qyS = -k.yy[1]*( (uC - uS) / ( (1/2)*(Δyv[1]+Δyv[2])) )
    qyN = -k.yy[2]*( (uN - uC) / ( (1/2)*(Δyv[2]+Δyv[3])) )

    # Return the residual function based on finite differences
    return -(-(qxE - qxW)/Δxv[2] - (qyN - qyS)/Δyv[2] + s) * (Δxv[2]*Δyv[2])
end



# This function loop over the whole set of 2D cells and computes the residual in each cells
function ResidualPoisson2D!(R, u, k, s, type, bc_val, nc, Δ)  # u_loc, s, type_loc, Δ

    # This is just a vector of zeros (off-diagonal terms of the conductivity tensor are 0 in the isotropic case)
    # Here StaticArrays are being used a bit everywhere. This is a Julia Library which is used for achieving good performance.
    # Have a look there: https://github.com/JuliaArrays/StaticArrays.jl
    # Basically all small arrays should be contained within such objects (e.g. heat flux vector, matrix containing the stencil values)
    k_loc_shear = @SVector(zeros(2))
                
    shift    = (x=1, y=1)
    
    # Loop over the cells in 2D
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x

        # vecteur dx
        Δxv = @SVector [ Δ.x[i-1], Δ.x[i], Δ.x[i+1] ]
        # vecteur dy
        Δyv = @SVector [ Δ.y[j-1], Δ.y[j], Δ.y[j+1] ]

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
        R[i,j]    = Poisson2D(u_loc, k_loc, s[i,j], type_loc, bcv_loc, Δxv, Δyv)
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

        # vecteur dx
        Δxv = @SVector [ Δ.x[i-1], Δ.x[i], Δ.x[i+1] ]
        # vecteur dy
        Δyv = @SVector [ Δ.y[j-1], Δ.y[j], Δ.y[j+1] ]

        # This is a 3*3 matrix that contains the number of each equation in the stencil (each cell has one corresponding value of u, which has a unique equation number)
        num_loc   = SMatrix{3,3}(number.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1) .* pattern.u.u

        # This is the 3*3 matrix that contains the current stencil values (see residual function above) 
        u_loc     = MMatrix{3,3}(u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        # Same as above: conductivity tensor
        k_loc_xx  = @SVector [k.x.xx[i-1,j-1], k.x.xx[i,j-1]] # west, east
        k_loc_yy  = @SVector [k.y.yy[i-1,j-1], k.y.yy[i-1,j]] # south, north
        k_loc     = (xx = k_loc_xx,    xy = k_loc_shear,
                     yx = k_loc_shear, yy = k_loc_yy)

        # Same as above: boundary condition VALUES
        bcv_loc   = SMatrix{3,3}(bc_val.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        # Same as above: boundary condition TYPES
        type_loc  = SMatrix{3,3}(  type.u[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)

        # Initialise partial derivatives to 0
        ∂R∂u     .= 0e0

        # Here the magic happens: we call a function from Enzyme that computes all, partial derivatives for the current stencil block 
        autodiff(Enzyme.Reverse, Poisson2D, Duplicated(u_loc, ∂R∂u), Const(k_loc), Const(s[i,j]), Const(type_loc), Const(bcv_loc), Const(Δxv), Const(Δyv))

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

function normal_linspace_interval(inflimit::Float64, suplimit::Float64, μ::Float64, σ::Float64, ncells::Int)
    dist = Normal(μ, σ)
    inf_cdf = cdf(dist, inflimit)
    sup_cdf = cdf(dist, suplimit)
    vec = range(inf_cdf, sup_cdf; length=ncells)
    return quantile.(dist, vec)
end


let
    # This is the main code that calls the above functions!
    to = TimerOutput()

    # Resolution in FD cells
    nc = (x = 10, y = 10)

    # Get ranges
    ranges = Ranges(nc)
    (; inx, iny) = ranges

    # Define node types
    type = Fields( fill(:out, (nc.x+2, nc.y+2)) )   # Achtung: geist nodes
    type.u[2:end-1,2:end-1].= :in                   # inside nodes are all type :in    
    type.u[1,:]            .= :Dirichlet            # one BC type is :Dirichlet # West
    type.u[end,:]          .= :Dirichlet            # East
    type.u[:,1]            .= :Neumann #Dirichlet            # South
    type.u[:,end]          .= :Neumann #Dirichlet            # North
    

    # Define values of the boundary conditions
    bc_val = Fields( fill(0., (nc.x+2, nc.y+2)) )   # Achtung: geist nodes
    bc_val.u        .= zeros(nc.x+2, nc.y+2)        # useless ?!
    bc_val.u[1,:]   .= 1.0                          # Boundary value is 1.0 and this will be a Dirichlet (u at West = 1) # West
    bc_val.u[end,:] .= 2.0                          # East
    bc_val.u[:,1]   .= 1.0                          # South
    bc_val.u[:,end] .= 20.0                          # North

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
    
    # Définir le maillage variable (gaussienne, sinh...) 
    true_variable_grid = true
    if true_variable_grid
        
        μ = ( x = 0.0, y = 0.0)
        σ = ( x = 0.2, y = 0.2)
        inflimit = (x = -L/2, y = -L/2)
        suplimit = (x =  L/2, y =  L/2)

        # nodes
        xv_in = normal_linspace_interval(inflimit.x, suplimit.x, μ.x, σ.x, nc.x+1)
        yv_in = normal_linspace_interval(inflimit.y, suplimit.y, μ.y, σ.y, nc.y+1)

        # spaces between nodes
        Δ = (x = zeros(nc.x+2), y = zeros(nc.y+2)) # nb cells
        Δ.x[2:end-1] = diff(xv_in)
        Δ.y[2:end-1] = diff(yv_in)
        Δ.x[1]   = Δ.x[2]
        Δ.x[end] = Δ.x[end-1]
        Δ.y[1]   = Δ.y[2]
        Δ.y[end] = Δ.y[end-1]

        xv  = zeros(nc.x+3)
        yv  = zeros(nc.y+3)
        xv[2:end-1] .= xv_in
        xv[1]   = xv[2] - Δ.x[1]
        xv[end] = xv[end-1] + Δ.x[end]
        yv[2:end-1] .= yv_in
        yv[1]   = yv[2] - Δ.y[1]
        yv[end] = yv[end-1] + Δ.y[end]
        xc = 0.5*(xv[2:end] + xv[1:end-1])
        yc = 0.5*(yv[2:end] + yv[1:end-1])

    else

        Δ   = (x = fill(L/nc.x,nc.x+2), y = fill(L/nc.y,nc.y+2))
        display(Δ.x)
        display(Δ.y)
        xc  = LinRange(-L/2-Δ.x[1]/2, L/2+Δ.x[end]/2, nc.x+2)
        yc  = LinRange(-L/2-Δ.y[1]/2, L/2+Δ.y[end]/2, nc.y+2)
        display(xc)
        display(yc)
    
    end

    # checker la cohérence:          |                   | 
    #                              xv[1]               xv[2]
    #                                |------ dx[1] ----- | 
    #                     xc[1]      |       xc[2]       |
    # Configuration
    s  .= 50*exp.(-(xc.^2 .+ (yc').^2)./0.4^2)

    # Residual check: div( q ) - s = r
    @timeit to "Residual" ResidualPoisson2D!(r, u, k, s, type, bc_val, nc, Δ) 
    @info norm(r)/sqrt(length(r))

    # Sparse matrix assembly
    nu  = maximum(number.u)
    M   = Fields( Fields( ExtendableSparseMatrix(nu, nu) ))
    
    @timeit to "Assembly Enzyme" begin
        AssemblyPoisson_Enzyme!(M, u, k, s, number, type, pattern, bc_val, nc, Δ)
    end

    @info "Symmetry"
    @show norm(M.u.u - M.u.u')
    # A one-step Newton iteration - the problem is linear: only one step is needed to reach maximum accurracy
    b  = r[inx,iny][:]                  # creates a 1D right hand side vector (whitout ghosts), values are the current residual
    # Solve
    du           = .-M.u.u\b              # apply inverse of matrix M.u.u to residual vector 
    u[inx,iny] .+= reshape(du, nc...)   # update the solution u using the correction du

    # Residual check
    @timeit to "Residual" ResidualPoisson2D!(r, u, k, s, type, bc_val, nc, Δ)
    @info norm(r)/sqrt(length(r))

    # Visualization
    p1 = heatmap(xc[inx], yc[iny], u[inx,iny]', aspect_ratio=1, xlim=extrema(xc), title="u")
    qx = -diff(u[:,iny],dims=1)/Δ.x[1]
    qy = -diff(u[inx,:],dims=2)/Δ.y[1]
    p2 = heatmap(xv[2:end-1], yc[iny], qx', aspect_ratio=1, xlim=extrema(xc), title="qx")
    p3 = heatmap(xc[inx], yv[2:end-1], qy', aspect_ratio=1, xlim=extrema(xc), title="qy")
    p4 = spy(M.u.u, title="M")
    display(plot(p1, p2, p3, p4))
    sleep(4)
    display(to)

end


# ────────────────────────────────────────────────────────────────────────────
#                                    Time                    Allocations      
#                           ───────────────────────   ────────────────────────
#     Tot / % measured:          676ms /  82.4%            177MiB /  89.2%    

# Section           ncalls     time    %tot     avg     alloc    %tot      avg
# ────────────────────────────────────────────────────────────────────────────
# Assembly Enzyme        1    115μs   81.7%   137μs    388KiB  100.0%   388KiB
# Residual               1   30.8μs   18.3%  30.8μs     32.0B    0.0%    32.0B
# ────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────
#                                         Time                    Allocations      
#                                ───────────────────────   ────────────────────────
#        Tot / % measured:            438ms /  75.0%            147MiB /  87.0%    

# Section                ncalls     time    %tot     avg     alloc    %tot      avg
# ─────────────────────────────────────────────────────────────────────────────────
# Assembly ForwardDiff        1   94.8μs   79.8%   139μs    294KiB  100.0%   294KiB
# Residual                    1   35.2μs   20.2%  35.2μs     32.0B    0.0%    32.0B
# ─────────────────────────────────────────────────────────────────────────────────
