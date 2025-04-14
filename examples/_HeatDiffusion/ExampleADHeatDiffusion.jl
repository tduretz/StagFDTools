using StagFDTools, StagFDTools.Poisson, ExtendableSparse, StaticArrays, LinearAlgebra, Statistics, UnPack, Plots
using TimerOutputs
# using Enzyme
using ForwardDiff, Enzyme  # AD backends you want to use 

######

# This function computes the residuals of the 2D Poisson equation using a 5-point stencil
# qx = -k*du_dx
# qy = -k*du_dy
# dqx_dx + dqy_dy = s
# where qx and qy are components of a flux vector (q) in 2D and s is a source term
# The 5-point stencil looks like this:
#      |           |
#      |     uN    |
#      |           |
#----- x -- qyN -- x -----
#      |           |
# uW  qxW    uC   qxE  uE
#      |           |
#----- x -- qyS -- x -----
#      |           |
#      |     uS    |
#      |           |

function Poisson2D(u_loc, k, s, type_loc, bcv_loc, Δ, u0, material)

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
    if type_loc[1,2] === :Dirichlet
        uW = 2*bcv_loc[1,2] - u_loc[2,2]
    elseif type_loc[1,2] === :Neumann
        uW = Δ.x*bcv_loc[1,2] + u_loc[2,2]
    elseif type_loc[1,2] === :periodic || type_loc[1,2] === :in
        uW = u_loc[1,2] 
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
    qxW = -k.xx[1]*(uC - uW)/Δ.x  # West
    qxE = -k.xx[2]*(uE - uC)/Δ.x  # East
    qyS = -k.yy[1]*(uC - uS)/Δ.y  # South
    qyN = -k.yy[2]*(uN - uC)/Δ.y  # North

    # Return the residual function based on finite differences
    return -(-(qxE - qxW)/Δ.x - (qyN - qyS)/Δ.y + s + material.ρ*material.cp*(uC - u0)/Δ.t)  
end

# This function loop over the whole set of 2D cells and computes the residual in each cells
function ResidualPoisson2D!(R, u, k, s, num, type, bc_val, nc, Δ, u0, material)  # u_loc, s, type_loc, Δ

    # This is just a vector of zeros (off-diagonal terms of the conductivity tensor are 0 in the isotropic case)
    # Here StaticArrays are being used a bit everywhere. This is a Julia Library which is used for achieving good performance.
    # Have a look there: https://github.com/JuliaArrays/StaticArrays.jl
    # Basically all small arrays should be contained within such objects (e.g. heat flux vector, matrix containing the stencil values)
    k_loc_shear = @SVector(zeros(2))
                
    shift    = (x=1, y=1)
    
    # Loop over the cells in 2D
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x

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
        R[i,j]    = Poisson2D(u_loc, k_loc, s[i,j], type_loc, bcv_loc, Δ, u0[i,j], material)
    end
    return nothing
end

# The 2 functions below do the same thing. They compute the partial derivatives of the residual function and puts the coefficient in the system matrix.
# One function does it with ForwardDiff: https://github.com/JuliaDiff/ForwardDiff.jl
# The other does it with Enzyme: https://github.com/EnzymeAD/Enzyme.jl

# Computation of the partial derivatives of the residual function. Sets the coefficient in the system matrix.
function AssemblyPoisson_Enzyme!(K, u, k, s, number, type, pattern, bc_val, nc, Δ, u0, material)

    # This is aocal matrix that stores the partial derivatives.
     
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
        autodiff(Enzyme.Reverse, Poisson2D, Duplicated(u_loc, ∂R∂u), Const(k_loc), Const(s[i,j]), Const(type_loc), Const(bcv_loc), Const(Δ), Const(u0[i,j]), Const(material))

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
    pattern = Fields( Fields( @SMatrix([0 1 0; 1 1 1; 0 1 0]) ) )

    # Equation number: a 2D table containing the equation numbers 
    number = Fields( fill(0, (nc.x+2, nc.y+2)) )  # Achtung: geist nodes  
    Numbering!(number, type, nc)                  # There is a StagFD function that does it
    
    @info "Node types"
    printxy(number.u) 

    # 5-point stencil
    pattern = Fields( Fields( @SMatrix([0 1 0; 1 1 1; 0 1 0]) ) )  # Ooops this is done twice, you can remove it :D

    # Parameters
    nt = 1e3
    Δt0 = 0.1
    # niter = 3
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
    material = (
        ρ = 1.0,
        cp = 1e-6,
    )
    r   = zeros(nc.x+2, nc.y+2)  # residual of the equation (right-hand side)
    s   = zeros(nc.x+2, nc.y+2)  # forcing term
    u   = zeros(nc.x+2, nc.y+2)  # solution
    u0  = zeros(nc.x+2, nc.y+2)  # solution of the previous time step
    k   = (x = (xx= ones(nc.x+1,nc.y), xy=zeros(nc.x+1,nc.y)), # conductivity tensor (can be simplidied)
           y = (yx=zeros(nc.x,nc.y+1), yy= ones(nc.x,nc.y+1)))
    Δ   = (x=L/nc.x, y=L/nc.y, t=Δt0)
    xc  = LinRange(-L/2-Δ.x/2, L/2+Δ.x/2, nc.x+2)
    yc  = LinRange(-L/2-Δ.y/2, L/2+Δ.y/2, nc.y+2)
    # Configuration
    s  .= 50*exp.(-(xc.^2 .+ (yc').^2)./0.4^2)
    # Initial condititon
    # u  .= 200*exp.(-(xc.^2 .+ (yc').^2)./0.4^2)
    
    # Sparse matrix assembly
    nu  = maximum(number.u)
    M   = Fields( Fields( ExtendableSparseMatrix(nu, nu) )) 

    for it=1:nt
        u0 .= u  # store the previous solution

        # Residual check: div( q ) - s = r
        @timeit to "Residual" ResidualPoisson2D!(r, u, k, s, number, type, bc_val, nc, Δ, u0, material) 
        @info norm(r)/sqrt(length(r))
        
  
        @timeit to "Assembly Enzyme" begin
            AssemblyPoisson_Enzyme!(M, u, k, s, number, type, pattern, bc_val, nc, Δ, u0, material)
        end

        @info "Symmetry"
        @show norm(M.u.u - M.u.u')
        # A one-step Newton iteration - the problem is linear: only one step is needed to reach maximum accurracy
        b  = r[inx,iny][:]                  # creates a 1D rhight hand side vector (whitout ghosts), values are the current residual
        # Solve
        du           = M.u.u\b              # apply inverse of matrix M.u.u to residual vector 
        u[inx,iny] .-= reshape(du, nc...)   # update the solution u using the correction du
        # Residual check
        ResidualPoisson2D!(r, u, k, s, number, type, bc_val, nc, Δ, u0, material)     
        @info norm(r)/sqrt(length(r))

        # Visualization
        if mod(it, 100) == 0
            p1 = heatmap(xc[inx], yc[iny], u[inx,iny]', aspect_ratio=1, xlim=extrema(xc))
            qx = -diff(u[inx,iny],dims=1)/Δ.x
            qy = -diff(u[inx,iny],dims=2)/Δ.y
            # # @show     mean(qx[1,:])
            # # @show     mean(qx[end,:])
            # # @show     mean(qy[:,1])
            # # @show     mean(qy[:,end])
            heatmap(xc[1:end-3], yc[iny], qx')
            heatmap(xc[inx], yc[1:end-3], qy')
            display(p1)
            display(to)
        end
    end

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
