using StagFDTools.Stokes, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays
import Statistics:mean
using DifferentiationInterface
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian
# import CairoMakie

include("BasicIterativeSolvers.jl")

# Estimate the Rayleigh Quotient for a vector v
function rayleigh_quotient(A, v)
    return (v' * A * v) / (v' * v)
end

function Base.getindex(x::Numbering, i::Int64)
    @assert 0 < i < 4 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
end

function Momentum_x(Vx, Vy, Pt, η, type, bcv, Δ)
    
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y

    for j=1:4
        if type.y[1,j] == :Dirichlet 
            Vy[1,j] = fma(2, bcv.y[1,j], -Vy[2,j])
        elseif type.y[1,j] == :Neumann
            Vy[1,j] = fma(Δ.x, bcv.y[1,j], Vy[2,j])
        end
        if type.y[4,j] == :Dirichlet 
            Vy[4,j] = fma(2, bcv.y[4,j], -Vy[3,j])
        elseif type.y[4,j] == :Neumann
            Vy[4,j] = fma(Δ.x, bcv.y[4,j], Vy[3,j])
        end
    end

    for i=1:3
        if type.x[i,1] == :Dirichlet 
            Vx[i,1] = fma(2, bcv.x[i,1], -Vx[i,2])
        elseif type.x[i,1] == :Neumann
            Vx[i,1] = fma(Δ.y, bcv.x[i,1], Vx[i,2])
        end
        if type.x[i,end] == :Dirichlet 
            Vx[i,end] = fma(2, bcv.x[i,end], -Vx[i,end-1])
        elseif type.x[i,end] == :Neumann
            Vx[i,end] = fma(Δ.y, bcv.x[i,end], Vx[i,end-1])
        end
    end
     
    Dxx = (Vx[2:end,:] - Vx[1:end-1,:]) * invΔx             # Static Arrays ???
    Dyy = (Vy[2:end-1,2:end] - Vy[2:end-1,1:end-1]) * invΔy             
    Dkk = Dxx + Dyy

    Dxy = (Vx[:,2:end] - Vx[:,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,2:end-1] - Vy[1:end-1,2:end-1]) * invΔx 

    ε̇xx = Dxx - 1/3*Dkk
    ε̇yy = Dyy - 1/3*Dkk
    ε̇xy = 1/2 * ( Dxy + Dyx ) 

    ηc = 0.25*(η.x[1:end-1,:] .+ η.x[2:end,:] .+ η.y[2:end-1,1:end-1] .+ η.y[2:end-1,2:end])
    ηv = 0.25*(η.x[:,1:end-1] .+ η.x[:,2:end] .+ η.y[1:end-1,2:end-1] .+ η.y[2:end,2:end-1])

    τxx = 2 * ηc .* ε̇xx
    τxy = 2 * ηv .* ε̇xy

    fx  = (τxx[2,2] - τxx[1,2]) * invΔx 
    fx += (τxy[2,2] - τxy[2,1]) * invΔy 
    fx -= ( Pt[2,2] -  Pt[1,2]) * invΔx
    # fx *= Δ.x*Δ.y

    return fx
end

function Momentum_y(Vx, Vy, Pt, η, type, bcv, Δ)
    
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    
    for i=1:4
        if type.x[i,1] == :Dirichlet 
            Vx[i,1] = fma(2, bcv.x[i,1], -Vx[i,2])
        elseif type.x[i,1] == :Neumann
            Vx[i,1] = fma(Δ.y, bcv.x[i,1], Vx[i,2])
        end
        if type.x[i,4] == :Dirichlet 
            Vx[i,4] = fma(2, bcv.x[i,4], -Vx[i,3])
        elseif type.x[i,4] == :Neumann
            Vx[i,4] = fma(Δ.y, bcv.x[i,4], Vx[i,3])
        end
    end

    for j=1:3
        if type.y[1,j] == :Dirichlet 
            Vy[1,j] = fma(2, bcv.y[1,j], -Vy[2,j])
        elseif type.y[1,j] == :Neumann
            Vy[1,j] = fma(Δ.x, bcv.y[1,j], Vy[2,j])
        end
        if type.y[end,j] == :Dirichlet 
            Vy[end,j] = fma(2, bcv.y[end,j], -Vy[end-1,j])
        elseif type.y[end,j] == :Neumann
            Vy[end,j] = fma(Δ.x, bcv.y[end,j], Vy[end-1,j])
        end
    end
     
    Dxx = (Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1]) * invΔx             # Static Arrays ???
    Dyy = (Vy[:,2:end] - Vy[:,1:end-1]) * invΔy             
    Dkk = Dxx + Dyy

    Dxy = (Vx[2:end-1,2:end] - Vx[2:end-1,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,:] - Vy[1:end-1,:]) * invΔx 

    ε̇xx = Dxx - 1/3*Dkk
    ε̇yy = Dyy - 1/3*Dkk
    ε̇xy = 1/2 * ( Dxy + Dyx ) 

    ηc = 0.25*(η.x[1:end-1,2:end-1] .+ η.x[2:end,2:end-1] .+ η.y[:,1:end-1] .+ η.y[:,2:end])
    ηv = 0.25*(η.x[2:end-1,1:end-1] .+ η.x[2:end-1,2:end] .+ η.y[1:end-1,:] .+ η.y[2:end,:])

    τyy = 2 * ηc .* ε̇yy
    τxy = 2 * ηv .* ε̇xy

    fy  = (τyy[2,2] - τyy[2,1]) * invΔy 
    fy += (τxy[2,2] - τxy[1,2]) * invΔx 
    fy -= (Pt[2,2] - Pt[2,1]) * invΔy
    # fy *= Δ.x*Δ.y

    return fy
end

function Continuity(Vx, Vy, Pt, η, type_loc, bcv_loc, Δ)
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    fp = ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy + 0*Pt[1]/(η))
    # fp *= η/(Δ.x+Δ.y)
    return fp
end

function ResidualMomentum2D_x!(R, V, P, η, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        ηx_loc     = SMatrix{3,3}(      η.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηy_loc     = SMatrix{4,4}(      η.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        P_loc      = SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vx[i,j] == :in
            R.x[i,j]   = Momentum_x(Vx_loc, Vy_loc, P_loc, η_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, η, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,3)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        ηx_loc     = SMatrix{3,3}(      η.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηy_loc     = SMatrix{4,4}(      η.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        P_loc      = MMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vx[i,j] == :in
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            forwarddiff_gradients!(Momentum_x, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(η_loc), Const(type_loc), Const(bcv_loc), Const(Δ))
            # Vx --- Vx
            Local = num.Vx[i-1:i+1,j-1:j+1] .* pattern[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][1][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vx --- Vy
            Local = num.Vy[i-1:i+2,j-2:j+1] .* pattern[1][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][2][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vx --- Pt
            Local = num.Pt[i-1:i,j-2:j] .* pattern[1][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][3][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end 
        end
    end
    return nothing
end

function ResidualMomentum2D_y!(R, V, P, η, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηx_loc     = SMatrix{4,4}(      η.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        ηy_loc     = SMatrix{3,3}(      η.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vy[i,j] == :in
            R.y[i,j]   = Momentum_y(Vx_loc, Vy_loc, P_loc, η_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, η, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(3,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηx_loc     = SMatrix{4,4}(      η.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        ηy_loc     = SMatrix{3,3}(      η.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vy[i,j] == :in
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            forwarddiff_gradients!(Momentum_y, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(η_loc), Const(type_loc), Const(bcv_loc), Const(Δ))
            # Vy --- Vx
            Local = num.Vx[i-2:i+1,j-1:j+2] .* pattern[2][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][1][num.Vy[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vy --- Vy
            Local = num.Vy[i-1:i+1,j-1:j+1] .* pattern[2][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][2][num.Vy[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj]  
                end
            end
            # Vy --- Pt
            Local = num.Pt[i-2:i,j-1:j] .* pattern[2][3]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][3][num.Vy[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
                end
            end       
        end
    end
    return nothing
end

function ResidualContinuity2D!(R, V, P, η, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    # (; bc_val, type, pattern, num) = numbering
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcx_loc    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        bcy_loc    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        typex_loc  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        typey_loc  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        R[i,j]     = Continuity(Vx_loc, Vy_loc, P[i,j], η.p[i,j], type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssembleContinuity2D!(K, V, Pt, η, num, pattern, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    # (; bc_val, type, pattern, num) = numbering
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)
    ∂R∂Pt = @MMatrix zeros(1,1)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pt_loc     = MMatrix{1,1}(       Pt[ii,jj] for ii in i:i, jj in j:j)
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcx_loc    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        bcy_loc    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        typex_loc  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        typey_loc  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        
        ∂R∂Vx .= 0.
        ∂R∂Vy .= 0.
        ∂R∂Pt .= 0.
        forwarddiff_gradients!(Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt_loc, ∂R∂Pt), Const(η.p[i,j]), Const(type_loc), Const(bcv_loc), Const(Δ))

        # Pt --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[3][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][1][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pt --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[3][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pt[i,j]>0
                K[3][2][num.Pt[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end
         # Pt --- Pt
         Local = num.Pt[i,j] .* pattern[3][3]
         for jj in axes(Local,2), ii in axes(Local,1)
             if (Local[ii,jj]>0) && num.Pt[i,j]>0
                 K[3][3][num.Pt[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
             end
         end
    end
    return nothing
end

@views function main(nc) 
    #--------------------------------------------#
    # Resolution
    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_c, iny_c, size_x, size_y, size_c = Ranges(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = Numbering(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    BC = Numbering(
        fill(0., (nc.x+3, nc.y+4)),
        fill(0., (nc.x+4, nc.y+3)),
        fill(0., (nc.x+2, nc.y+2)),
    )
    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx] .= :in       
    type.Vx[2,iny_Vx]       .= :Dirichlet_normal 
    type.Vx[end-1,iny_Vx]   .= :Dirichlet_normal 
    type.Vx[inx_Vx,2]       .= :Neumann
    type.Vx[inx_Vx,end-1]   .= :Neumann
    BC.Vx[2,iny_Vx]         .= 0.0
    BC.Vx[end-1,iny_Vx]     .= 0.0
    BC.Vx[inx_Vx,2]         .= 0.0
    BC.Vx[inx_Vx,end-1]     .= 0.0
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy] .= :in       
    type.Vy[2,iny_Vy]       .= :Neumann
    type.Vy[end-1,iny_Vy]   .= :Neumann
    type.Vy[inx_Vy,2]       .= :Dirichlet_normal 
    type.Vy[inx_Vy,end-1]   .= :Dirichlet_normal 
    BC.Vy[2,iny_Vy]         .= 0.0
    BC.Vy[end-1,iny_Vy]     .= 0.0
    BC.Vy[inx_Vy,2]         .= 0.0
    BC.Vy[inx_Vy,end-1]     .= 0.0
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in

    #--------------------------------------------#
    # Equation numbering
    number = Numbering(
        fill(0, size_x),
        fill(0, size_y),
        fill(0, size_c),
    )
    Numbering!(number, type, nc)

    #--------------------------------------------#
    # Stencil extent for each block matrix
    pattern = Numbering(
        Numbering(@SMatrix([0 1 0; 1 1 1; 0 1 0]),                 @SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]), @SMatrix([0 1 0; 0 1 0])), 
        Numbering(@SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]),  @SMatrix([0 1 0; 1 1 1; 0 1 0]),                @SMatrix([0 0; 1 1; 0 0])), 
        Numbering(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Numbering(
        Numbering(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Numbering(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Numbering(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )

    #--------------------------------------------#
    # Intialise field
    L   = (x=10.0, y=10.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y)
    R   = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (x= ones(size_x...), y= ones(size_y...), p=ones(size_c...) )
    Rp  = zeros(size_c...)
    Pt  = zeros(size_c...)
    xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc  = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc  = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    xvx = LinRange(-L.x/2-Δ.x, L.x/2+Δ.x, nc.x+3)
    xvy = LinRange(-L.x/2-3Δ.x/2, L.x/2+3Δ.x/2, nc.x+4)
    yvy = LinRange(-L.y/2-Δ.y, L.y/2+Δ.y, nc.y+3)
    yvx = LinRange(-L.y/2-3Δ.y/2, L.y/2+3Δ.y/2, nc.y+4)

    # Initial configuration
    ε̇  = -1.0
    V.x[inx_Vx,iny_Vx] .=  ε̇*xv .+ 0*yc' 
    V.y[inx_Vy,iny_Vy] .= 0*xc .-  ε̇*yv' 
    η.y[(xvy.^2 .+ (yvy').^2) .<= 1^2] .= 0.01
    η.x[(xvx.^2 .+ (yvx').^2) .<= 1^2] .= 0.01 
    η.p .= 0.25.*(η.x[1:end-1,2:end-1].+η.x[2:end-0,2:end-1].+η.y[2:end-1,1:end-1].+η.y[2:end-1,2:end-0])
    
    #--------------------------------------------#
    # Residual check
    ResidualContinuity2D!(Rp, V, Pt, η, number, type, BC, nc, Δ) 
    ResidualMomentum2D_x!(R,  V, Pt, η, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R,  V, Pt, η, number, type, BC, nc, Δ)

    # Set global residual vector
    r = zeros(nVx + nVy + nPt)
    SetRHS!(r, R, number, type, nc)

    #--------------------------------------------#
    # Assembly
    @info "Assembly, ndof  = $(nVx + nVy + nPt)"
    AssembleContinuity2D!(M, V, Pt, η, number, pattern, type, BC, nc, Δ)
    AssembleMomentum2D_x!(M, V, Pt, η, number, pattern, type, BC, nc, Δ)
    AssembleMomentum2D_y!(M, V, Pt, η, number, pattern, type, BC, nc, Δ)

    # Stokes operator as block matrices
    K  = [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
    Q  = [M.Vx.Pt; M.Vy.Pt]
    Qᵀ = [M.Pt.Vx M.Pt.Vy]
    𝑀 = [K Q; Qᵀ M.Pt.Pt]

    #--------------------------------------------#
    # Direct solver 
    # dx = - 𝑀 \ r

    #--------------------------------------------#
    # Iterative solver 
    D_PC    = I(size(𝑀,1)) # no preconditioner

    # Diagonal preconditioner
    D_PC    = spdiagm(diag(𝑀))
    diag_Pt = max(nc...) ./ η.p[inx_c, iny_c]
    D_PC[(nVx+nVy+1):end, (nVx+nVy+1):end] .+= spdiagm(diag_Pt[:])
    D_PC_inv =  spdiagm(1 ./ diag(D_PC))

    # dx = preconditioned_minres(𝑀, -r, ApplyPC, D_PC_inv)
    # dx = preconditioned_bicgstab(𝑀, b, ApplyPC, D_PC_inv)
   
    Xmax = copy(r)
    Xmin = copy(r)
    Lmax = 0.
    Lmin = 0.
    for iter in 1:100
        Xmax .= (𝑀*D_PC_inv) * Xmax / norm((𝑀*D_PC_inv) * Xmax)
        Lmax = rayleigh_quotient(𝑀*D_PC_inv, Xmax)
        Xmin .= ( D_PC_inv) * Xmin / norm((D_PC_inv) * Xmin)
        Lmin = rayleigh_quotient( D_PC_inv, Xmin)
        @show Lmin, Lmax
    end

    Dinv_K    = spdiagm(1 ./ diag(K))
    # Dinv_M    = spdiagm(1 ./diag_Pt[:])
    # SC = Dinv_M * (Qᵀ*(Dinv_K*K)*Q) 
    # X = r[1:size(Qᵀ,1)]
    # Lmax = 0.
    # for iter in 1:100
    #     X .= SC * X / norm(SC * X)
    #     # @show Lmax = rayleigh_quotient(SC, X)
    # end

    @show maximum(sum(abs.(Dinv_K * K), dims=2)), maximum(sum(abs.(Dinv_K * Q), dims=2))
    @show maximum(sum(abs.(Dinv_K * K), dims=1)), maximum(sum(abs.(Dinv_K * Q), dims=1))
    
    @show maximum(sum(abs.(Dinv_K * (K)), dims=2)), maximum(sum(abs.(Dinv_K * Q), dims=2))


    @show Lmin = maximum( sum(D_PC_inv*D_PC, dims=2) + sum(-abs.(D_PC_inv * (𝑀 - D_PC)), dims=2) )

    # @show Lmax = maximum(sum(abs.(Dinv_K * K), dims=2))*size(Q,1)/size(𝑀,1) +  maximum(sum(abs.(Dinv_K * Q), dims=2))*size(Q,2)/size(𝑀,1)

    # dx = preconditioned_chebyshev_with_spectral_bounds(𝑀, -r, ApplyPC, D_PC_inv, zeros(size(r)), 50000, 1e-5, 2.02*Lmax)
    dx = SolChebyshev002(𝑀, -r, ApplyPC, D_PC_inv, zeros(size(r)), 50000, 1e-5, Lmin, Lmax)

    #--------------------------------------------#

    # Dinv   = (x=zeros(size_x...), y=zeros(size_y...))
    # Dinv_p = zeros(size_c...)
    # UpdateSolution!(Dinv, Dinv_p, diag(D_PC_inv), number, type, nc)

    # # #--------------------------------------------#
    # n = nVx + nVy + nPt

    # dV   = (x=zeros(size_x...), y=zeros(size_y...))
    # dPt  = zeros(size_c...)

    # Ap   = (x=zeros(size_x...), y=zeros(size_y...))
    # Ap_p = zeros(size_c...)
    # z    = (x=zeros(size_x...), y=zeros(size_y...))
    # z_p  = zeros(size_c...)
    # p    = (x=zeros(size_x...), y=zeros(size_y...))
    # p_p  = zeros(size_c...)

    # # Initial guess (zero vector)
    # dV.x .= 0.; dV.y .= 0.; dPt  .= 0.
    
    # # Initial residual and preconditioned residual
    # z.x  .= Dinv.x.*R.x; z.y  .= Dinv.y.*R.y; z_p   .= Dinv_p.*Rp
    # p.x  .= z.x;          p.y .= z.y;         p_p   .= z_p
    
    # # Initialize residual and preconditioned residual
    # norm_r0 = sqrt(sum(R.x.*R.x) + sum(R.y.*R.y) + sum(Rp.*Rp)) 

    # max_iter = n
    # tol      = 1e-8
    
    # # Iteration loop
    # for k in 1:max_iter

    #     # Compute A * p
    #     ResidualContinuity2D!(Ap_p, p, p_p, η, number, type, BC, nc, Δ) 
    #     ResidualMomentum2D_x!(Ap,   p, p_p, η, number, type, BC, nc, Δ)
    #     ResidualMomentum2D_y!(Ap,   p, p_p, η, number, type, BC, nc, Δ)
        
    #     # Compute step size alpha
    #     r_dot_z = (dot(R.x, z.x) + dot(R.y, z.y) + dot(Rp, z_p))
    #     alpha   = r_dot_z / (dot(p.x, Ap.x) + dot(p.y, Ap.y) + dot(p_p, Ap_p) )
 
    #     # Update the solution vector x
    #     dV.x .+= alpha * p.x
    #     dV.y .+= alpha * p.y
    #     dPt  .+= alpha * p_p
        
    #     # Compute new residual
    #     R.x .-= alpha * Ap.x
    #     R.y .-= alpha * Ap.y
    #     Rp  .-= alpha * Ap_p
    #     norm_r_new = sqrt(sum(R.x.*R.x) + sum(R.y.*R.y) + sum(Rp.*Rp)) 
        
    #     # Check for convergence
    #     if norm_r_new / norm_r0 < tol  #|| norm_r_new/sqrt(n) < 2*tol 
    #         println("Converged in $k iterations.")
    #         break
    #     end
        
    #     # Apply preconditioner to the new residual
    #     z.x .= Dinv.x.*R.x; z.y .= Dinv.y.*R.y; z_p  .= Dinv_p.*Rp
        
    #     # Compute the beta value for the direction update
    #     beta = (dot(R.x, z.x) + dot(R.y, z.y) + dot(Rp, z_p)) / r_dot_z

    #     # Update the direction p and residual r
    #     p.x .= z.x .+ beta .* p.x
    #     p.y .= z.y .+ beta .* p.y
    #     p_p .= z_p .+ beta .* p_p
    # end

    # dx = zeros(nVx + nVy + nPt)
    # Δx = (x=dV.x, y=dV.y, p=dPt )
    # SetRHS!(dx, Δx, number, type, nc)

    #--------------------------------------------#
    UpdateSolution!(V, Pt, dx, number, type, nc)

    #--------------------------------------------#
    # Residual check
    ResidualContinuity2D!(Rp, V, Pt, η, number, type, BC, nc, Δ) 
    ResidualMomentum2D_x!(R,  V, Pt, η, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R,  V, Pt, η, number, type, BC, nc, Δ)
    
    @info "Residuals"
    @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
    @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
    @show norm(Rp[inx_c,iny_c])/sqrt(nPt)

    #--------------------------------------------#
    @info "Velocity block symmetry"
    # display(K - K')
    @show norm(K-K')
    𝑀diff = 𝑀 - 𝑀'
    dropzeros!(𝑀diff)
    @show norm(𝑀diff)
    # # f = CairoMakie.spy(rotr90(𝑀diff))
    # # f = CairoMakie.spy(rotr90(𝑀))
    # f = CairoMakie.spy(rotr90(D_PC_inv))
    # CairoMakie.DataInspector(f)
    # display(f)

    #--------------------------------------------#

    p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc))
    p2 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc))
    p3 = heatmap(xc, yc,  Pt[inx_c,iny_c]' .- mean(Pt[inx_c,iny_c]), aspect_ratio=1, xlim=extrema(xc))
    display(plot(p1, p2, p3))

    #--------------------------------------------#
end

# Preconditioned Chebyshev Iteration with spectral bounds tracking (min and max eigenvalues)
function preconditioned_chebyshev_with_spectral_bounds(A, b, ApplyPC, Dinv, x0, max_iter, tol, Lmax)
    x = copy(x0)  # Initial guess
    r = b - A * x  # Initial residual
    norm_b = norm(b)  # Norm of the right-hand side
    @show residual = norm(r) / norm_b  # Relative residual
    
    # Preconditioned residual
    z = ApplyPC(Dinv, r)  # Solve M*z = r, where M is the preconditioner (typically M is symmetric)
    
    # Initial spectral bounds estimation using the Rayleigh quotient on the residual
    λ_min_estimate = 0#rayleigh_quotient(A, z)  # Approximate minimum eigenvalue using residual
    λ_max_estimate = Lmax;#2.02*Lmax#rayleigh_quotient(A, r)  # This will be refined
    alpha = 2 / (λ_max_estimate - λ_min_estimate)
    beta = (λ_max_estimate + λ_min_estimate) / (λ_max_estimate - λ_min_estimate)
    
    @show λ_min_estimate, λ_max_estimate

    nout = 2

    X = copy(r)

    # Track the smallest and largest eigenvalues dynamically
    for k in 1:max_iter
        if residual < tol
            println("Convergence reached after $k iterations")
            break
        end
        
        # Apply Chebyshev iteration with preconditioned residual
        r_new = b - A * x  # New residual
        z_new = ApplyPC(Dinv, r_new)  # Preconditioned new residual: M * z_new = r_new
        
        # Chebyshev update: Update the solution using alpha and beta
        x += (alpha * z + beta * z_new)  # Update solution
        # Update residual and preconditioned residual
        r = r_new
        z = z_new
        
        # Estimate the maximum eigenvalue (use the Rayleigh quotient on r)
        # λ_max_estimate = rayleigh_quotient(A, r)  # Update the estimate of the maximum eigenvalue
        

        # For the minimum eigenvalue, we use an inverse iteration-like approach
        # Here, you can apply a simple inverse power iteration method for better accuracy
        # Inverse iteration (preconditioned) applied to the residual
        if mod(k, 500)==0

            X = Dinv*A*X / norm(Dinv*A*X)
            @show rayleigh_quotient(Dinv*A, X)

            # z_inv = ApplyPC(Dinv, r)  # Applying preconditioner (inverse iteration)
            # λ_min_estimate = (rayleigh_quotient(A, z_inv)) / 8  # Update the estimate of the minimum eigenvalue
            # # λ_min_estimate = rayleigh_quotient(Dinv*A, r) * 2.2
            
            # Dynamically adjust alpha and beta based on updated eigenvalue estimates
            alpha = 2 / (λ_max_estimate - λ_min_estimate)
            beta = (λ_max_estimate + λ_min_estimate) / (λ_max_estimate - λ_min_estimate)
            
            # Compute the relative residual
            residual = norm(r) / norm_b
            isnan(residual) ? error("nan") : nothing
        
            # Optionally, print or track the estimated spectral bounds
            println("Iteration $k: res: $(residual) --- Lmin: $λ_min_estimate, Lmax: $λ_max_estimate")
        end
         # Compute the relative residual
         residual = norm(r) / norm_b
         isnan(residual) ? error("nan") : nothing
    end
    
    if residual >= tol
        println("Maximum iterations reached, solution may not have converged.")
    end
    
    return x
end


function SolChebyshev002(A, b, ApplyPC, Dinv, x0, max_iter, tol, Lmin, Lmax)

    # Lmax = 2.26
    # Lmax = 2.26
    # Lmin /= 5
    Lmin = 0
    Lmax *= 1.1
    d = (Lmax + Lmin) / 2;
    c = (Lmax - Lmin) / 2;
    beta = 0.0
    alpha = 0
    x = x0;
    r = b - A * x;
    nr0 = norm(r)
    p = copy(x0)

    r0 = copy(r) 

    # theta = (Lmin+Lmax)/2
    # delta = (Lmax-Lmin)/2
    # sigma1 = theta/delta
    # rho = 1/sigma1
    # w = ApplyPC(Dinv, r)
    # d = 1/theta * w 
    # for i = 1:max_iter
    #     x .+= d
    #     r .-= A * d
    #     w   = ApplyPC(Dinv, r)
    #     # print ("Iteration", it, "err=", err)
    #     d .*= rho
    #     d .+= 2/delta * ApplyPC(Dinv, r)
    #     rho = 1/(2*sigma1-rho)
    #     d .*= rho
    #     if mod(i, 100)==0
    #         @show norm(r)/nr0 
    #         isnan( norm(r)) ? error("nan") : nothing
    #     end
    # end
  
    X = copy(r)
    for i = 1:max_iter
        r0 .= r 
        z = ApplyPC(Dinv, r)
        if (i == 1)
            p .= z;
            alpha = 1/d;
        elseif (i == 2)
            beta = (1/2) * (c * alpha)^2
            alpha = 1/(d - beta / alpha);
            p .= z + beta * p;
        else
            beta = (c * alpha / 2)^2;
            alpha = 1/(d - beta / alpha);
            p .= z + beta * p;
        end;
  
        x .= x + alpha * p;
        r .= b - A * x; #%(= r - alpha * A * p)
        if mod(i, 100)==0 || i == 1
            @show (rayleigh_quotient(Dinv*A,  r))
            dr = r - r0

            X = Dinv*A*X / norm(Dinv*A*X)
            # @show Lmax = rayleigh_quotient(Dinv*A, X)

            @show   Lmin = abs(dot(p, alpha*(Dinv*p)) / dot( z,p) )

            d = (Lmax + Lmin) / 2;
            c = (Lmax - Lmin) / 2;
            
            @show norm(r)/nr0 
            isnan( norm(r)) ? error("nan") : nothing
        end
        if (norm(r)/nr0 < tol) 
            @show i
            break; 
        end; # stop if necessary
    end;
    @show Lmin, Lmax

    return x
  end

main((x = 30, y = 32))
main((x = 2*30, y = 2*32))
main((x = 4*30, y = 4*32))
main((x = 8*30, y = 8*32))