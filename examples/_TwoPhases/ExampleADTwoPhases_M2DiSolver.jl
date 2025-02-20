using StagFDTools, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use
# import GLMakie

struct NumberingV <: AbstractPattern
    Vx
    Vy
    Pt
    Pf
end

struct Numbering{Tx,Ty,Tp,Tpf}
    Vx::Tx
    Vy::Ty
    Pt::Tp
    Pf::Tpf
end

function Base.getindex(x::Numbering, i::Int64)
    @assert 0 < i < 5 
    i == 1 && return x.Vx
    i == 2 && return x.Vy
    i == 3 && return x.Pt
    i == 4 && return x.Pf
end

function Momentum_x(Vx, Vy, Pt, Pf, η, type, bcv, Δ)
    
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
    fx *= -1
    fx *= Δ.x*Δ.y

    return fx
end

function Momentum_y(Vx, Vy, Pt, Pf, η, type, bcv, Δ)
    
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
    fy *= -1
    fy *= Δ.x*Δ.y

    return fy
end

function Continuity(Vx, Vy, Pt, Pf, ηϕ, ϕ, type_loc, bcv_loc, Δ)
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    fp = ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy + (Pt[1] - Pf[2,2])/((1-ϕ)*ηϕ))
    fp *= ηϕ/(Δ.x+Δ.y)/2*Δ.x*Δ.y
    return fp
end

function FluidContinuity(Vx, Vy, Pt, Pf, ηϕ, ϕ, kμ, type_loc, bcv_loc, Δ)
    
    PfC       = Pf[2,2]

    if type_loc[1,2] === :Dirichlet
        PfW = 2*bcv_loc[1,2] - PfC
    elseif type_loc[1,2] === :Neumann
        PfW = Δ.x*bcv_loc[1,2] + PfC
    elseif type_loc[1,2] === :periodic || type_loc[1,2] === :in
        PfW = Pf[1,2] 
    end

    if type_loc[3,2] === :Dirichlet
        PfE = 2*bcv_loc[3,2] - PfC
    elseif type_loc[3,2] === :Neumann
        PfE = -Δ.x*bcv_loc[3,2] + PfC
    elseif type_loc[3,2] === :periodic || type_loc[3,2] === :in
        PfE = Pf[3,2] 
    end

    if type_loc[2,1] === :Dirichlet
        PfS = 2*bcv_loc[2,1] - PfC
    elseif type_loc[2,1] === :Neumann
        PfS = Δ.y*bcv_loc[2,1] + PfC
    elseif type_loc[2,1] === :periodic || type_loc[2,1] === :in
        PfS = Pf[2,1] 
    end

    if type_loc[2,3] === :Dirichlet
        PfN = 2*bcv_loc[2,3] - PfC
    elseif type_loc[2,3] === :Neumann
        PfN = -Δ.y*bcv_loc[2,3] + PfC
    elseif type_loc[2,3] === :periodic || type_loc[2,3] === :in
        PfN = Pf[2,3] 
    end

    qxW = -kμ.xx[1]*(PfC - PfW)/Δ.x
    qxE = -kμ.xx[2]*(PfE - PfC)/Δ.x
    qyS = -kμ.yy[1]*(PfC - PfS)/Δ.y
    qyN = -kμ.yy[2]*(PfN - PfC)/Δ.y

    fp = (qxE - qxW)/Δ.x + (qyN - qyS)/Δ.y - (Pt[1]-Pf[2,2])/((1-ϕ)*ηϕ)
    fp *= Δ.x*Δ.y

    return fp
end

function ResidualMomentum2D_x!(R, V, P, rheo, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        ηx_loc     = SMatrix{3,3}( rheo.η.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηy_loc     = SMatrix{4,4}( rheo.η.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        Pt_loc     = SMatrix{2,3}(      P.t[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        Pf_loc     = SMatrix{2,3}(      P.f[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vx[i,j] == :in
            R.x[i,j]   = Momentum_x(Vx_loc, Vy_loc, Pt_loc, Pf_loc, η_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, rheo, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,3)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
    ∂R∂Pf = @MMatrix zeros(2,3)
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        ηx_loc     = SMatrix{3,3}( rheo.η.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηy_loc     = SMatrix{4,4}( rheo.η.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        Pt_loc     = MMatrix{2,3}(      P.t[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        Pf_loc     = MMatrix{2,3}(      P.f[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vx[i,j] == :in
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            autodiff(Enzyme.Reverse, Momentum_x, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt_loc, ∂R∂Pt), Duplicated(Pf_loc, ∂R∂Pf), Const(η_loc), Const(type_loc), Const(bcv_loc), Const(Δ))
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
            # Vx --- Pf
            Local = num.Pf[i-1:i,j-2:j] .* pattern[1][4]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][4][num.Vx[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
                end
            end 
        end
    end
    return nothing
end

function ResidualMomentum2D_y!(R, V, P, rheo, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηx_loc     = SMatrix{4,4}( rheo.η.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        ηy_loc     = SMatrix{3,3}( rheo.η.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt_loc     = MMatrix{3,2}(      P.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf_loc     = MMatrix{3,2}(      P.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vy[i,j] == :in
            R.y[i,j]   = Momentum_y(Vx_loc, Vy_loc, Pt_loc, Pf_loc, η_loc, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, rheo, num, pattern, type, BC, nc, Δ) 
    
    ∂R∂Vy = @MMatrix zeros(3,3)
    ∂R∂Vx = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(3,2)
    ∂R∂Pf = @MMatrix zeros(3,2)

    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        ηx_loc     = SMatrix{4,4}( rheo.η.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        ηy_loc     = SMatrix{3,3}( rheo.η.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Pt_loc     = MMatrix{3,2}(      P.t[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        Pf_loc     = MMatrix{3,2}(      P.f[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        η_loc      = (x=ηx_loc, y=ηy_loc)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        if type.Vy[i,j] == :in
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            autodiff(Enzyme.Reverse, Momentum_y, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt_loc, ∂R∂Pt), Duplicated(Pf_loc, ∂R∂Pf), Const(η_loc), Const(type_loc), Const(bcv_loc), Const(Δ))
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
            # Vy --- Pf
            Local = num.Pf[i-2:i,j-1:j] .* pattern[2][4]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vy[i,j]>0
                    K[2][4][num.Vy[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
                end
            end       
        end
    end
    return nothing
end

function ResidualContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    # (; bc_val, type, pattern, num) = numbering
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcx_loc    = SMatrix{3,2}(    BC.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        bcy_loc    = SMatrix{2,3}(    BC.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        typex_loc  = SMatrix{3,2}(  type.Vx[ii,jj] for ii in i:i+2, jj in j:j+1) 
        typey_loc  = SMatrix{2,3}(  type.Vy[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        R.pt[i,j]  = Continuity(Vx_loc, Vy_loc, P.t[i,j], Pf_loc, rheo.ηϕ[i,j], rheo.ϕ[i,j], type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, rheo, num, pattern, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)
    ∂R∂Pt = @MMatrix zeros(1,1)
    ∂R∂Pf = @MMatrix zeros(3,3)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pt_loc     = MMatrix{1,1}(      P.t[ii,jj] for ii in i:i, jj in j:j)
        Pf_loc     = MMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
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
        ∂R∂Pf .= 0.
        autodiff(Enzyme.Reverse, Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt_loc, ∂R∂Pt), Duplicated(Pf_loc, ∂R∂Pf), Const(rheo.ηϕ[i,j]), Const(rheo.ϕ[i,j]), Const(type_loc), Const(bcv_loc), Const(Δ))

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
        # Pt --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[3][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pt[i,j]>0
                K[3][4][num.Pt[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
            end
        end
    end
    return nothing
end

function ResidualFluidContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pf_loc     = SMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc   = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc    = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        k_loc_xx  = @SVector [rheo.kμf.x[i-1,j-1], rheo.kμf.x[i,j-1]]
        k_loc_yy  = @SVector [rheo.kμf.y[i-1,j-1], rheo.kμf.y[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = 0.,
                     yx = 0.,          yy = k_loc_yy)
        R.pf[i,j]  = FluidContinuity(Vx_loc, Vy_loc, P.t[i,j], Pf_loc, rheo.ηϕ[i,j], rheo.ϕ[i,j], k_loc, type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssembleFluidContinuity2D!(K, V, P, rheo, num, pattern, type, BC, nc, Δ) 
                
    shift    = (x=1, y=1)
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)
    ∂R∂Pt = @MMatrix zeros(1,1)
    ∂R∂Pf = @MMatrix zeros(3,3)

    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x
        Pt_loc     = MMatrix{1,1}(      P.t[ii,jj] for ii in i:i, jj in j:j)
        Pf_loc     = MMatrix{3,3}(      P.f[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        type_loc   = SMatrix{3,3}(  type.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcv_loc    = SMatrix{3,3}(    BC.Pf[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        k_loc_xx  = @SVector [rheo.kμf.x[i-1,j-1], rheo.kμf.x[i,j-1]]
        k_loc_yy  = @SVector [rheo.kμf.y[i-1,j-1], rheo.kμf.y[i-1,j]]
        k_loc     = (xx = k_loc_xx,    xy = 0.,
                     yx = 0.,          yy = k_loc_yy)

        ∂R∂Vx .= 0.
        ∂R∂Vy .= 0.
        ∂R∂Pt .= 0.
        ∂R∂Pf .= 0.
        autodiff(Enzyme.Reverse, FluidContinuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(Pt_loc,∂R∂Pt), Duplicated(Pf_loc, ∂R∂Pf), Const(rheo.ηϕ[i,j]), Const(rheo.ϕ[i,j]), Const(k_loc), Const(type_loc), Const(bcv_loc), Const(Δ))
             
        # Pf --- Vx
        Local = num.Vx[i:i+1,j:j+2] .* pattern[4][1]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pf[i,j]>0
                K[4][1][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
            end
        end
        # Pf --- Vy
        Local = num.Vy[i:i+2,j:j+1] .* pattern[4][2]
        for jj in axes(Local,2), ii in axes(Local,1)
            if Local[ii,jj]>0 && num.Pf[i,j]>0
                K[4][2][num.Pf[i,j], Local[ii,jj]] = ∂R∂Vy[ii,jj] 
            end
        end
        # Pf --- Pt
        Local = num.Pt[i,j] .* pattern[4][3]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][3][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pt[ii,jj]  
            end
        end
        # Pf --- Pf
        Local = num.Pf[i-1:i+1,j-1:j+1] .* pattern[4][4]
        for jj in axes(Local,2), ii in axes(Local,1)
            if (Local[ii,jj]>0) && num.Pf[i,j]>0
                K[4][4][num.Pf[i,j], Local[ii,jj]] = ∂R∂Pf[ii,jj]  
            end
        end
           
    end
    return nothing
end

function NumberingTwoPhases!(N, type, nc)
    
    ndof  = 0
    neq   = 0
    noisy = false

    ############ Numbering Vx ############
    periodic_west  = sum(any(i->i==:periodic, type.Vx[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vx[:,2], dims=1)) > 0

    shift  = (periodic_west) ? 1 : 0 
    # Loop through inner nodes of the mesh
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vx[i,j] == :Dirichlet_normal || (type.Vx[i,j] != :periodic && i==nc.x+3-1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vx[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_west
        N.Vx[1,:] .= N.Vx[end-2,:]
    end

    # Copy equation indices for periodic cases
    if periodic_south
        # South
        N.Vx[:,1] .= N.Vx[:,end-3]
        N.Vx[:,2] .= N.Vx[:,end-2]
        # North
        N.Vx[:,end]   .= N.Vx[:,4]
        N.Vx[:,end-1] .= N.Vx[:,3]
    end
    noisy ? printxy(N.Vx) : nothing

    neq = maximum(N.Vx)

    ############ Numbering Vy ############
    ndof  = 0
    periodic_west  = sum(any(i->i==:periodic, type.Vy[2,:], dims=2)) > 0
    periodic_south = sum(any(i->i==:periodic, type.Vy[:,2], dims=1)) > 0
    shift = periodic_south ? 1 : 0
    # Loop through inner nodes of the mesh
    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vy[i,j] == :Dirichlet_normal || (type.Vy[i,j] != :periodic && j==nc.y+3-1)
            # Avoid nodes with constant velocity or redundant periodic nodes
        else
            ndof+=1
            N.Vy[i,j] = ndof  
        end
    end

    # Copy equation indices for periodic cases
    if periodic_south
        N.Vy[:,1] .= N.Vy[:,end-2]
    end

    # Copy equation indices for periodic cases
    if periodic_west
        # West
        N.Vy[1,:] .= N.Vy[end-3,:]
        N.Vy[2,:] .= N.Vy[end-2,:]
        # East
        N.Vy[end,:]   .= N.Vy[4,:]
        N.Vy[end-1,:] .= N.Vy[3,:]
    end
    noisy ? printxy(N.Vy) : nothing

    neq = maximum(N.Vy)

    ############ Numbering Pt ############
    neq_Pt                     = nc.x * nc.y
    N.Pt[2:end-1,2:end-1] .= reshape((1:neq_Pt) .+ 0*neq, nc.x, nc.y)

    if periodic_west
        N.Pt[1,:]   .= N.Pt[end-1,:]
        N.Pt[end,:] .= N.Pt[2,:]
    end

    if periodic_south
        N.Pt[:,1]   .= N.Pt[:,end-1]
        N.Pt[:,end] .= N.Pt[:,2]
    end
    noisy ? printxy(N.Pt) : nothing

    neq = maximum(N.Pt)

    ############ Numbering Pf ############

    neq_Pf                    = nc.x * nc.y
    N.Pf[2:end-1,2:end-1] .= reshape(1:neq_Pf, nc.x, nc.y)

    # Make periodic in x
    for j in axes(type.Pf,2)
        if type.Pf[1,j] === :periodic
            N.Pf[1,j] = N.Pf[end-1,j]
        end
        if type.Pf[end,j] === :periodic
            N.Pf[end,j] = N.Pf[2,j]
        end
    end

    # Make periodic in y
    for i in axes(type.Pf,1)
        if type.Pf[i,1] === :periodic
            N.Pf[i,1] = N.Pf[i,end-1]
        end
        if type.Pf[i,end] === :periodic
            N.Pf[i,end] = N.Pf[i,2]
        end
    end

end

function SetRHS_TwoPhases!(r, R, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            r[ind] = R.x[i,j]
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            r[ind] = R.y[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            r[ind] = R.pt[i,j]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            r[ind] = R.pf[i,j]
        end
    end
end

function UpdateSolution_TwoPhases!(V, P, dx, number, type, nc)

    nVx, nVy, nPt   = maximum(number.Vx), maximum(number.Vy), maximum(number.Pt)

    for j=2:nc.y+3-1, i=3:nc.x+4-2
        if type.Vx[i,j] == :in
            ind = number.Vx[i,j]
            V.x[i,j] += dx[ind] 
        end
    end
    for j=3:nc.y+4-2, i=2:nc.x+3-1
        if type.Vy[i,j] == :in
            ind = number.Vy[i,j] + nVx
            V.y[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pt[i,j] == :in
            ind = number.Pt[i,j] + nVx + nVy
            P.t[i,j] += dx[ind]
        end
    end
    for j=2:nc.y+1, i=2:nc.x+1
        if type.Pf[i,j] == :in
            ind = number.Pf[i,j] + nVx + nVy + nPt
            P.f[i,j] += dx[ind]
        end
    end
end

@views function (@main)(nc)
    
    # Resolution

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_Pt, iny_Pt, size_x, size_y, size_c = Ranges(nc)
    
    # Define node types and set BC flags
    type = Numbering(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    BC = Numbering(
        fill(0., (nc.x+3, nc.y+4)),
        fill(0., (nc.x+4, nc.y+3)),
        fill(0., (nc.x+2, nc.y+2)),
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
    # -------- Pf -------- #
    type.Pf[2:end-1,2:end-1] .= :in
    type.Pf[1,:]             .= :Dirichlet 
    type.Pf[end,:]           .= :Dirichlet 
    type.Pf[:,1]             .= :Dirichlet
    type.Pf[:,end]           .= :Dirichlet
    
    # Equation numbering
    number = Numbering(
        fill(0, (nc.x+3, nc.y+4)),
        fill(0, (nc.x+4, nc.y+3)),
        fill(0, (nc.x+2, nc.y+2)),
        fill(0, (nc.x+2, nc.y+2)),
    )
    NumberingTwoPhases!(number, type, nc)

    # Stencil extent for each block matrix
    pattern = Numbering(
        Numbering(@SMatrix([0 1 0; 1 1 1; 0 1 0]),                 @SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]), @SMatrix([0 1 0;  0 1 0]),        @SMatrix([0 1 0;  0 1 0])), 
        Numbering(@SMatrix([0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]),  @SMatrix([0 1 0; 1 1 1; 0 1 0]),                @SMatrix([0 0; 1 1; 0 0]),        @SMatrix([0 0; 1 1; 0 0])),
        Numbering(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                    @SMatrix([1])),
        Numbering(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]),                    @SMatrix([1 1 1; 1 1 1; 1 1 1])),
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    nPf   = maximum(number.Pf)
    M = Numbering(
        Numbering(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt), ExtendableSparseMatrix(nVx, nPt)), 
        Numbering(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt), ExtendableSparseMatrix(nVy, nPt)), 
        Numbering(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt), ExtendableSparseMatrix(nPt, nPf)),
        Numbering(ExtendableSparseMatrix(nPf, nVx), ExtendableSparseMatrix(nPf, nVy), ExtendableSparseMatrix(nPf, nPt), ExtendableSparseMatrix(nPf, nPf)),
    )

    #--------------------------------------------#
    # Intialise field
    L   = (x=10.0, y=10.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y)
    R   = (x=zeros(size_x...), y=zeros(size_y...), pt=zeros(size_c...), pf=zeros(size_c...))
    V   = (x=zeros(size_x...), y=zeros(size_y...))
    η   = (x= ones(size_x...), y= ones(size_y...), p=ones(size_c...) )
    ηϕ  = ones(size_c...) 
    ϕ   = ones(size_c...) 
    kμf = (x= ones(size_x...), y= ones(size_y...))
    P   = (t=zeros(size_c...), f=zeros(size_c...))
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
    η.y[(xvy.^2 .+ (yvy').^2) .<= 1^2] .= 0.1
    η.x[(xvx.^2 .+ (yvx').^2) .<= 1^2] .= 0.1 
    η.p .= 0.25.*(η.x[1:end-1,2:end-1].+η.x[2:end-0,2:end-1].+η.y[2:end-1,1:end-1].+η.y[2:end-1,2:end-0])
    ηϕ  .= η.p
    ϕ   .= 1e-3
    rheo = (η=η, ηϕ=ηϕ, kμf=kμf, ϕ=ϕ)

    #--------------------------------------------#
    # Residual check
    ResidualContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
    ResidualMomentum2D_x!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualFluidContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 

    # Set global residual vector
    r = zeros(nVx + nVy + nPt + nPf)
    SetRHS_TwoPhases!(r, R, number, type, nc)

    #--------------------------------------------#
    # Assembly
    @info "Assembly, ndof  = $(nVx + nVy + nPt + nPf)"
    AssembleContinuity2D!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)
    AssembleMomentum2D_x!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)
    AssembleMomentum2D_y!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)
    AssembleFluidContinuity2D!(M, V, P, rheo, number, pattern, type, BC, nc, Δ)

    # Two-phases operator as block matrix
    𝑀 = [
        M.Vx.Vx M.Vx.Vy M.Vx.Pt M.Vx.Pf;
        M.Vy.Vx M.Vy.Vy M.Vy.Pt M.Vy.Pf;
        M.Pt.Vx M.Pt.Vy M.Pt.Pt M.Pt.Pf;
        M.Pf.Vx M.Pf.Vy M.Pf.Pt M.Pf.Pf;
    ]

    @info "System symmetry"
    𝑀diff = 𝑀 - 𝑀'
    dropzeros!(𝑀diff)
    @show norm(𝑀diff)

    #--------------------------------------------#
    # # Direct solver 
    @time dx = - 𝑀 \ r


    # A  = [M.Vx.Vx M.Vx.Vy;
    #       M.Vy.Vx M.Vy.Vy]
    
    # B  = [M.Vx.Pt M.Vx.Pf;
    #       M.Vy.Pt M.Vy.Pf;].*1e-0

    # C  = [M.Pt.Vx M.Pt.Vy
    #       M.Pf.Vx M.Pf.Vy] .*1e-0

    # D  = [M.Pt.Pt M.Pt.Pf;
    #       M.Pf.Pt M.Pf.Pf]

    # Ac = cholesky(A)
    # Dc = cholesky(D)

    # A_D_inv = spdiagm(1 ./ diag(A  ))
    # D_D_inv = spdiagm(1 ./ diag(D  ))

    # fv = -r[1:(nVx+nVy)]
    # fp = -r[(nVx+nVy+1):end]
    # dv = zeros(nVx+nVy)
    # dp = zeros(nPt+nPf)
    # dv0 = zeros(nVx+nVy)
    # dp0 = zeros(nPt+nPf)
    # for iter=1:40

    #     rv  = fv - (A*dv + B*dp) 
    #     rp  = fp - (C*dv + D*dp)

    #     dv .= (A_D_inv*A)\(A_D_inv*(fv - B*dp))
    #     dp .= (D_D_inv*D)\(D_D_inv*(fp - C*dv))

    #     @show norm(rv), norm(rp)
    # end

    # dx = zeros(nVx + nVy + nPt + nPf)
    # dx[1:(nVx+nVy)] .= dv
    # dx[(nVx+nVy+1):end] .= dp

    # M2Di solver
    fv    = -r[1:(nVx+nVy)]
    fpt   = -r[(nVx+nVy+1):(nVx+nVy+nPt)]
    fpf   = -r[(nVx+nVy+nPt+1):end]
    dv    = zeros(nVx+nVy)
    dpt   = zeros(nPt)
    dpf   = zeros(nPf)
    rv    = zeros(nVx+nVy)
    rpt   = zeros(nPt)
    rpf   = zeros(nPf)
    rv_t  = zeros(nVx+nVy)
    rpt_t = zeros(nPt)
    s     = zeros(nPf)
    ddv   = zeros(nVx+nVy)
    ddpt  = zeros(nPt)
    ddpf  = zeros(nPf)


    Jvv  = [M.Vx.Vx M.Vx.Vy;
            M.Vy.Vx M.Vy.Vy]
    Jvp  = [M.Vx.Pt;
            M.Vy.Pt]
    Jpv  = [M.Pt.Vx M.Pt.Vy]
    Jpp  = M.Pt.Pt
    Jppf = M.Pt.Pf
    Jpfv = [M.Pf.Vx M.Pf.Vy]
    Jpfp = M.Pf.Pt
    Jpf  = M.Pf.Pf
    Kvv  = Jvv

    @time begin 
        # Pre-conditionning (~Jacobi)
        Jpv_t  = Jpv  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfv
        Jpp_t  = Jpp  - Jppf*spdiagm(1 ./ diag(Jpf  ))*Jpfp
        Jvv_t  = Kvv  - Jvp *spdiagm(1 ./ diag(Jpp_t))*Jpv 
        Jpf_h  = cholesky(Jpf, check = false  )        # Cholesky factors
        Jvv_th = cholesky(Jvv_t, check = false)        # Cholesky factors
        Jpp_th = spdiagm(1 ./diag(Jpp_t));             # trivial inverse
        @views for itPH=1:15
            rv    .= -( Jvv*dv  + Jvp*dpt             - fv  )
            rpt   .= -( Jpv*dv  + Jpp*dpt  + Jppf*dpf - fpt )
            rpf   .= -( Jpfv*dv + Jpfp*dpt + Jpf*dpf  - fpf )
            s     .= Jpf_h \ rpf
            rpt_t .= -( Jppf*s - rpt)
            s     .=    Jpp_th*rpt_t
            rv_t  .= -( Jvp*s  - rv )
            ddv   .= Jvv_th \ rv_t
            s     .= -( Jpv_t*ddv - rpt_t )
            ddpt  .=    Jpp_th*s
            s     .= -( Jpfp*ddpt + Jpfv*ddv - rpf )
            ddpf  .= Jpf_h \ s
            dv   .+= ddv
            dpt  .+= ddpt
            dpf  .+= ddpf
            @printf("  --- iteration %d --- \n",itPH);
            @printf("  ||res.v ||=%2.2e\n", norm(rv)/ 1)
            @printf("  ||res.pt||=%2.2e\n", norm(rpt)/1)
            @printf("  ||res.pf||=%2.2e\n", norm(rpf)/1)
        #     if ((norm(rv)/length(rv)) < tol_linv) && ((norm(rpt)/length(rpt)) < tol_linpt) && ((norm(rpf)/length(rpf)) < tol_linpf), break; end
        #     if ((norm(rv)/length(rv)) > (norm(rv0)/length(rv0)) && norm(rv)/length(rv) < tol_glob && (norm(rpt)/length(rpt)) > (norm(rpt0)/length(rpt0)) && norm(rpt)/length(rpt) < tol_glob && (norm(rpf)/length(rpf)) > (norm(rpf0)/length(rpf0)) && norm(rpf)/length(rpf) < tol_glob),
        #         if noisy>=1, fprintf(' > Linear residuals do no converge further:\n'); break; end
        #     end
        #     rv0=rv; rpt0=rpt; rpf0=rpf; if (itPH==nPH), nfail=nfail+1; end
        end
    end
    
    dx = zeros(nVx + nVy + nPt + nPf)
    dx[1:(nVx+nVy)] .= dv
    dx[(nVx+nVy+1):(nVx+nVy+nPt)] .= dpt
    dx[(nVx+nVy+nPt+1):end] .= dpf

    #--------------------------------------------#
    UpdateSolution_TwoPhases!(V, P, dx, number, type, nc)

    #--------------------------------------------#
    # Residual check
    ResidualContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
    ResidualMomentum2D_x!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualMomentum2D_y!(R, V, P, rheo, number, type, BC, nc, Δ)
    ResidualFluidContinuity2D!(R, V, P, rheo, number, type, BC, nc, Δ) 
    
    @info "Residuals"
    @show norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
    @show norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
    @show norm(R.pt[inx_Pt,iny_Pt])/sqrt(nPt)
    @show norm(R.pf[inx_Pt,iny_Pt])/sqrt(nPf)

    #--------------------------------------------#

    p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vx")
    p2 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc), title="Vy")
    p3 = heatmap(xc, yc, P.t[inx_Pt,iny_Pt]', aspect_ratio=1, xlim=extrema(xc), title="Pt")
    p4 = heatmap(xc, yc, P.f[inx_Pt,iny_Pt]', aspect_ratio=1, xlim=extrema(xc), title="Pf")
    display(plot(p1, p2, p3, p4))

    #--------------------------------------------#
end

main( (x=300, y=300) )