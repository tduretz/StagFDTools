using StagFDTools.Stokes, ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays
import Statistics:mean
using DifferentiationInterface
using Enzyme  # AD backends you want to use

function Momentum_x(Vx, Vy, Pt, phases, materials, type, bcv, Δ)
    
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

    ε̇xx = Dxx - 1/2*Dkk
    ε̇yy = Dyy - 1/2*Dkk
    ε̇xy = 1/2 * ( Dxy + Dyx )
    
    # Average vertex to centroid
    ε̇̄xy = 0.25*(ε̇xy[1:end-1,1:end-1] .+ ε̇xy[1:end-1,2:end-0] .+ ε̇xy[2:end-0,1:end-1] .+ ε̇xy[2:end,2:end])
    # Average centroid to vertex
    ε̇̄xx = 0.25*(ε̇xx[1:end-1,1:end-1] .+ ε̇xx[1:end-1,2:end-0] .+ ε̇xx[2:end-0,1:end-1] .+ ε̇xx[2:end,2:end])
    ε̇̄yy = 0.25*(ε̇yy[1:end-1,1:end-1] .+ ε̇yy[1:end-1,2:end-0] .+ ε̇yy[2:end-0,1:end-1] .+ ε̇yy[2:end,2:end])

    ε̇II_c = sqrt.(1/2*(ε̇xx[:,2:2].^2 .+ ε̇yy[:,2:2].^2) + ε̇̄xy.^2)
    ε̇II_v = sqrt.(1/2*(ε̇̄xx.^2 .+ ε̇̄yy.^2) + ε̇xy[2:2,1:end].^2)
    n_c   = materials.n[phases.c]
    n_v   = materials.n[phases.v]
    η0_c  = materials.η0[phases.c]
    η0_v  = materials.η0[phases.v]

    η_c  =  η0_c .* ε̇II_c.^(1 ./ n_c .- 1.0 )
    η_v  =  η0_v .* ε̇II_v.^(1 ./ n_v .- 1.0 )

    τxx = 2 * η_c .* ε̇xx[:,2:2]
    τxy = 2 * η_v .* ε̇xy[2:2,1:end]

    fx  = (τxx[2,1] - τxx[1,1]) * invΔx
    fx += (τxy[1,2] - τxy[1,1]) * invΔy
    fx -= ( Pt[2,2] -  Pt[1,2]) * invΔx
    fx *= -1 * Δ.x * Δ.y

    return fx
end

function Momentum_y(Vx, Vy, Pt, phases, materials, type, bcv, Δ)
    
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

    ε̇xx = Dxx - 1/2*Dkk
    ε̇yy = Dyy - 1/2*Dkk
    ε̇xy = 1/2 * ( Dxy + Dyx )
    
    # Average vertex to centroid
    ε̇̄xy = 0.25*(ε̇xy[1:end-1,1:end-1] .+ ε̇xy[1:end-1,2:end-0] .+ ε̇xy[2:end-0,1:end-1] .+ ε̇xy[2:end,2:end])
    # Average centroid to vertex
    ε̇̄xx = 0.25*(ε̇xx[1:end-1,1:end-1] .+ ε̇xx[1:end-1,2:end-0] .+ ε̇xx[2:end-0,1:end-1] .+ ε̇xx[2:end,2:end])
    ε̇̄yy = 0.25*(ε̇yy[1:end-1,1:end-1] .+ ε̇yy[1:end-1,2:end-0] .+ ε̇yy[2:end-0,1:end-1] .+ ε̇yy[2:end,2:end])
  

    ε̇II_c = sqrt.(1/2*(ε̇xx[2:2,:].^2 .+ ε̇yy[2:2,:].^2) + ε̇̄xy.^2)
    ε̇II_v = sqrt.(1/2*(ε̇̄xx.^2 .+ ε̇̄yy.^2) + ε̇xy[1:end,2:2].^2)
    n_c   = materials.n[phases.c]
    n_v   = materials.n[phases.v]
    η0_c  = materials.η0[phases.c]
    η0_v  = materials.η0[phases.v]

    η_c  =  η0_c .* ε̇II_c.^(1 ./ n_c .- 1.0 )
    η_v  =  η0_v .* ε̇II_v.^(1 ./ n_v .- 1.0 )

    τyy = 2 * η_c .* ε̇yy[2:2,:]
    τxy = 2 * η_v .* ε̇xy[1:end,2:2]

    fy  = 0
    fy += (τyy[1,2] - τyy[1,1]) * invΔy
    fy += (τxy[2,1] - τxy[1,1]) * invΔx
    fy -= ( Pt[2,2] -  Pt[2,1]) * invΔy
    fy *= -1 * Δ.x * Δ.y
    
    return fy
end

function Continuity(Vx, Vy, Pt, type_loc, bcv_loc, Δ)
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
    return ((Vx[2,2] - Vx[1,2]) * invΔx + (Vy[2,2] - Vy[2,1]) * invΔy)
end

function ResidualMomentum2D_x!(R, V, P, phases, materials, number, type, BC, nc, Δ) 
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
        phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
        phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-1:i-1, jj in j-2:j-1)
        P_loc      = SMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)
        if type.Vx[i,j] == :in
            R.x[i,j]   = Momentum_x(Vx_loc, Vy_loc, P_loc, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_x!(K, V, P, phases, materials, num, pattern, type, BC, nc, Δ) 

    ∂R∂Vx = @MMatrix zeros(3,3)
    ∂R∂Vy = @MMatrix zeros(4,4)
    ∂R∂Pt = @MMatrix zeros(2,3)
                
    shift    = (x=1, y=2)
    for j in 1+shift.y:nc.y+shift.y, i in 1+shift.x:nc.x+shift.x+1
        
        if type.Vx[i,j] == :in

            bcx_loc    = SMatrix{3,3}(    BC.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            bcy_loc    = SMatrix{4,4}(    BC.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            typex_loc  = SMatrix{3,3}(  type.Vx[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            typey_loc  = SMatrix{4,4}(  type.Vy[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            phc_loc    = SMatrix{2,1}( phases.c[ii,jj] for ii in i-1:i,   jj in j-1:j-1)
            phv_loc    = SMatrix{1,2}( phases.v[ii,jj] for ii in i-1:i-1, jj in j-2:j-1) 
            Vx_loc     = MMatrix{3,3}(      V.x[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
            Vy_loc     = MMatrix{4,4}(      V.y[ii,jj] for ii in i-1:i+2, jj in j-2:j+1)
            P_loc      = MMatrix{2,3}(        P[ii,jj] for ii in i-1:i,   jj in j-2:j  )

            bcv_loc    = (x=bcx_loc, y=bcy_loc)
            type_loc   = (x=typex_loc, y=typey_loc)
            ph_loc     = (c=phc_loc, v=phv_loc)

            
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            autodiff(Enzyme.Reverse, Momentum_x, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
            # Vx --- Vx
            Local = num.Vx[i-1:i+1,j-1:j+1] .* pattern[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) && num.Vx[i,j]>0
                    K[1][1][num.Vx[i,j], Local[ii,jj]] = ∂R∂Vx[ii,jj] 
                end
            end
            # Vx --- Vy
            # Attempt to make it symmetric
            # Local = num.Vy[i-1:i+2,j-2:j+1]' .* pattern[1][2]
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

function ResidualMomentum2D_y!(R, V, P, phases, materials, number, type, BC, nc, Δ)                 
    shift    = (x=2, y=1)
    for j in 1+shift.y:nc.y+shift.y+1, i in 1+shift.x:nc.x+shift.x
        Vx_loc     = MMatrix{4,4}(      V.x[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        Vy_loc     = MMatrix{3,3}(      V.y[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        bcx_loc    = SMatrix{4,4}(    BC.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        bcy_loc    = SMatrix{3,3}(    BC.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        typex_loc  = SMatrix{4,4}(  type.Vx[ii,jj] for ii in i-2:i+1, jj in j-1:j+2)
        typey_loc  = SMatrix{3,3}(  type.Vy[ii,jj] for ii in i-1:i+1, jj in j-1:j+1)
        phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-2:i-1, jj in j-1:j-1) 
        P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)

        if type.Vy[i,j] == :in
            R.y[i,j]   = Momentum_y(Vx_loc, Vy_loc, P_loc, ph_loc, materials, type_loc, bcv_loc, Δ)
        end
    end
    return nothing
end

function AssembleMomentum2D_y!(K, V, P, phases, materials, num, pattern, type, BC, nc, Δ) 
    
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
        phc_loc    = SMatrix{1,2}( phases.c[ii,jj] for ii in i-1:i-1, jj in j-1:j  )
        phv_loc    = SMatrix{2,1}( phases.v[ii,jj] for ii in i-2:i-1, jj in j-1:j-1) 
        P_loc      = MMatrix{3,2}(        P[ii,jj] for ii in i-2:i,   jj in j-1:j  )
        bcv_loc    = (x=bcx_loc, y=bcy_loc)
        type_loc   = (x=typex_loc, y=typey_loc)
        ph_loc     = (c=phc_loc, v=phv_loc)

        if type.Vy[i,j] == :in
            ∂R∂Vx .= 0.
            ∂R∂Vy .= 0.
            ∂R∂Pt .= 0.
            autodiff(Enzyme.Reverse, Momentum_y, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Duplicated(P_loc, ∂R∂Pt), Const(ph_loc), Const(materials), Const(type_loc), Const(bcv_loc), Const(Δ))
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

function ResidualContinuity2D!(R, V, P, phases, materials, number, type, BC, nc, Δ) 
                
    for j in 2:size(R.p,2)-1, i in 2:size(R.p,1)-1
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (;)
        type_loc   = (;)
        R.p[i,j]   = Continuity(Vx_loc, Vy_loc, P[i,j], type_loc, bcv_loc, Δ)
    end
    return nothing
end

function AssembleContinuity2D!(K, V, P, phases, materials, num, pattern, type, BC, nc, Δ) 
                
    ∂R∂Vx = @MMatrix zeros(3,2)
    ∂R∂Vy = @MMatrix zeros(2,3)

    for j in 2:size(P, 2)-1, i in 2:size(P, 1)-1
        Vx_loc     = MMatrix{3,2}(      V.x[ii,jj] for ii in i:i+2, jj in j:j+1)
        Vy_loc     = MMatrix{2,3}(      V.y[ii,jj] for ii in i:i+1, jj in j:j+2)
        bcv_loc    = (;)
        type_loc   = (;)
        
        ∂R∂Vx .= 0.
        ∂R∂Vy .= 0.
        autodiff(Enzyme.Reverse, Continuity, Duplicated(Vx_loc, ∂R∂Vx), Duplicated(Vy_loc, ∂R∂Vy), Const(P[i,j]), Const(type_loc), Const(bcv_loc), Const(Δ))

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
    end
    return nothing
end

@views function main(nc)
    #--------------------------------------------#
    # Resolution

    inx_Vx, iny_Vx, inx_Vy, iny_Vy, inx_Pt, iny_Pt, size_x, size_y, size_c, size_v = Ranges(nc)

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = Fields(
        fill(:out, (nc.x+3, nc.y+4)),
        fill(:out, (nc.x+4, nc.y+3)),
        fill(:out, (nc.x+2, nc.y+2)),
    )
    BC = Fields(
        fill(0., (nc.x+3, nc.y+4)),
        fill(0., (nc.x+4, nc.y+3)),
        fill(0., (nc.x+2, nc.y+2)),
    )
    # -------- Vx -------- #
    type.Vx[inx_Vx,iny_Vx] .= :in       
    type.Vx[2,iny_Vx]       .= :Dir_conf 
    type.Vx[end-1,iny_Vx]   .= :Dir_conf 
    type.Vx[inx_Vx,2]       .= :Neumann
    type.Vx[inx_Vx,end-1]   .= :Neumann
    # -------- Vy -------- #
    type.Vy[inx_Vy,iny_Vy] .= :in       
    type.Vy[2,iny_Vy]       .= :Neumann
    type.Vy[end-1,iny_Vy]   .= :Neumann
    type.Vy[inx_Vy,2]       .= :Dir_conf 
    type.Vy[inx_Vy,end-1]   .= :Dir_conf 
    # -------- Pt -------- #
    type.Pt[2:end-1,2:end-1] .= :in

    #--------------------------------------------#
    # Equation numbering
    number = Fields(
        fill(0, size_x),
        fill(0, size_y),
        fill(0, size_c),
    )
    Numbering!(number, type, nc)

    #--------------------------------------------#
    # Stencil extent for each block matrix
    pattern = Fields(
        Fields(@SMatrix([1 1 1; 1 1 1; 1 1 1]),                 @SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]), @SMatrix([0 1 0; 0 1 0])), 
        Fields(@SMatrix([0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0]),  @SMatrix([1 1 1; 1 1 1; 1 1 1]),                @SMatrix([0 0; 1 1; 0 0])), 
        Fields(@SMatrix([0 1 0; 0 1 0]),                        @SMatrix([0 0; 1 1; 0 0]),                      @SMatrix([1]))
    )

    # Sparse matrix assembly
    nVx   = maximum(number.Vx)
    nVy   = maximum(number.Vy)
    nPt   = maximum(number.Pt)
    M = Fields(
        Fields(ExtendableSparseMatrix(nVx, nVx), ExtendableSparseMatrix(nVx, nVy), ExtendableSparseMatrix(nVx, nPt)), 
        Fields(ExtendableSparseMatrix(nVy, nVx), ExtendableSparseMatrix(nVy, nVy), ExtendableSparseMatrix(nVy, nPt)), 
        Fields(ExtendableSparseMatrix(nPt, nVx), ExtendableSparseMatrix(nPt, nVy), ExtendableSparseMatrix(nPt, nPt))
    )

    #--------------------------------------------#
    # Intialise field
    L  = (x=1.0, y=1.0)
    Δ  = (x=L.x/nc.x, y=L.y/nc.y)
    R  = (x=zeros(size_x...), y=zeros(size_y...), p=zeros(size_c...))
    V  = (x=zeros(size_x...), y=zeros(size_y...))
    η  = (x= ones(size_x...), y= ones(size_y...), p=ones(size_c...) )
    Pt = zeros(size_c...)
    xv = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc = LinRange(-L.x/2+Δ.x/2, L.x/2-Δ.x/2, nc.x)
    yc = LinRange(-L.y/2+Δ.y/2, L.y/2-Δ.y/2, nc.y)
    phases  = (c= ones(Int64, size_c...), v= ones(Int64, size_v...))  # phase on velocity points

    materials = ( 
        n  = [3.0 1.0],
        η0 = [1e2 1e-1] 
    )

    # Initial configuration
    D_BC = [-1  0;
             0  1]
    V.x[inx_Vx,iny_Vx] .= D_BC[1,1]*xv .+ D_BC[1,2]*yc' 
    V.y[inx_Vy,iny_Vy] .= D_BC[2,1]*xc .+ D_BC[2,2]*yv'

    @show size(inx_Vx)
    @show size(inx_Vx)

    BC = ( Vx = zeros(size_x...), Vy = zeros(size_y...))
    BC.Vx[     2, iny_Vx] .= (type.Vx[     1, iny_Vx] .== :Neumann_conf) .* D_BC[1,1]
    BC.Vx[ end-1, iny_Vx] .= (type.Vx[   end, iny_Vx] .== :Neumann_conf) .* D_BC[1,1]
    BC.Vx[inx_Vx,      2] .= (type.Vx[inx_Vx,      2] .== :Neumann) .* D_BC[1,2] .+ (type.Vx[inx_Vx,     2] .== :Dirichlet) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[1]  )
    BC.Vx[inx_Vx,  end-1] .= (type.Vx[inx_Vx,  end-1] .== :Neumann) .* D_BC[1,2] .+ (type.Vx[inx_Vx, end-1] .== :Dirichlet) .* (D_BC[1,1]*xv .+ D_BC[1,2]*yv[end])
    BC.Vy[inx_Vy,     2 ] .= (type.Vy[inx_Vy,     2 ] .== :Neumann_conf) .* D_BC[1,1]
    BC.Vy[inx_Vy, end-1 ] .= (type.Vy[inx_Vy, end-1 ] .== :Neumann_conf) .* D_BC[1,1]
    BC.Vy[     2, iny_Vy] .= (type.Vy[     2, iny_Vy] .== :Neumann) .* D_BC[2,1] .+ (type.Vy[    2, iny_Vy] .== :Dirichlet) .* (D_BC[2,1]*xv[1]   .+ D_BC[2,2]*yv)
    BC.Vy[ end-1, iny_Vy] .= (type.Vy[ end-1, iny_Vy] .== :Neumann) .* D_BC[2,1] .+ (type.Vy[end-1, iny_Vy] .== :Dirichlet) .* (D_BC[2,1]*xv[end] .+ D_BC[2,2]*yv)

    phases.c[inx_Pt, iny_Pt][(xc.^2 .+ (yc').^2) .<= 0.1^2] .= 2
    phases.v[(xv.^2 .+ (yv').^2) .<= 0.1^2] .= 2

    p1 = heatmap(xc, yc, phases.c[inx_Pt,iny_Pt]', aspect_ratio=1, xlim=extrema(xc))
    p2 = heatmap(xv, yv, phases.v', aspect_ratio=1, xlim=extrema(xc))
    display(plot(p1, p2))
    #--------------------------------------------#
    # Newton solver
    niter = 10
    err   = (x = zeros(niter), y = zeros(niter), p = zeros(niter))

    for iter=1:niter
    
        #--------------------------------------------#
        # Residual check
        ResidualContinuity2D!(R, V, Pt, phases, materials, number, type, BC, nc, Δ) 
        ResidualMomentum2D_x!(R, V, Pt, phases, materials, number, type, BC, nc, Δ)
        ResidualMomentum2D_y!(R, V, Pt, phases, materials, number, type, BC, nc, Δ)

        err.x[iter] = norm(R.x[inx_Vx,iny_Vx])/sqrt(nVx)
        err.y[iter] = norm(R.y[inx_Vy,iny_Vy])/sqrt(nVy)
        err.p[iter] = norm(R.p[inx_Pt,iny_Pt])/sqrt(nPt)

        #--------------------------------------------#
        # Set global residual vector
        r = zeros(nVx + nVy + nPt)
        SetRHS!(r, R, number, type, nc)

        #--------------------------------------------#
        # Assembly
        AssembleContinuity2D!(M, V, Pt, phases, materials, number, pattern, type, BC, nc, Δ)
        AssembleMomentum2D_x!(M, V, Pt, phases, materials, number, pattern, type, BC, nc, Δ)
        AssembleMomentum2D_y!(M, V, Pt, phases, materials, number, pattern, type, BC, nc, Δ)

        # Stokes operator as block matrices
        𝐊  = [M.Vx.Vx M.Vx.Vy; M.Vy.Vx M.Vy.Vy]
        𝐐  = [M.Vx.Pt; M.Vy.Pt]
        𝐐ᵀ = [M.Pt.Vx M.Pt.Vy]
        𝐌 = [𝐊 𝐐; 𝐐ᵀ M.Pt.Pt]

        𝐊diff =  𝐊 - 𝐊'
        droptol!(𝐊diff, 1e-11)
        display(𝐊diff)
        
        #--------------------------------------------#
        # Direct solver (TODO: need a better solver)
        dx = - 𝐌 \ r

        #--------------------------------------------#
        # Update solutions (TODO: need a line search here)
        UpdateSolution!(V, Pt, dx, number, type, nc)

    end

    #--------------------------------------------#
    p1 = heatmap(xv, yc, V.x[inx_Vx,iny_Vx]', aspect_ratio=1, xlim=extrema(xc), title="Vx")
    p2 = heatmap(xc, yv, V.y[inx_Vy,iny_Vy]', aspect_ratio=1, xlim=extrema(xc), title="Vy")
    p3 = heatmap(xc, yc,  Pt[inx_Pt,iny_Pt]' .- mean(Pt[inx_Pt,iny_Pt]), aspect_ratio=1, xlim=extrema(xc), title="Pt")
    p4 = plot(xlabel="Iterations", ylabel="log₁₀ error")
    p4 = plot!(1:niter, log10.(err.x[1:niter]), label="Vx")
    p4 = plot!(1:niter, log10.(err.y[1:niter]), label="Vy")
    p4 = plot!(1:niter, log10.(err.p[1:niter]), label="Pt")
    display(plot(p1, p2, p3, p4))
    
end

main((x = 12, y = 11))