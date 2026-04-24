using StagFDTools, StagFDTools.StokesFSG
using ExtendableSparse, StaticArrays, Plots, LinearAlgebra, SparseArrays, Printf
import Statistics:mean
using DifferentiationInterface
using StagFDTools: Duplicated, Const, forwarddiff_gradients!, forwarddiff_gradient, forwarddiff_jacobian
max_all(R) = max(maximum(R.x[1]), maximum(R.x[2]), maximum(R.y[1]), maximum(R.y[2]), maximum(R.p[1]), maximum(R.p[2]) )

const rheology = :anisotropic
# const rheology = :powerlaw

function ViscosityTensor(η0, δ, n, engineering)
    two   = engineering ? 2 : 1
    μ_N   = η0
    C_ISO = 2 * μ_N * [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0/two] # Viscosity tensor for isotropic flow

    # we need to normalise the director every time it is updated
    Norm_dir   = norm(n)
    n ./= Norm_dir

    # once we know then we compute anisotropy matrix
    a0 = 2 * n[1]^2 * n[2]^2
    a1 = n[1] * n[2] * (-n[1]^2 + n[2]^2)

    # build the matrix 
    C_ANI = [-a0 a0 2*a1/two; a0 -a0 -2*a1/two; a1 -a1 (-1+2*a0)/two]

    # operator
    μ_S = μ_N / δ
    𝐷     = C_ISO + 2 * (μ_N - μ_S) * C_ANI 
    return  𝐷
end

function Momentum_x(Vx, V̄x, Vy, V̄y, Pt, P̄t, phase, p̄hase, materials, tx, t̄x, ty, t̄y, bc_val, Δ)
    
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y

    # SW
    if ty[1,1] == :Neu_norm_half
        Vy[1,1] = Vy[1,2] - Δ.y*bc_val.D[2,2]
    elseif ty[1,1] == :Dir_norm_half
        Vy[1,1] = 2*bc_val.y.N[1] - Vy[1,2]
    end

    # SE
    if ty[2,1] == :Neu_norm_half
        Vy[2,1] = Vy[2,2] - Δ.y*bc_val.D[2,2]
    elseif ty[2,1] == :Dir_norm_half
        Vy[2,1] = 2*bc_val.y.N[2] - Vy[2,2]
    end

    # NW
    if ty[1,2] == :Neu_norm_half
        Vy[1,2] = Vy[1,1] + Δ.y*bc_val.D[2,2]
    elseif ty[1,2] == :Dir_norm_half
        Vy[1,2] = 2*bc_val.y.N[1] - Vy[1,1]
    end

    # NE
    if ty[2,2] == :Neu_norm_half
        Vy[2,2] = Vy[2,1] + Δ.y*bc_val.D[2,2]
    elseif ty[2,2] == :Dir_norm_half
        Vy[2,2] = 2*bc_val.y.N[2] - Vy[2,1]
    end

    # TODO: add BC for shear stress on sides
    ###########################################
    if tx[2,1] == :Neu_tang_half  # South
        Vx[2,1] = Vx[2,2] #- Δ.y*bc_val.D[1,2] 
    elseif tx[2,1] == :Dir_tang_half
        Vx[2,1] = 2*bc_val.x.S[1] - Vx[2,2]
    elseif tx[2,1] ==:Neu_tang_conf
        # 2*bc_val.D[1,2] = ∂Vx∂yN + (Vx[2,2] - Vx[2,1])/Δ.y
        ∂Vx∂yN  = (Vx[2,3] - Vx[2,2])/Δ.y
        # Vx[2,1] = (∂Vx∂yN - 2*bc_val.D[1,2])*Δ.y + Vx[2,2]
        # 2*bc_val.D[1,2] = ∂Vx∂yN + (Vx[2,2] - Vx[2,1])/Δ.y
        # (2*bc_val.D[1,2] - ∂Vx∂yN)*Δ.y =  Vx[2,2] - Vx[2,1]
        Vx[2,1] = Vx[2,2] - (2*bc_val.D[1,2] - ∂Vx∂yN)*Δ.y
    end

    if tx[1,2] == :Neu_norm_half # West
        Vx[1,2] = Vx[2,2] - Δ.x*bc_val.D[1,1]
    elseif tx[1,2] == :Dir_norm_half
        Vx[1,2] = 2*bc_val.x.W[1] - Vx[2,2]
    end

    if tx[3,2] == :Neu_norm_half # East
        Vx[3,2] = Vx[2,2] + Δ.x*bc_val.D[1,1]
    elseif tx[3,2] == :Dir_norm_half
        Vx[3,2] = 2*bc_val.x.E[1] - Vx[2,2]
    end

    if tx[2,3] == :Neu_tang_half # North
        Vx[2,3] = Vx[2,2] #+ Δ.y*bc_val.D[1,2]
    elseif tx[2,3] == :Dir_tang_half 
        Vx[2,3] = 2*bc_val.x.N[1] - Vx[2,2]
    elseif tx[2,3] ==:Neu_tang_conf
        ∂Vx∂yS  = (Vx[2,2] - Vx[2,1])/Δ.y
        Vx[2,3] = (2*bc_val.D[1,2] - ∂Vx∂yS)*Δ.y + Vx[2,2]
        # 2*bc_val.D[1,2] = ∂Vx∂yS + (Vx[2,3] - Vx[2,2])/Δ.y
        # (2*bc_val.D[1,2] - ∂Vx∂yS)*Δ.y =  Vx[2,3] - Vx[2,2]
        # Vx[2,3] = (2*bc_val.D[1,2] - ∂Vx∂yS)*Δ.y + Vx[2,2]

    end

    ############################################

    if t̄y[2,1] == :Neu_norm_half # South
        V̄y[2,1] = V̄y[2,2] - Δ.y*bc_val.D[2,2] 
    elseif t̄y[2,1] == :Dir_norm_half
        V̄y[2,1] = 2*bc_val.y.S[1] - V̄y[2,2]
    end

    if t̄y[1,2] == :Neu_tang_half # West
        V̄y[1,2] = V̄y[2,2] - Δ.x*bc_val.D[2,1]
    elseif t̄y[1,2] == :Dir_tang_half
        V̄y[1,2] = 2*bc_val.y.W[1] - V̄y[2,2]
    end

    if t̄y[3,2] == :Neu_tang_half # East
        V̄y[3,2] = V̄y[2,2] + Δ.x*bc_val.D[2,1]
    elseif t̄y[3,2] == :Dir_tang_half
        V̄y[3,2] = 2*bc_val.y.E[1] - V̄y[2,2]
    end

    if t̄y[2,3] == :Neu_norm_half # North
        V̄y[2,3] = V̄y[2,2] + Δ.y*bc_val.D[2,2]
    elseif t̄y[2,3] == :Dir_norm_half 
        V̄y[2,3] = 2*bc_val.y.N[1] - V̄y[2,2]
    end

    ############################################
     
    Dxx = (Vx[2:end,:] - Vx[1:end-1,:]) * invΔx             # Static Arrays ???
    Dyy = (Vy[:,2:end] - Vy[:,1:end-1]) * invΔy             
    Dkk = Dxx[:,2:end-1] + Dyy
    Dxy = (Vx[:,2:end] - Vx[:,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,:] - Vy[1:end-1,:]) * invΔx 


    # if tx[2,1] == :Neu_tang_half 
    #     # @show "South tangent half grid 1"
    #     Dxy[:,1] .= 0.0
    # end

    #  if tx[2,1] == :Neu_tang_conf 
    #     # @show "South tangent conf grid 1"
    #     Dxy[:,1] .= -Dxy[:,2]
    #     #  @show Dxy
    # end

    D̄xx = (V̄x[2:end,:] - V̄x[1:end-1,:]) * invΔx             # Static Arrays ???
    D̄yy = (V̄y[:,2:end] - V̄y[:,1:end-1]) * invΔy             
    D̄kk = D̄xx + D̄yy[2:end-1,:]
    D̄xy = (V̄x[:,2:end] - V̄x[:,1:end-1]) * invΔy 
    D̄yx = (V̄y[2:end,:] - V̄y[1:end-1,:]) * invΔx 

    # if tx[2,3] == :Neu_tang_half
    #     println("Neu_tang_half")
    #     # println(Dxy)
    #     println(0.5*(Dxy[2,2] + Dxy[2,1]) )
    # end

    # if tx[2,3] == :Neu_tang_conf
    #     println("Neu_tang_conf")
    #     # printxy(Dxy)
    #     println(Dxy[2,2], Dxy[2,1] )
    #     println(0.5*(Dxy[2,2] + Dxy[2,1]) )
    # end
    # if tx[2,3] == :Neu_tang_conf

    ε̇xx = Dxx[:,2:end-1] - 1/3*Dkk
    ε̇yy = Dyy - 1/3*Dkk
    ε̇xy = 1/2 * ( Dxy[2:end-1,:] + Dyx )
    ε̇̄xx = D̄xx - 1/3*D̄kk
    ε̇̄yy = D̄yy[2:end-1,:] - 1/3*D̄kk
    ε̇̄xy = 1/2 * ( D̄xy + D̄yx[:,2:end-1] ) 

    if rheology == :powerlaw
        ε̇II = sqrt.(1/2*(ε̇xx.^2 .+ ε̇yy.^2) .+ ε̇̄xy.^2)
        ε̇̄II = sqrt.(1/2*(ε̇̄xx.^2 .+ ε̇̄yy.^2) .+ ε̇xy.^2)
        η  = materials.η0[phase] .* ε̇II.^(1 ./ materials.n[phase] .- 1.0 )
        η̄  = materials.η0[p̄hase] .* ε̇̄II.^(1 ./ materials.n[p̄hase] .- 1.0 )
        τxx = 2 * η .* ε̇xx
        τxy = 2 * η̄ .* ε̇xy
    end

    if rheology == :anisotropic
        D  = materials.D[phase] 
        D̄  = materials.D[p̄hase] 
        τxx = zeros(2,1)
        τxy = zeros(1,2)
        for ii=1:2
            τxx[ii,1] = D[ii][1,1] .* ε̇xx[ii] .+ D[ii][1,2] .* ε̇yy[ii] .+ D[ii][1,3] .* ε̇̄xy[ii]
            τxy[1,ii] = D̄[ii][3,1] .* ε̇̄xx[ii] .+ D̄[ii][3,2] .* ε̇̄yy[ii] .+ D̄[ii][3,3] .* ε̇xy[ii]       
        end
    end

    fx = 0
    fx  = (τxx[2,1] - τxx[1,1]) * invΔx 
    fx += (τxy[1,2] - τxy[1,1]) * invΔy 
    fx -= ( Pt[2,1] -  Pt[1,1]) * invΔx
    fx *= -1*Δ.x*Δ.y

    return fx
end

function Momentum_y(Vx, V̄x, Vy, V̄y, Pt, P̄t, phase, p̄hase, materials, tx, t̄x, ty, t̄y, bc_val, Δ)
    
    invΔx    = 1 / Δ.x
    invΔy    = 1 / Δ.y
   
    # TODO: add BC for shear stress on sides
    ###########################################
    if ty[2,1] == :Neu_norm_half # South
        Vy[2,1] = Vy[2,2] - Δ.y*bc_val.D[2,2]
    elseif ty[2,1] == :Dir_norm_half
        Vy[2,1] = 2*bc_val.y.S[1] - Vy[2,2]
    end

    if ty[1,2] == :Neu_tang_half # West
        Vy[1,2] = Vy[2,2] - Δ.x*bc_val.D[2,1] 
    elseif ty[1,2] == :Dir_tang_half
        Vy[1,2] =  2*bc_val.y.W[1] - Vy[2,2] 
    end

    if ty[3,2] == :Neu_tang_half # East
        Vy[3,2] = Vy[2,2] + Δ.x*bc_val.D[2,1] 
    elseif ty[3,2] == :Dir_tang_half
        Vy[3,2] = 2*bc_val.y.E[1] - Vy[2,2] 
    end
  
    if ty[2,end] == :Neu_norm_half # North
        Vy[2,end] = Vy[2,end-1] + Δ.y*bc_val.D[2,2] 
    elseif ty[2,end] == :Dir_norm_half 
        Vy[2,end] = 2*bc_val.y.N[1] - Vy[2,end-1]
    end

    ############################################

    if t̄x[2,1] == :Neu_tang_half # Shouth
        V̄x[2,1] = V̄x[2,2] - Δ.y*bc_val.D[1,2] 
    elseif t̄x[2,1] == :Dir_tang_half
        V̄x[2,1] = 2*bc_val.x.S[1] - V̄x[2,2]
    end

    if t̄x[1,2] == :Neu_norm_half # West
        V̄x[1,2] = V̄x[2,2] - Δ.x*bc_val.D[1,1] 
    elseif t̄x[1,2] == :Dir_norm_half
        V̄x[1,2] =  2*bc_val.x.W[1] - V̄x[2,2] 
    end

    if t̄x[3,2] == :Neu_norm_half # East
        V̄x[3,2] = V̄x[2,2] + Δ.x*bc_val.D[1,1] 
    elseif t̄x[3,2] == :Dir_norm_half
        V̄x[3,2] = 2*bc_val.x.E[1] - V̄x[2,2] 
    end

    if t̄x[2,3] == :Neu_tang_half # North
        V̄x[2,3] = V̄x[2,2] + Δ.y*bc_val.D[1,2] 
    elseif t̄x[2,3] == :Dir_tang_half 
        V̄x[2,3] = 2*bc_val.x.N[1] - V̄x[2,2]
    end

    ############################################

    D̄yy = (V̄y[:,2:end] - V̄y[:,1:end-1]) * invΔy             # Static Arrays ???
    D̄xx = (V̄x[2:end,:] - V̄x[1:end-1,:]) * invΔx             
    D̄kk = D̄xx[:,2:end-1] + D̄yy
    D̄xy = (V̄x[:,2:end] - V̄x[:,1:end-1]) * invΔy 
    D̄yx = (V̄y[2:end,:] - V̄y[1:end-1,:]) * invΔx 

    Dyy = (Vy[:,2:end] - Vy[:,1:end-1]) * invΔy             # Static Arrays ???
    Dxx = (Vx[2:end,:] - Vx[1:end-1,:]) * invΔx             
    Dkk = Dxx + Dyy[2:end-1,:]
    Dxy = (Vx[:,2:end] - Vx[:,1:end-1]) * invΔy 
    Dyx = (Vy[2:end,:] - Vy[1:end-1,:]) * invΔx 

    ε̇xx = Dxx            - 1/3*Dkk
    ε̇yy = Dyy[2:end-1,:] - 1/3*Dkk
    ε̇xy = 1/2 * ( Dxy + Dyx[:,2:end-1] ) 
    ε̇̄xx = D̄xx[:,2:end-1] - 1/3*D̄kk
    ε̇̄yy = D̄yy - 1/3*D̄kk
    ε̇̄xy = 1/2 * ( D̄xy[2:end-1,:] + D̄yx ) 

    if rheology == :powerlaw
    ε̇II = sqrt.(1/2*(ε̇xx.^2 .+ ε̇yy.^2) .+ ε̇̄xy.^2)
    ε̇̄II = sqrt.(1/2*(ε̇̄xx.^2 .+ ε̇̄yy.^2) .+ ε̇xy.^2)
    η  = materials.η0[phase] .* ε̇II.^(1 ./ materials.n[phase] .- 1.0 )
    η̄  = materials.η0[p̄hase] .* ε̇̄II.^(1 ./ materials.n[p̄hase] .- 1.0 )
    τyy = 2 * η .* ε̇yy
    τxy = 2 * η̄ .* ε̇xy
    end

    if rheology == :anisotropic
        D  = materials.D[phase] 
        D̄  = materials.D[p̄hase] 
        τyy = zeros(1,2)
        τxy = zeros(2,1)
        for ii=1:2
            τyy[1,ii] = D[ii][2,1] .* ε̇xx[ii] + D[ii][2,2] .* ε̇yy[ii] + D[ii][2,3] .* ε̇̄xy[ii]
            τxy[ii,1] = D̄[ii][3,1] .* ε̇̄xx[ii] + D̄[ii][3,2] .* ε̇̄yy[ii] + D̄[ii][3,3] .* ε̇xy[ii]
        end
    end
    fy  = 0 
    fy  = (τyy[1,2] - τyy[1,1]) * invΔy 
    fy += (τxy[2,1] - τxy[1,1]) * invΔx 
    fy -= ( Pt[1,2] -  Pt[1,1]) * invΔy
    fy *= -1*Δ.x*Δ.y

    return fy
end

function ResidualMomentum2D_1!(R, V, Pt, phases, materials, num, pattern, types, BC, nc, Δ) 
    for j in 2:size(V.x[1],2)-1, i in 2:size(V.x[1],1)-1
        Vx    = FSG_Array( MMatrix{3,3}(       V.x[1][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),      
                           MMatrix{2,2}(       V.x[2][ii,jj] for ii in i-1:i,   jj in j:j+1  )) 
        Vy    = FSG_Array( MMatrix{3,3}(       V.y[1][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),
                           MMatrix{2,2}(       V.y[2][ii,jj] for ii in i-1:i,   jj in j:j+1  )) 
        typex = FSG_Array( SMatrix{3,3}(  types.Vx[1][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),
                           SMatrix{2,2}(  types.Vy[2][ii,jj] for ii in i-1:i,   jj in j:j+1  ))
        typey = FSG_Array( SMatrix{3,3}(  types.Vy[1][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),
                           SMatrix{2,2}(  types.Vy[2][ii,jj] for ii in i-1:i,   jj in j:j+1  ))
        P     = FSG_Array( MMatrix{2,1}(        Pt[1][ii,jj] for ii in i-1:i,   jj in j:j  ),      
                           MMatrix{1,2}(        Pt[2][ii,jj] for ii in i-1:i-1, jj in j-1:j))
        phase = FSG_Array( SMatrix{2,1}(    phases[1][ii,jj] for ii in i-1:i,   jj in j:j  ),      
                           SMatrix{1,2}(    phases[2][ii,jj] for ii in i-1:i-1, jj in j-1:j))
        bcx = (
            W  = SMatrix{1,2}(   BC.W.Vx[jj] for jj in j-1:j),
            E  = SMatrix{1,2}(   BC.E.Vx[jj] for jj in j-1:j),
            S  = SMatrix{1,1}(   BC.S.Vx[ii] for ii in i-1:i-1),
            N  = SMatrix{1,1}(   BC.N.Vx[ii] for ii in i-1:i-1),
        )
        bcy = (
            W  = SMatrix{1,2}(   BC.W.Vy[jj] for jj in j-1:j),
            E  = SMatrix{1,2}(   BC.E.Vy[jj] for jj in j-1:j),
            S  = SMatrix{1,1}(   BC.S.Vy[ii] for ii in i-1:i-1),
            N  = SMatrix{1,1}(   BC.N.Vy[ii] for ii in i-1:i-1),
        )
        bc_val = (x=bcx, y=bcy, D=BC.W.D)

        if types.Vx[1][i,j] == :in
            R.x[1][i,j]     = Momentum_x(Vx[1], Vx[2], Vy[2], Vy[1], P[1], P[2], phase[1], phase[2], materials, typex[1], typex[2], typey[2], typey[1], bc_val, Δ)
        end

        if types.Vy[1][i,j] == :in
            R.y[1][i,j]     = Momentum_y(Vx[2], Vx[1], Vy[1], Vy[2], P[2], P[1], phase[2], phase[1], materials, typex[2], typex[1], typey[1], typey[2], bc_val, Δ)
        end

    end
end

function ResidualMomentum2D_2!(R, V, Pt, phases, materials, num, pattern, types, BC, nc, Δ) 
    for j in 2:size(V.x[2],2)-1, i in 2:size(V.x[2],1)-1
        Vx    = FSG_Array( MMatrix{2,2}(       V.x[1][ii,jj] for ii in i:i+1,   jj in j-1:j  ),
                           MMatrix{3,3}(       V.x[2][ii,jj] for ii in i-1:i+1, jj in j-1:j+1)) 
        Vy    = FSG_Array( MMatrix{2,2}(       V.y[1][ii,jj] for ii in i:i+1,   jj in j-1:j  ),
                           MMatrix{3,3}(       V.y[2][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),) 
        typex = FSG_Array( SMatrix{2,2}(  types.Vy[1][ii,jj] for ii in i:i+1,   jj in j-1:j  ),
                           SMatrix{3,3}(  types.Vx[2][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),)
        typey = FSG_Array( SMatrix{2,2}(  types.Vy[1][ii,jj] for ii in i:i+1,   jj in j-1:j  ),
                           SMatrix{3,3}(  types.Vy[2][ii,jj] for ii in i-1:i+1, jj in j-1:j+1))
        P     = FSG_Array( MMatrix{1,2}(        Pt[1][ii,jj] for ii in i:i,     jj in j-1:j  ),      
                           MMatrix{2,1}(        Pt[2][ii,jj] for ii in i-1:i,   jj in j-1:j-1))
        phase = FSG_Array( SMatrix{1,2}(    phases[1][ii,jj] for ii in i:i,     jj in j-1:j  ),      
                           SMatrix{2,1}(    phases[2][ii,jj] for ii in i-1:i,   jj in j-1:j-1))

        bcx = (
            W  = SMatrix{1,1}(   BC.W.Vx[jj] for jj in j-1:j-1),
            E  = SMatrix{1,1}(   BC.E.Vx[jj] for jj in j-1:j-1),
            S  = SMatrix{1,2}(   BC.S.Vx[ii] for ii in i-1:i  ),
            N  = SMatrix{1,2}(   BC.N.Vx[ii] for ii in i-1:i  ),
        )
        bcy = (
            W  = SMatrix{1,1}(   BC.W.Vy[jj] for jj in j-1:j-1),
            E  = SMatrix{1,1}(   BC.E.Vy[jj] for jj in j-1:j-1),
            S  = SMatrix{1,2}(   BC.S.Vy[ii] for ii in i-1:i  ),
            N  = SMatrix{1,2}(   BC.N.Vy[ii] for ii in i-1:i  ),
        )
        bc_val = (x=bcx, y=bcy, D=BC.W.D)

        if types.Vx[2][i,j] == :in
            R.x[2][i,j]     = Momentum_x(Vx[2], Vx[1], Vy[1], Vy[2], P[2], P[1], phase[2], phase[1], materials, typex[2], typex[1], typey[1], typey[2], bc_val, Δ)
        end

        if types.Vy[2][i,j] == :in
            R.y[2][i,j]     = Momentum_y(Vx[1], Vx[2], Vy[2], Vy[1], P[1], P[2], phase[1], phase[2], materials, typex[1], typex[2], typey[2], typey[1], bc_val, Δ)
        end

    end
end


function AssembleMomentum2D_1!(K, V, Pt, phases, materials, num, pattern, types, BC, nc, Δ) 

    ∂Rx∂Vx1 = @MMatrix ones(3,3)
    ∂Rx∂Vx2 = @MMatrix ones(2,2)
    ∂Rx∂Vy1 = @MMatrix ones(3,3)
    ∂Rx∂Vy2 = @MMatrix ones(2,2)
    ∂Rx∂Pt1 = @MMatrix ones(2,1)
    ∂Rx∂Pt2 = @MMatrix ones(1,2)

    ∂Ry∂Vx1 = @MMatrix ones(3,3)
    ∂Ry∂Vx2 = @MMatrix ones(2,2)
    ∂Ry∂Vy1 = @MMatrix ones(3,3)
    ∂Ry∂Vy2 = @MMatrix ones(2,2)
    ∂Ry∂Pt1 = @MMatrix ones(2,1)
    ∂Ry∂Pt2 = @MMatrix ones(1,2)

    for j in 2:size(V.x[1],2)-1, i in 2:size(V.x[1],1)-1
        Vx    = FSG_Array( MMatrix{3,3}(       V.x[1][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),      
        MMatrix{2,2}(       V.x[2][ii,jj] for ii in i-1:i,   jj in j:j+1  )) 
        Vy    = FSG_Array( MMatrix{3,3}(       V.y[1][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),
                MMatrix{2,2}(       V.y[2][ii,jj] for ii in i-1:i,   jj in j:j+1  )) 
        typex = FSG_Array( SMatrix{3,3}(  types.Vx[1][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),
                SMatrix{2,2}(  types.Vy[2][ii,jj] for ii in i-1:i,   jj in j:j+1  ))
        typey = FSG_Array( SMatrix{3,3}(  types.Vy[1][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),
                SMatrix{2,2}(  types.Vy[2][ii,jj] for ii in i-1:i,   jj in j:j+1  ))
        P     = FSG_Array( MMatrix{2,1}(        Pt[1][ii,jj] for ii in i-1:i,   jj in j:j  ),      
                MMatrix{1,2}(        Pt[2][ii,jj] for ii in i-1:i-1, jj in j-1:j))
        phase = FSG_Array( SMatrix{2,1}(    phases[1][ii,jj] for ii in i-1:i,   jj in j:j  ),      
                SMatrix{1,2}(    phases[2][ii,jj] for ii in i-1:i-1, jj in j-1:j))
        bcx = (
        W  = SMatrix{1,2}(   BC.W.Vx[jj] for jj in j-1:j),
        E  = SMatrix{1,2}(   BC.E.Vx[jj] for jj in j-1:j),
        S  = SMatrix{1,1}(   BC.S.Vx[ii] for ii in i-1:i-1),
        N  = SMatrix{1,1}(   BC.N.Vx[ii] for ii in i-1:i-1),
        )
        bcy = (
        W  = SMatrix{1,2}(   BC.W.Vy[jj] for jj in j-1:j),
        E  = SMatrix{1,2}(   BC.E.Vy[jj] for jj in j-1:j),
        S  = SMatrix{1,1}(   BC.S.Vy[ii] for ii in i-1:i-1),
        N  = SMatrix{1,1}(   BC.N.Vy[ii] for ii in i-1:i-1),
        )
        bc_val = (x=bcx, y=bcy, D=BC.W.D)

        if types.Vx[1][i,j] == :in
            ieq_x = num.Vx[1][i,j]
            ∂Rx∂Vx1 .= 0.
            ∂Rx∂Vx2 .= 0.
            ∂Rx∂Vy1 .= 0.
            ∂Rx∂Vy2 .= 0.
            ∂Rx∂Pt1 .= 0.
            ∂Rx∂Pt2 .= 0.
            forwarddiff_gradients!(Momentum_x, Duplicated(Vx[1], ∂Rx∂Vx1), Duplicated(Vx[2], ∂Rx∂Vx2), Duplicated(Vy[2], ∂Rx∂Vy2), Duplicated(Vy[1], ∂Rx∂Vy1), Duplicated(P[1], ∂Rx∂Pt1), Duplicated(P[2], ∂Rx∂Pt2), Const(phase[1]), Const(phase[2]), Const(materials), Const(typex[1]), Const(typex[2]), Const(typey[2]), Const(typey[1]), Const(bc_val), Const(Δ))
            # ∂Rx∂Pt1 .= 1.
            # ∂Rx∂Pt2 .= 0.

            ##################################################################
            # Vx1 --> Vx1, Vy1
            Local_xx = num.Vx[1][i-1:i+1,j-1:j+1] .* pattern.Vx.Vx[1][1]
            Local_xy = num.Vy[1][i-1:i+1,j-1:j+1] .* pattern.Vx.Vy[1][1]
            for jj in axes(Local_xx,2), ii in axes(Local_xx,1)
                if (Local_xx[ii,jj]>0)
                    K.Vx.Vx[1][1][ieq_x, Local_xx[ii,jj]] = ∂Rx∂Vx1[ii,jj] 
                    K.Vx.Vy[1][1][ieq_x, Local_xy[ii,jj]] = ∂Rx∂Vy1[ii,jj] 
                end
            end
            ##################################################################
            # Vx1 --> Vx2, Vy2
            Local_xx = num.Vx[2][i-1:i,j:j+1] .* pattern.Vx.Vx[1][2]
            Local_xy = num.Vy[2][i-1:i,j:j+1] .* pattern.Vx.Vy[1][2]
            for jj in axes(Local_xx,2), ii in axes(Local_xx,1)
                if (Local_xx[ii,jj]>0)
                    K.Vx.Vx[1][2][ieq_x, Local_xx[ii,jj]] = ∂Rx∂Vx2[ii,jj]
                end
                if (Local_xy[ii,jj]>0)
                    K.Vx.Vy[1][2][ieq_x, Local_xy[ii,jj]] = ∂Rx∂Vy2[ii,jj] 
                end
            end
            ##################################################################
            # Vx1 --> P1
            Local = num.Pt[1][i-1:i,j:j] .* pattern.Vx.Pt[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0)
                    K.Vx.Pt[1][1][ieq_x, Local[ii,jj]] = ∂Rx∂Pt1[ii,jj]  
                end
            end
            ##################################################################
            # Vx1 --> P2
            Local = num.Pt[2][i-1:i-1,j-1:j] .* pattern.Vx.Pt[1][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) 
                    K.Vx.Pt[1][2][ieq_x, Local[ii,jj]] = ∂Rx∂Pt2[ii,jj]  
                end
            end
        end

        if types.Vy[1][i,j] == :in
            ieq_y = num.Vy[1][i,j]
            ∂Ry∂Vx1 .= 0.
            ∂Ry∂Vx2 .= 0.
            ∂Ry∂Vy1 .= 0.
            ∂Ry∂Vy2 .= 0.
            ∂Ry∂Pt1 .= 0.
            ∂Ry∂Pt2 .= 0.
            forwarddiff_gradients!(Momentum_y, Duplicated(Vx[2], ∂Ry∂Vx2), Duplicated(Vx[1], ∂Ry∂Vx1), Duplicated(Vy[1], ∂Ry∂Vy1), Duplicated(Vy[2], ∂Ry∂Vy2), Duplicated(P[2], ∂Ry∂Pt2), Duplicated(P[1], ∂Ry∂Pt1), Const(phase[2]), Const(phase[1]), Const(materials), Const(typex[2]), Const(typex[1]), Const(typey[1]), Const(typey[2]), Const(bc_val), Const(Δ))            ##################################################################
            # ∂Ry∂Pt1 .= 0.
            # ∂Ry∂Pt2 .= 0.

            # Vy1 --> Vx1, Vy1
            Local_yx = num.Vx[1][i-1:i+1,j-1:j+1] .* pattern.Vy.Vx[1][1]
            Local_yy = num.Vy[1][i-1:i+1,j-1:j+1] .* pattern.Vy.Vy[1][1]
            for jj in axes(Local_yy,2), ii in axes(Local_yy,1)
                if (Local_yy[ii,jj]>0) 
                    K.Vy.Vy[1][1][ieq_y, Local_yy[ii,jj]] = ∂Ry∂Vy1[ii,jj] 
                    K.Vy.Vx[1][1][ieq_y, Local_yx[ii,jj]] = ∂Ry∂Vx1[ii,jj] 
                end
            end
            ##################################################################
            # Vy1 --> Vx2, Vy2
            Local_yx = num.Vx[2][i-1:i,j:j+1] .* pattern.Vy.Vx[1][2]
            Local_yy = num.Vy[2][i-1:i,j:j+1] .* pattern.Vy.Vy[1][2]
            for jj in axes(Local_yy,2), ii in axes(Local_yy,1)
                if (Local_yx[ii,jj]>0)
                    K.Vy.Vx[1][2][ieq_y, Local_yx[ii,jj]] = ∂Ry∂Vx2[ii,jj] 
                end
                if (Local_yy[ii,jj]>0)
                    K.Vy.Vy[1][2][ieq_y, Local_yy[ii,jj]] = ∂Ry∂Vy2[ii,jj] 
                end
            end
            ##################################################################
            # Vy1 --- P1
            Local = num.Pt[1][i-1:i,j:j] .* pattern.Vy.Pt[1][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0)
                    K.Vy.Pt[1][1][ieq_y, Local[ii,jj]] = ∂Ry∂Pt1[ii,jj]  
                end
            end
            ##################################################################
            # Vy1 --> P2
            Local = num.Pt[2][i-1:i-1,j-1:j] .* pattern.Vy.Pt[1][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) 
                    K.Vy.Pt[1][2][ieq_y, Local[ii,jj]] = ∂Ry∂Pt2[ii,jj]  
                end
            end
        end
    end
    return nothing
end

function AssembleMomentum2D_2!(K, V, Pt, phases, materials, num, pattern, types, BC, nc, Δ) 

    ∂Rx∂Vx2 = @MMatrix ones(3,3)
    ∂Rx∂Vx1 = @MMatrix ones(2,2)
    ∂Rx∂Vy2 = @MMatrix ones(3,3)
    ∂Rx∂Vy1 = @MMatrix ones(2,2)
    ∂Rx∂Pt2 = @MMatrix ones(2,1)
    ∂Rx∂Pt1 = @MMatrix ones(1,2)

    ∂Ry∂Vx2 = @MMatrix ones(3,3)
    ∂Ry∂Vx1 = @MMatrix ones(2,2)
    ∂Ry∂Vy2 = @MMatrix ones(3,3)
    ∂Ry∂Vy1 = @MMatrix ones(2,2)
    ∂Ry∂Pt2 = @MMatrix ones(2,1)
    ∂Ry∂Pt1 = @MMatrix ones(1,2)
    
    for j in 2:size(V.x[2],2)-1, i in 2:size(V.x[2],1)-1
        
        Vx    = FSG_Array( MMatrix{2,2}(       V.x[1][ii,jj] for ii in i:i+1,   jj in j-1:j  ),
                           MMatrix{3,3}(       V.x[2][ii,jj] for ii in i-1:i+1, jj in j-1:j+1)) 
        Vy    = FSG_Array( MMatrix{2,2}(       V.y[1][ii,jj] for ii in i:i+1,   jj in j-1:j  ),
                           MMatrix{3,3}(       V.y[2][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),) 
        typex = FSG_Array( SMatrix{2,2}(  types.Vy[1][ii,jj] for ii in i:i+1,   jj in j-1:j  ),
                           SMatrix{3,3}(  types.Vx[2][ii,jj] for ii in i-1:i+1, jj in j-1:j+1),)
        typey = FSG_Array( SMatrix{2,2}(  types.Vy[1][ii,jj] for ii in i:i+1,   jj in j-1:j  ),
                           SMatrix{3,3}(  types.Vy[2][ii,jj] for ii in i-1:i+1, jj in j-1:j+1))
        P     = FSG_Array( MMatrix{1,2}(        Pt[1][ii,jj] for ii in i:i,     jj in j-1:j  ),      
                           MMatrix{2,1}(        Pt[2][ii,jj] for ii in i-1:i,   jj in j-1:j-1))
        phase = FSG_Array( SMatrix{1,2}(    phases[1][ii,jj] for ii in i:i,     jj in j-1:j  ),      
                           SMatrix{2,1}(    phases[2][ii,jj] for ii in i-1:i,   jj in j-1:j-1))
        bcx = (
        W  = SMatrix{1,1}(   BC.W.Vx[jj] for jj in j-1:j-1),
        E  = SMatrix{1,1}(   BC.E.Vx[jj] for jj in j-1:j-1),
        S  = SMatrix{1,2}(   BC.S.Vx[ii] for ii in i-1:i  ),
        N  = SMatrix{1,2}(   BC.N.Vx[ii] for ii in i-1:i  ),
        )
        bcy = (
        W  = SMatrix{1,1}(   BC.W.Vy[jj] for jj in j-1:j-1),
        E  = SMatrix{1,1}(   BC.E.Vy[jj] for jj in j-1:j-1),
        S  = SMatrix{1,2}(   BC.S.Vy[ii] for ii in i-1:i  ),
        N  = SMatrix{1,2}(   BC.N.Vy[ii] for ii in i-1:i  ),
        )
        bc_val = (x=bcx, y=bcy, D=BC.W.D)

        if types.Vx[2][i,j] == :in
            ieq_x = num.Vx[2][i,j]
            ∂Rx∂Vx1 .= 0.
            ∂Rx∂Vx2 .= 0.
            ∂Rx∂Vy1 .= 0.
            ∂Rx∂Vy2 .= 0.
            ∂Rx∂Pt1 .= 0.
            ∂Rx∂Pt2 .= 0.
            forwarddiff_gradients!(Momentum_x, Duplicated(Vx[2], ∂Rx∂Vx2), Duplicated(Vx[1], ∂Rx∂Vx1), Duplicated(Vy[1], ∂Rx∂Vy1), Duplicated(Vy[2], ∂Rx∂Vy2), Duplicated(P[2], ∂Rx∂Pt2), Duplicated(P[1], ∂Rx∂Pt1), Const(phase[2]), Const(phase[1]), Const(materials), Const(typex[2]), Const(typex[1]), Const(typey[1]), Const(typey[2]), Const(bc_val), Const(Δ))

            ##################################################################
            # Vx2 --> Vx2, Vy2
            Local_xx = num.Vx[2][i-1:i+1,j-1:j+1] .* pattern.Vx.Vx[2][2]
            Local_xy = num.Vy[2][i-1:i+1,j-1:j+1] .* pattern.Vx.Vy[2][2]
            for jj in axes(Local_xx,2), ii in axes(Local_xx,1)
                if (Local_xx[ii,jj]>0)
                    K.Vx.Vx[2][2][ieq_x, Local_xx[ii,jj]] = ∂Rx∂Vx2[ii,jj] 
                end
                if (Local_xy[ii,jj]>0)
                    K.Vx.Vy[2][2][ieq_x, Local_xy[ii,jj]] = ∂Rx∂Vy2[ii,jj] 
                end
            end  
            ##################################################################
            # Vx2 --> Vx1, Vy1
            Local_xx = num.Vx[1][i:i+1,j-1:j] .* pattern.Vx.Vx[2][1]
            Local_xy = num.Vy[1][i:i+1,j-1:j] .* pattern.Vx.Vy[2][1]
            for jj in axes(Local_xx,2), ii in axes(Local_xx,1)
                if (Local_xx[ii,jj]>0)
                    K.Vx.Vx[2][1][ieq_x, Local_xx[ii,jj]] = ∂Rx∂Vx1[ii,jj]
                end
                if (Local_xy[ii,jj]>0)
                    K.Vx.Vy[2][1][ieq_x, Local_xy[ii,jj]] = ∂Rx∂Vy1[ii,jj] 
                end
            end
            ##################################################################
            # Vx2 --> P1
            Local = num.Pt[1][i:i,j-1:j] .* pattern.Vx.Pt[2][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0)
                    K.Vx.Pt[2][1][ieq_x, Local[ii,jj]] = ∂Rx∂Pt1[ii,jj]  
                end
            end            
            ##################################################################
            # Vx2 --> P2
            Local = num.Pt[2][i-1:i,j-1:j-1] .* pattern.Vx.Pt[2][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) 
                    K.Vx.Pt[2][2][ieq_x, Local[ii,jj]] = ∂Rx∂Pt2[ii,jj]  
                end
            end
        end

        if types.Vy[2][i,j] == :in
            ieq_y = num.Vy[2][i,j]
            ∂Ry∂Vx1 .= 0.
            ∂Ry∂Vx2 .= 0.
            ∂Ry∂Vy1 .= 0.
            ∂Ry∂Vy2 .= 0.
            ∂Ry∂Pt1 .= 0.
            ∂Ry∂Pt2 .= 0.
            forwarddiff_gradients!(Momentum_y, Duplicated(Vx[1], ∂Ry∂Vx1), Duplicated(Vx[2], ∂Ry∂Vx2), Duplicated(Vy[2], ∂Ry∂Vy2), Duplicated(Vy[1], ∂Ry∂Vy1), Duplicated(P[1], ∂Ry∂Pt1), Duplicated(P[2], ∂Ry∂Pt2), Const(phase[1]), Const(phase[2]), Const(materials), Const(typex[1]), Const(typex[2]), Const(typey[2]), Const(typey[1]), Const(bc_val), Const(Δ))
    
            ##################################################################
            # Vy1 --> Vy1, Vx1
            Local_yy = num.Vy[2][i-1:i+1,j-1:j+1] .* pattern.Vy.Vy[2][2]
            Local_yx = num.Vx[2][i-1:i+1,j-1:j+1] .* pattern.Vy.Vx[2][2]
            for jj in axes(Local_yy,2), ii in axes(Local_yy,1)
                if (Local_yy[ii,jj]>0)
                    K.Vy.Vy[2][2][ieq_y, Local_yy[ii,jj]] = ∂Ry∂Vy2[ii,jj] 
                end
                if (Local_yx[ii,jj]>0)
                    K.Vy.Vx[2][2][ieq_y, Local_yx[ii,jj]] = ∂Ry∂Vx2[ii,jj] 
                end
            end
            ##################################################################
            # Vy2 --> Vx1, Vy1
            Local_yx = num.Vx[1][i:i+1,j-1:j] .* pattern.Vy.Vx[2][1]
            Local_yy = num.Vy[1][i:i+1,j-1:j] .* pattern.Vy.Vy[2][1]
            for jj in axes(Local_yy,2), ii in axes(Local_yy,1)
                if (Local_yx[ii,jj]>0)
                    K.Vy.Vx[2][1][ieq_y, Local_yx[ii,jj]] = ∂Ry∂Vx1[ii,jj] 
                end
                if (Local_yy[ii,jj]>0)
                    K.Vy.Vy[2][1][ieq_y, Local_yy[ii,jj]] = ∂Ry∂Vy1[ii,jj] 
                end
            end
            ##################################################################
            # Vy2 --> P1
            Local = num.Pt[1][i:i,j-1:j] .* pattern.Vy.Pt[2][1]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0)
                    K.Vy.Pt[2][1][ieq_y, Local[ii,jj]] = ∂Ry∂Pt1[ii,jj]  
                end
            end
            ##################################################################
            # Vy2 --> P2
            Local = num.Pt[2][i-1:i,j-1:j-1] .* pattern.Vy.Pt[2][2]
            for jj in axes(Local,2), ii in axes(Local,1)
                if (Local[ii,jj]>0) 
                    K.Vy.Pt[2][2][ieq_y, Local[ii,jj]] = ∂Ry∂Pt2[ii,jj]  
                end
            end
        end
    end
    return nothing
end

function main(nc) 
    #--------------------------------------------#
    # Resolution

    inx_V  = FSG_Array( 2:nc.x+2, 2:nc.x+1 )
    iny_V  = FSG_Array( 2:nc.y+1, 2:nc.y+2 )
    inx_P  = FSG_Array( 2:nc.x+1, 1:nc.x+1 )
    iny_P  = FSG_Array( 2:nc.y+1, 1:nc.y+1 )
    size_V = FSG_Array( (nc.x+3, nc.y+2), (nc.x+2, nc.y+3))
    size_P = FSG_Array( (nc.x+2, nc.y+2), (nc.x+1, nc.y+1))

    #--------------------------------------------#
    # Boundary conditions

    # Define node types and set BC flags
    type = Fields(
        FSG_Array( fill(:out, size_V[1]), fill(:out, size_V[2]) ),
        FSG_Array( fill(:out, size_V[1]), fill(:out, size_V[2]) ),
        FSG_Array( fill(:out, size_P[1]), fill(:out, size_P[2]) ),
    )

    # ----- Inside ----- #
    type.Vx[1][inx_V[1],iny_V[1]] .= :in
    type.Vx[2][inx_V[2],iny_V[2]] .= :in       
    type.Vy[1][inx_V[1],iny_V[1]] .= :in
    type.Vy[2][inx_V[2],iny_V[2]] .= :in  
    # ----- West ----- #   
    type.Vx[1][2,iny_V[1]]        .= :Dirichlet_normal 
    type.Vx[2][1,iny_V[2]]        .= :Dir_norm_half
    type.Vy[1][2,iny_V[1]]        .= :Dirichlet_normal 
    type.Vy[2][1,iny_V[2]]        .= :Dir_tang_half
    # ----- East ----- # 
    type.Vx[1][end-1,iny_V[1]]    .= :Dirichlet_normal 
    type.Vx[2][end-0,iny_V[2]]    .= :Dir_norm_half
    type.Vy[1][end-1,iny_V[1]]    .= :Dirichlet_normal 
    type.Vy[2][end-0,iny_V[2]]    .= :Dir_tang_half
    # ----- South ----- #
    type.Vx[1][inx_V[1],1]        .= :Dir_tang_half
    # type.Vx[1][inx_V[1],1]        .= :Neu_tang_half
    type.Vx[2][inx_V[2],2]        .= :Dirichlet_normal
    # type.Vx[2][inx_V[2],1]        .= :Neu_tang_conf
    
    type.Vy[1][inx_V[1],1]        .= :Dir_norm_half
    type.Vy[2][inx_V[2],2]        .= :Dirichlet_normal 
    # ----- North ----- #
    type.Vx[1][inx_V[1],end]      .= :Dir_tang_half
    # type.Vx[1][inx_V[1],end]      .= :Neu_tang_half
    type.Vx[2][inx_V[2],end-1]    .= :Dirichlet_normal 
    # type.Vx[2][inx_V[2],end-0]    .= :Neu_tang_conf 

    type.Vy[1][inx_V[1],end]      .= :Dir_norm_half
    type.Vy[2][inx_V[2],end-1]    .= :Dirichlet_normal 
    # -------- Pt -------- #
    type.Pt[1][inx_P[1],iny_P[1]] .= :in
    type.Pt[2]                    .= :Dirichlet_normal
    type.Pt[2][2:end-1,2:end-0]   .= :in

    #--------------------------------------------#
    # Equation numbering
    number = Fields(
        FSG_Array( fill(0, size_V[1]), fill(0, size_V[2]) ),
        FSG_Array( fill(0, size_V[1]), fill(0, size_V[2]) ),
        FSG_Array( fill(0, size_P[1]), fill(0, size_P[2]) ),
    )
    Numbering!(number, type, nc)

    # #--------------------------------------------#
    # Stencil extent for each block matrix
    pattern = Patterns()

    ################################
    nVx   = [maximum(number.Vx[1]) maximum(number.Vx[2])]
    nVy   = [maximum(number.Vy[1]) maximum(number.Vy[2])]
    nPt   = [maximum(number.Pt[1]) maximum(number.Pt[2])]

    # Sparse matrix assembly
    M = AllocateSparseMatrix(number)

    # Intialise field
    L   = (x=1.0, y=1.0)
    Δ   = (x=L.x/nc.x, y=L.y/nc.y)
    R   = (x = FSG_Array(zeros(size_V[1]...), zeros(size_V[2]...)), 
           y = FSG_Array(zeros(size_V[1]...), zeros(size_V[2]...)),
           p = FSG_Array(zeros(size_P[1]...), zeros(size_P[2]...)))
    V   = (x = FSG_Array(zeros(size_V[1]...), zeros(size_V[2]...)), 
           y = FSG_Array(zeros(size_V[1]...), zeros(size_V[2]...)))
    Pt  = FSG_Array(zeros(size_P[1]...), zeros(size_P[2]...))
    phases = FSG_Array(ones(Int64, size_P[1]...), ones(Int64, size_P[2]...))

    # Materials
    θ  = 10
    N  = [sind(θ) cosd(θ)]
    η0 = [1e0 1e2]
    δ  = [1.0 1]
    D1 = ViscosityTensor(η0[1], δ[1], N, false)
    D2 = ViscosityTensor(η0[2], δ[2], N, false)

    materials = ( 
        n  = [20 1.0],
        η0 = [1e0 1e2],
        D  = [D1, D2], 
    )

    # Pure Shear
    D_BC = [-1  0;
             0  1]
    # D_BC = [0  1;
    #         0  0]
    xv  = LinRange(-L.x/2, L.x/2, nc.x+1)
    yv  = LinRange(-L.y/2, L.y/2, nc.y+1)
    xc  = LinRange(-L.x/2-Δ.x/2, L.x/2+Δ.x/2, nc.x+2)
    yc  = LinRange(-L.y/2-Δ.y/2, L.y/2+Δ.y/2, nc.y+2)

    phases[1][xc.^2 .+ (yc').^2 .< 0.1^2] .= 2 
    phases[2][xv.^2 .+ (yv').^2 .< 0.1^2] .= 2

    VxHR  = zeros(2*nc.x+1, 2*nc.y+1)
    VyHR  = zeros(2*nc.x+1, 2*nc.y+1)
    xHR   = LinRange(-L.x/2, L.x/2, 2*nc.x+1)
    yHR   = LinRange(-L.y/2, L.y/2, 2*nc.y+1)
    VxHR .= D_BC[1,1]*xHR .+ D_BC[1,2]*yHR' 
    VyHR .= D_BC[2,1]*xHR .+ D_BC[2,2]*yHR'

    V.x[1][inx_V[1],iny_V[1]] .= VxHR[1:2:end-0, 2:2:end-1]
    V.y[1][inx_V[1],iny_V[1]] .= VyHR[1:2:end-0, 2:2:end-1]
    V.x[2][inx_V[2],iny_V[2]] .= VxHR[2:2:end-1, 1:2:end-0]
    V.y[2][inx_V[2],iny_V[2]] .= VyHR[2:2:end-1, 1:2:end-0]

    BC = (
        W = (
            Vx    = VxHR[1, 1:2:end],
            Vy    = VyHR[1, 1:2:end],
            D     = D_BC
        ),
        E = (
            Vx    = VxHR[end, 1:2:end],
            Vy    = VyHR[end, 1:2:end],
            D     = D_BC
        ),
        S = (
            Vx    = VxHR[1:2:end, 1],
            Vy    = VyHR[1:2:end, 1],
            D     = D_BC
        ),
        N = (
            Vx    = VxHR[1:2:end, end],
            Vy    = VyHR[1:2:end, end],
            D     = D_BC
        )       
    )

    # Newton solver
    niter_nl = 2
    niter_l  = 10
    ϵ_nl     = 1e-6
    ϵ_l      = 1e-10
    ηb       = 1e3

    err = Fields(
        FSG_Array( zeros(niter_nl), zeros(niter_nl) ),
        FSG_Array( zeros(niter_nl), zeros(niter_nl) ),
        FSG_Array( zeros(niter_nl), zeros(niter_nl) ),
    )
    
    for iter=1:niter_nl
        @info "iteration $(iter)"
        ResidualContinuity2D_1!(R, V, Pt, phases, materials, number, type, BC, nc, Δ) 
        ResidualContinuity2D_2!(R, V, Pt, phases, materials, number, type, BC, nc, Δ) 
        ResidualMomentum2D_1!(R, V, Pt, phases, materials, number, pattern, type, BC, nc, Δ) 
        ResidualMomentum2D_2!(R, V, Pt, phases, materials, number, pattern, type, BC, nc, Δ) 

        @show norm(R.x[1])
        @show norm(R.x[2])
        @show norm(R.y[1])
        @show norm(R.y[2])
        @show norm(R.p[1])
        @show norm(R.p[2])

        max_all(R) < ϵ_nl ? break : nothing 

        err.Vx[1][iter] = norm(R.x[1][inx_V[1],iny_V[1]])/sqrt(nVx[1])
        err.Vy[1][iter] = norm(R.y[1][inx_V[2],iny_V[2]])/sqrt(nVy[1])
        err.Pt[1][iter] = norm(R.p[1][inx_P[1],iny_P[1]])/sqrt(nPt[1])
        err.Vx[2][iter] = norm(R.x[2][inx_V[2],iny_V[2]])/sqrt(nVx[2])
        err.Vy[2][iter] = norm(R.y[2][inx_V[1],iny_V[1]])/sqrt(nVy[2])
        err.Pt[2][iter] = norm(R.p[2][inx_P[2],iny_P[2]])/sqrt(nPt[2])

        AssembleMomentum2D_1!(M, V, Pt, phases, materials, number, pattern, type, BC, nc, Δ) 
        AssembleMomentum2D_2!(M, V, Pt, phases, materials, number, pattern, type, BC, nc, Δ) 
        AssembleContinuity2D_1!(M, V, Pt, phases, materials, number, pattern, type, BC, nc, Δ) 
        AssembleContinuity2D_2!(M, V, Pt, phases, materials, number, pattern, type, BC, nc, Δ)

        # Stokes operator as block matrices
        𝐊  = [M.Vx.Vx[1][1] M.Vx.Vx[1][2] M.Vx.Vy[1][1] M.Vx.Vy[1][2]; 
            M.Vx.Vx[2][1] M.Vx.Vx[2][2] M.Vx.Vy[2][1] M.Vx.Vy[2][2];
            M.Vy.Vx[1][1] M.Vy.Vx[1][2] M.Vy.Vy[1][1] M.Vy.Vy[1][2]
            M.Vy.Vx[2][1] M.Vy.Vx[2][2] M.Vy.Vy[2][1] M.Vy.Vy[2][2]
            ]
    
        𝐐  = [M.Vx.Pt[1][1] M.Vx.Pt[1][2];
            M.Vx.Pt[2][1] M.Vx.Pt[2][2];  
            M.Vy.Pt[1][1] M.Vy.Pt[1][2];
            M.Vy.Pt[2][1] M.Vy.Pt[2][2];]
        𝐐ᵀ = [M.Pt.Vx[1][1] M.Pt.Vx[1][2] M.Pt.Vy[1][1] M.Pt.Vy[1][2];
            M.Pt.Vx[2][1] M.Pt.Vx[2][2] M.Pt.Vy[2][1] M.Pt.Vy[2][2];]
        𝐏  = [M.Pt.Pt[1][1] M.Pt.Pt[1][2];
            M.Pt.Pt[2][1] M.Pt.Pt[2][2];] 
        𝐌 = [𝐊 𝐐; 𝐐ᵀ 𝐏]

        𝐊diff =  𝐊 - 𝐊'
        droptol!(𝐊diff, 1e-11)
        display(𝐊diff)
        @show 𝐊diff[2,:]
        @show 𝐊diff[:,2]
        @show norm(𝐊diff)

        # display( M.Vx.Vy[1][1] - M.Vy.Vx[1][1]')

        # Set global residual vector
        r  = zeros(sum(nVx) + sum(nVy) + sum(nPt))
        dx = zeros(sum(nVx) + sum(nVy) + sum(nPt))

        SetRHS!(r, R, number, type, nc)

        # dx = - 𝐌 \ r

        𝐏inv = -ηb .* I(size(𝐏,1)) 
        𝐊sc   = 𝐊 .- 𝐐*(𝐏inv*𝐐ᵀ)
        # 𝐊fact = cholesky(Hermitian(𝐊sc), check=false)
        𝐊fact = lu(𝐊sc)

        u     = zeros(size(𝐊,1))
        ru    = zeros(size(𝐊,1))
        fusc  = zeros(size(𝐊,1))
        p     = zeros(size(𝐐,2))
        rp    = zeros(size(𝐐,2))
        fu    = -r[1:size(𝐊,1)]
        fp    = -r[size(𝐊,1)+1:end]
        # Iterations
        for rit=1:niter_l           
            ru   .= fu .- 𝐊*u  .- 𝐐*p
            rp   .= fp .- 𝐐ᵀ*u .- 𝐏*p
            nrmu = norm(ru)
            nrmp = norm(rp)
            @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
            if nrmu/sqrt(length(ru)) < ϵ_l && nrmp/sqrt(length(rp)) < ϵ_l
                break
            end
            fusc .= fu  .- 𝐐*(𝐏inv*fp .+ p)
            u    .= 𝐊fact\fusc
            p   .+= 𝐏inv*(fp .- 𝐐ᵀ*u .- 𝐏*p)
        end
        
        dx[1:size(𝐊,1)]     .= u
        dx[size(𝐊,1)+1:end] .= p
        
        UpdateSolution!(V, Pt, dx, number, type, nc)

        # ############# TEST SG1
        # 𝐊  = [M.Vx.Vx[1][1] M.Vx.Vy[1][2] ; 
        #       M.Vy.Vx[2][1] M.Vy.Vy[2][2] 
        #       ]
    
        # 𝐐  = [M.Vx.Pt[1][1] 
        #       M.Vy.Pt[2][1];
        #       ]
        # 𝐐ᵀ = [M.Pt.Vx[1][1] M.Pt.Vy[1][2];]
        # 𝐏  = [M.Pt.Pt[1][1];]
        # 𝐌 = [𝐊 𝐐; 𝐐ᵀ 𝐏]

        # display(𝐊)
        # 𝐊diff =  𝐊 - 𝐊'
        # droptol!(𝐊diff, 1e-11)
        # display(𝐊diff)   

        # # Set global residual vector
        # r = zeros(sum(nVx[1]) + sum(nVy[2]) + sum(nPt[1]))
        # SetRHSSG1!(r, R, number, type, nc)

        # dx = - 𝐌 \ r
        # cholesky(𝐊)

        # UpdateSolutionSG1!(V, Pt, dx, number, type, nc)
        # ############# TEST SG1

        ############# TEST SG2
        # 𝐊  = [M.Vx.Vx[2][2] M.Vx.Vy[2][1]; 
        #       M.Vy.Vx[1][2] M.Vy.Vy[1][1] 
        #       ]
    
        # 𝐐  = [M.Vx.Pt[2][2] 
        #       M.Vy.Pt[1][2];
        #       ]
        # 𝐐ᵀ = [M.Pt.Vx[2][2] M.Pt.Vy[2][1];]
        # 𝐏  = [M.Pt.Pt[2][2];]
        # 𝐌 = [𝐊 𝐐; 𝐐ᵀ 𝐏]

        # display(𝐊)
        # 𝐊diff =  𝐊 - 𝐊'
        # droptol!(𝐊diff, 1e-11)
        # display(𝐊diff)        

        # # Set global residual vector
        # r = zeros(sum(nVx[2]) + sum(nVy[1]) + sum(nPt[2]))
        # SetRHSSG2!(r, R, number, type, nc)

        # dx = - 𝐌 \ r
        # cholesky(𝐊)
        
        # UpdateSolutionSG2!(V, Pt, dx, number, type, nc)

        # ############# TEST SG2
    end

    # Data on SG1
    p1 = heatmap(xv, yc[iny_V[1]], V.x[1][inx_V[1],iny_V[1]]', aspect_ratio=1, xlim=extrema(xc))
    p2 = heatmap(xc[inx_V[2]], yv, V.y[2][inx_V[2],iny_V[2]]', aspect_ratio=1, xlim=extrema(xc))
    p3 = heatmap(xc[inx_P[1]], yc[iny_P[1]],  Pt[1][inx_P[1],iny_P[1]]' .- mean(Pt[1][inx_P[1],iny_P[1]]'), aspect_ratio=1, xlim=extrema(xc), clims=(-3.2,3.2))
    p4 = plot(xlabel="Iterations", ylabel="log₁₀ error")
    p4 = plot!(1:niter_nl, log10.(err.Vx[1][1:niter_nl]), label="Vx")
    p4 = plot!(1:niter_nl, log10.(err.Vy[2][1:niter_nl]), label="Vy")
    p4 = plot!(1:niter_nl, log10.(err.Pt[1][1:niter_nl]), label="Pt")
    display(plot(p1, p2, p3, p4))

    # Data on SG2
    p1 = heatmap(xc[inx_V[2]], yv, V.x[2][inx_V[2],iny_V[2]]', aspect_ratio=1, xlim=extrema(xc))
    p2 = heatmap(xv, yc[iny_V[1]], V.y[1][inx_V[1],iny_V[1]]', aspect_ratio=1, xlim=extrema(xc))
    p3 = heatmap(xv[2:end-1], yv[2:end-0],  Pt[2][2:end-1,2:end-0]' .- mean(Pt[2][inx_P[1],iny_P[1]]'), aspect_ratio=1, xlim=extrema(xc), clims=(-3.2,3.2))
    p4 = plot(xlabel="Iterations", ylabel="log₁₀ error")
    p4 = plot!(1:niter_nl, log10.(err.Vx[2][1:niter_nl]), label="Vx")
    p4 = plot!(1:niter_nl, log10.(err.Vy[1][1:niter_nl]), label="Vy")
    p4 = plot!(1:niter_nl, log10.(err.Pt[2][1:niter_nl]), label="Pt")
    display(plot(p1, p2, p3, p4))

    # printxy(type.Vx[2])
    # printxy(number.Vx[2])

    #--------------------------------------------#
end

main((x=100, y=100)) 

