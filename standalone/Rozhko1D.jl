using Plots, Printf, Statistics 

function Rozhko2008(rho, phi, r1, rc, P0, dPf, m, G, ν)
    eta   = (1.0 - 2.0*ν)/(1.0-ν)/2.0
    kappa = 3. - 4. * ν
    if rho < r1
        Pf   = dPf
        Ux   = 0.
        Uy   = 0.
        Ur   = 0.
        Ut   = 0.
        Pt   = 0.
        Sxx  = 0.  
        Syy  = 0.  
        Sxy  = 0. 
        Srr  = 0. 
        Stt  = 0. 
        Srt  = 0.  
    else
        Srr = (eta*rho^2*dPf*m^3*cos(2*phi)*log(1/(rho^32))+eta*rho^2*dPf*m^4*log(1/(rc^8))+eta*dPf*m^4*log(rho^8*rc^8)+eta*rho^6*dPf*log(rc^8)+eta*m^2*rho^4*P0*log(1/(rho^32))+eta*m^2*rho^4*dPf*log(rho^32)+eta*m*rho^6*P0*cos(2*phi)*log(rho^32)+eta*m^2*rho^2*dPf*log(1/(rc^8))+eta*rho^2*P0*m^4*log(rc^8)+eta*rho^8*dPf*log(1/rc^8*rho^8)+eta*rho^8*P0*log(1/rho^8*rc^8)+8*eta*P0*m^4+24*eta*m^2*rho^2*dPf+16*eta*m^2*rho^4*P0-16*eta*m^2*rho^4*dPf-8*eta*rho^6*dPf*m^2+8*eta*rho^6*P0*m^2-8*eta*dPf*m^4-24*eta*m^2*rho^2*P0+eta*rho^6*dPf*m^2*log(rc^8)+8*eta*rho^2*dPf*m^4-8*eta*rho^2*P0*m^4+8*eta*rho^2*dPf*m^3*cos(2*phi)-8*eta*m^3*dPf*cos(2*phi)+8*eta*m^3*P0*cos(2*phi)+eta*P0*m^4*log(1/(rho^8*rc^8))+eta*rho^6*P0*log(1/(rc^8))+24*eta*m*rho^4*P0*cos(2*phi)-8*eta*m^2*rho^2*P0*cos(4*phi)+8*eta*m^2*rho^4*P0*cos(4*phi)-24*eta*m*rho^6*P0*cos(2*phi)+8*eta*m^2*rho^2*dPf*cos(4*phi)-8*eta*m^2*rho^4*dPf*cos(4*phi)-8*eta*rho^2*P0*m^3*cos(2*phi)+eta*m^2*rho^4*dPf*log(rho^16)*cos(4*phi)+eta*m^2*rho^4*P0*log(1/(rho^16))*cos(4*phi)-24*eta*m*rho^4*dPf*cos(2*phi)+eta*rho^2*P0*m^3*cos(2*phi)*log(rho^32)+24*eta*m*rho^6*dPf*cos(2*phi)+eta*m^2*rho^2*P0*log(rc^8)+eta*rho^6*P0*m^2*log(1/(rc^8))+eta*m*rho^6*dPf*cos(2*phi)*log(1/(rho^32)))/(m^2*rho^4*cos(4*phi)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*phi)*log(1/(rc^32))+rho^2*m^3*cos(2*phi)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
        Stt = (eta*dPf*m^4*log(rho^8*rc^8)+eta*m^2*rho^4*P0*log(1/(rho^32))+eta*m^2*rho^4*dPf*log(rho^32)+eta*m^2*rho^2*dPf*log(rc^8)+eta*rho^6*P0*m^2*log(rc^8)+eta*rho^2*dPf*m^4*log(rc^8)+eta*m^2*rho^2*P0*log(1/(rc^8))+eta*rho^8*dPf*log(1/rc^8*rho^8)+eta*rho^8*P0*log(1/rho^8*rc^8)+16*eta*P0*m^4-24*eta*m^2*rho^2*dPf+16*eta*m^2*rho^4*P0-16*eta*m^2*rho^4*dPf+8*eta*rho^6*dPf*m^2-8*eta*rho^6*P0*m^2-16*eta*dPf*m^4+24*eta*m^2*rho^2*P0-8*eta*rho^2*dPf*m^4+8*eta*rho^2*P0*m^4+56*eta*rho^2*dPf*m^3*cos(2*phi)+8*eta*m^3*dPf*cos(2*phi)-8*eta*m^3*P0*cos(2*phi)+eta*P0*m^4*log(1/(rho^8*rc^8))-24*eta*m*rho^4*P0*cos(2*phi)+8*eta*m^2*rho^2*P0*cos(4*phi)+8*eta*m^2*rho^4*P0*cos(4*phi)+24*eta*m*rho^6*P0*cos(2*phi)-8*eta*m^2*rho^2*dPf*cos(4*phi)-8*eta*m^2*rho^4*dPf*cos(4*phi)-56*eta*rho^2*P0*m^3*cos(2*phi)+eta*m^2*rho^4*dPf*log(rho^16)*cos(4*phi)+eta*m^2*rho^4*P0*log(1/(rho^16))*cos(4*phi)+24*eta*m*rho^4*dPf*cos(2*phi)-24*eta*m*rho^6*dPf*cos(2*phi)+eta*rho^6*dPf*log(1/(rc^8))+eta*rho^6*dPf*m^2*log(1/(rc^8))+eta*rho^2*P0*m^3*cos(2*phi)*log(rho^32*rc^32)+eta*rho^6*P0*log(rc^8)+eta*rho^2*P0*m^4*log(1/(rc^8))+eta*m*rho^6*P0*cos(2*phi)*log(1/rc^32*rho^32)+8*eta*rho^8*dPf-8*eta*rho^8*P0+eta*m*rho^6*dPf*cos(2*phi)*log(1/rho^32*rc^32)+eta*rho^2*dPf*m^3*cos(2*phi)*log(1/(rho^32*rc^32)))/(m^2*rho^4*cos(4*phi)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*phi)*log(1/(rc^32))+rho^2*m^3*cos(2*phi)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
        Srt = eta*m*sin(2*phi)*(-2*rho^6*dPf*log(rc)+2*rho^2*log(rc)*P0*m^2-2*rho^4*log(rc)*P0*m^2+2*rho^4*dPf*log(rc)*m^2+2*m*dPf*rho^2*cos(2*phi)-2*m*P0*rho^2*cos(2*phi)-2*rho^4*dPf*m^2+2*rho^6*log(rc)*P0+2*rho^4*P0*m^2-2*m*rho^4*dPf*cos(2*phi)-m^2*dPf+m^2*P0-3*rho^2*P0*m^2+3*rho^4*P0-3*rho^4*dPf+3*rho^2*dPf*m^2-3*rho^6*P0+3*rho^6*dPf+2*m*rho^4*P0*cos(2*phi)+2*rho^4*dPf*log(rc)-2*rho^2*dPf*log(rc)*m^2-2*rho^4*log(rc)*P0)/log(rc)/(4*m^2*rho^4*cos(2*phi)^2+2*m^2*rho^4-4*rho^6*m*cos(2*phi)-4*rho^2*m^3*cos(2*phi)+rho^8+m^4);
                
        Ux  = -1/8*eta*r1*cos(phi)*(11*m*rho^4*dPf-11*m*rho^4*P0+kappa*rho^6*dPf+4*m^3*log(rho)*P0-4*m^3*log(rho)*dPf+5*rho^2*P0*m^2-kappa*rho^6*P0+4*rho^2*P0*m^3-3*kappa*m^3*dPf+3*kappa*m^3*P0-4*rho^2*dPf*m^3+12*m*P0*rho^2-12*rho^2*m*dPf-4*rho^2*m*log(rc)*P0+2*kappa*log(rc)*rho^6*P0+4*rho^2*dPf*log(rc)*m^3+4*kappa*log(rc)*m^3*dPf-20*rho^4*m*dPf*cos(phi)^2+6*rho^4*m*log(rc)*P0+20*m*P0*rho^4*cos(phi)^2+12*dPf*m^2*cos(phi)^2*rho^2-16*m*P0*cos(phi)^2*rho^2-12*P0*m^2*cos(phi)^2*rho^2+16*m*dPf*cos(phi)^2*rho^2+dPf*m^3+4*P0*m^2-rho^6*P0+rho^6*dPf-5*rho^2*dPf*m^2-8*kappa*log(rc)*m*P0*rho^4*cos(phi)^2+8*kappa*log(rc)*m^2*dPf*rho^2+2*kappa*log(rc)*m*P0*rho^4-2*kappa*log(rc)*m^2*P0*rho^2-4*kappa*rho^4*dPf*m*cos(phi)^2+4*kappa*log(rc)*m*dPf*rho^4+4*kappa*m*P0*rho^4*cos(phi)^2-8*rho^4*m*log(rc)*P0*cos(phi)^2+16*cos(phi)^2*dPf*rho^2*log(rho)*m^2+4*rho^4*dPf*m^2-4*rho^4*P0*m^2-5*kappa*rho^2*dPf*m^2+5*kappa*m^2*P0*rho^2+16*cos(phi)^2*m*log(rho)*dPf*rho^4-16*cos(phi)^2*rho^2*P0*log(rho)*m^2-16*cos(phi)^2*m*log(rho)*P0*rho^4+8*kappa*log(rc)*m^2*P0*cos(phi)^2*rho^2-16*kappa*log(rc)*m^2*dPf*cos(phi)^2*rho^2-12*kappa*m^2*P0*cos(phi)^2*rho^2-16*log(rc)*m^2*dPf*cos(phi)^2*rho^2+4*rho^6*dPf*log(rc)+4*rho^4*log(rc)*P0-4*rho^4*dPf*log(rc)+8*log(rc)*m^2*P0*cos(phi)^2*rho^2+12*kappa*m^2*dPf*cos(phi)^2*rho^2-2*rho^6*log(rc)*P0-4*dPf*m^2-4*rho^4*dPf*log(rc)*m^2+4*rho^4*log(rc)*P0*m^2+12*rho^2*dPf*log(rc)*m^2-6*rho^2*log(rc)*P0*m^2-P0*m^3+kappa*m*P0*rho^4-kappa*rho^4*dPf*m-12*dPf*rho^2*log(rho)*m^2+12*rho^2*P0*log(rho)*m^2-4*rho^2*log(rc)*P0*m^3-12*m*log(rho)*dPf*rho^4+12*m*log(rho)*P0*rho^4-2*kappa*log(rc)*m^3*P0+4*rho^2*m*dPf*log(rc)+4*rho^6*P0*log(rho)-4*dPf*rho^6*log(rho)+2*log(rc)*m^3*P0)/rho/log(rc)/G/(-m^2+4*m*rho^2*cos(phi)^2-2*m*rho^2-rho^4);       
        Uy  = -1/8*eta*r1*sin(phi)*(-9*m*rho^4*dPf+9*m*rho^4*P0-kappa*rho^6*dPf+4*m^3*log(rho)*P0-4*m^3*log(rho)*dPf+7*rho^2*P0*m^2+kappa*rho^6*P0+4*rho^2*P0*m^3-3*kappa*m^3*dPf+3*kappa*m^3*P0-4*rho^2*dPf*m^3-4*m*P0*rho^2+4*rho^2*m*dPf-4*rho^2*m*log(rc)*P0-2*kappa*log(rc)*rho^6*P0+4*rho^2*dPf*log(rc)*m^3+4*kappa*log(rc)*m^3*dPf+20*rho^4*m*dPf*cos(phi)^2-2*rho^4*m*log(rc)*P0-20*m*P0*rho^4*cos(phi)^2+12*dPf*m^2*cos(phi)^2*rho^2+16*m*P0*cos(phi)^2*rho^2-12*P0*m^2*cos(phi)^2*rho^2-16*m*dPf*cos(phi)^2*rho^2+dPf*m^3-4*P0*m^2+rho^6*P0-rho^6*dPf-7*rho^2*dPf*m^2+8*kappa*log(rc)*m*P0*rho^4*cos(phi)^2+8*kappa*log(rc)*m^2*dPf*rho^2-6*kappa*log(rc)*m*P0*rho^4-6*kappa*log(rc)*m^2*P0*rho^2+4*kappa*rho^4*dPf*m*cos(phi)^2+4*kappa*log(rc)*m*dPf*rho^4-4*kappa*m*P0*rho^4*cos(phi)^2+8*rho^4*m*log(rc)*P0*cos(phi)^2+16*cos(phi)^2*dPf*rho^2*log(rho)*m^2-4*rho^4*dPf*m^2+4*rho^4*P0*m^2-7*kappa*rho^2*dPf*m^2+7*kappa*m^2*P0*rho^2-16*cos(phi)^2*m*log(rho)*dPf*rho^4-16*cos(phi)^2*rho^2*P0*log(rho)*m^2+16*cos(phi)^2*m*log(rho)*P0*rho^4+8*kappa*log(rc)*m^2*P0*cos(phi)^2*rho^2-16*kappa*log(rc)*m^2*dPf*cos(phi)^2*rho^2-12*kappa*m^2*P0*cos(phi)^2*rho^2-16*log(rc)*m^2*dPf*cos(phi)^2*rho^2-4*rho^6*dPf*log(rc)-4*rho^4*log(rc)*P0+4*rho^4*dPf*log(rc)+8*log(rc)*m^2*P0*cos(phi)^2*rho^2+12*kappa*m^2*dPf*cos(phi)^2*rho^2+2*rho^6*log(rc)*P0+4*dPf*m^2+4*rho^4*dPf*log(rc)*m^2-4*rho^4*log(rc)*P0*m^2+4*rho^2*dPf*log(rc)*m^2-2*rho^2*log(rc)*P0*m^2-P0*m^3+5*kappa*m*P0*rho^4-5*kappa*rho^4*dPf*m-4*dPf*rho^2*log(rho)*m^2+4*rho^2*P0*log(rho)*m^2-4*rho^2*log(rc)*P0*m^3+4*m*log(rho)*dPf*rho^4-4*m*log(rho)*P0*rho^4-2*kappa*log(rc)*m^3*P0+4*rho^2*m*dPf*log(rc)-4*rho^6*P0*log(rho)+4*dPf*rho^6*log(rho)+2*log(rc)*m^3*P0)/rho/log(rc)/G/(m^2-4*m*rho^2*cos(phi)^2+2*m*rho^2+rho^4);

        Ur  =  1/8*r1*eta*(-4*rho^2*dPf*log(rc)*m^2+4*m^2*log(rho)*dPf-4*m^2*log(rho)*P0-4*rho^2*log(rc)*m*P0*cos(2*phi)-2*rho^4*log(rc)*P0+4*rho^4*dPf*log(rc)+4*rho^2*log(rc)*P0*m^2+4*kappa*log(rc)*m*dPf*rho^2*cos(2*phi)+4*rho^2*log(rc)*m*dPf*cos(2*phi)+2*kappa*log(rc)*m^2*P0-4*kappa*log(rc)*m^2*dPf+2*kappa*log(rc)*rho^4*P0-4*m*P0*cos(2*phi)-4*kappa*rho^2*dPf*m*cos(2*phi)+4*kappa*rho^2*P0*m*cos(2*phi)-2*log(rc)*m^2*P0-3*kappa*P0*m^2+kappa*rho^4*dPf+4*rho^2*log(rc)*P0+3*kappa*dPf*m^2-4*dPf*rho^2*log(rc)-kappa*rho^4*P0+4*m*dPf*cos(2*phi)+P0*m^2-dPf*m^2-8*m*dPf*rho^2*cos(2*phi)+8*m*P0*rho^2*cos(2*phi)-4*rho^4*dPf*log(rho)+4*rho^4*P0*log(rho)-4*kappa*log(rc)*rho^2*P0*m*cos(2*phi)-rho^4*P0+rho^4*dPf-4*rho^2*P0*m^2+4*rho^2*dPf*m^2)/(-2*m*rho^2*cos(2*phi)+rho^4+m^2)^(1/2)/rho/G/log(rc);
        Ut  = -1/4*r1*eta*m*sin(2*phi)*(2*dPf*rho^2*log(rc)-kappa*rho^2*dPf+rho^2*kappa*P0+2*kappa*log(rc)*dPf*rho^2-rho^2*P0+rho^2*dPf-2*dPf+2*P0-4*rho^2*dPf*log(rho)+4*rho^2*P0*log(rho))/(-2*m*rho^2*cos(2*phi)+rho^4+m^2)^(1/2)/rho/G/log(rc);

        Pf  = P0 + dPf - dPf*log(rho)/log(rc);
        Sxx =  1/2*(((-2*rho.^2+1+rho.^4).*Srr+(-2*rho.^2-1-rho.^4).*Stt).*cos(2*phi)+(-2*rho.^2+1+rho.^4).*Srr+(2*rho.^2+1+rho.^4).*Stt+(-2*rho.^4 .*sin(2*phi)+2*sin(2*phi)).*Srt)./(-2*rho.^2 .*cos(2*phi)+rho.^4+1);
        Syy = -1/2*(((2*rho.^2+1+rho.^4).*Srr+(-rho.^4+2*rho.^2-1).*Stt).*cos(2*phi)+(-2*rho.^2-1-rho.^4).*Srr+(-rho.^4+2*rho.^2-1).*Stt+(-2*rho.^4 .*sin(2*phi)+2*sin(2*phi)).*Srt)./(-2*rho.^2 .*cos(2*phi)+rho.^4+1);
        Sxy =  1/2*((2+2*rho.^4).*Srt.*cos(2*phi)+(-sin(2*phi)+rho.^4 .*sin(2*phi)).*Srr+(sin(2*phi)-rho.^4 .*sin(2*phi)).*Stt-4*Srt.*rho.^2)./(-2*rho.^2 .*cos(2*phi)+rho.^4+1);
        Pt  = -1/2*(Sxx + Stt) # ??? 
    end   
    return (ux=Ux, uy=Uy, ur=Ur, ut=Ut, pt=Pt, pf=Pf, sxx=Sxx, syy=Syy, sxy=Sxy, srr=Srr, stt=Stt, srt=Srt)
end

function main1D()

    two_way = false

    P0     = 0.0
    ΔPf    = 1.0
    r_in   = 1.0
    r_out  = 20.0
    m      = 0.
    G      = 1.
    ν      = 0.3
    K      = 2/3*G*(1+ν)/(1-2ν) 
    k_muf  = 1.0

    ncr  = 400
    Lr   = 30
    Δr   = Lr/ncr
    rv   = LinRange(0, Lr, ncr+1)
    rc   = LinRange(0-Δr/2, Lr+Δr/2, ncr+2)
    Δt   = 1.0
    nt   = 1
    dmp  = 0.15

    Pt    = zeros(ncr+2)
    Pt0   = zeros(ncr+2)
    Pf    = zeros(ncr+2)
    Pf0   = zeros(ncr+2)
    Vr    = zeros(ncr+1)
    Vrc   = zeros(ncr+2)
    qDr   = zeros(ncr+1)
    τrr   = zeros(ncr+2)
    τrr0  = zeros(ncr+2)
    σrr   = zeros(ncr+2)
    τtt   = zeros(ncr+2)
    τtt0  = zeros(ncr+2)
    σtt   = zeros(ncr+2)
    ε̇rr   = zeros(ncr+2)
    ε̇tt   = zeros(ncr+2)
    divV  = zeros(ncr+2)
    RPt   = zeros(ncr+2)
    RPf   = zeros(ncr+2)
    RVr   = zeros(ncr+1)
    ∂Pt∂τ = zeros(ncr+2)
    ∂Pf∂τ = zeros(ncr+2)
    ∂Vr∂τ = zeros(ncr+1)
    σrri  = zeros(ncr+1)
    σtti  = zeros(ncr+1)
    Pfv   = zeros(ncr+1)
    tagVr = zeros(ncr+1)
    tagPf = zeros(ncr+2)

    for i in eachindex(Vr)
        ρ     = sqrt(rv[i].^2 + 0.0^2)
        ϕ     = atan(0., rv[i])
        sol   = Rozhko2008(ρ, ϕ, r_in, r_out, P0, ΔPf, m, G, ν)
        Vr[i] = sol.ur
        if ρ>r_in && ρ<r_out
            tagVr[i] = 1
        end
    end
    Vra = copy(Vr)

    for i in eachindex(Pf)
        ρ     = sqrt(rc[i].^2 + 0.0^2)
        ϕ     = atan(0., rc[i])
        sol   = Rozhko2008(ρ, ϕ, r_in, r_out, P0, ΔPf, m, G, ν)
        Pf[i] = sol.pf
        Pt[i] = sol.pt
        σrr[i] = sol.srr
        σtt[i] = sol.stt
        if ρ>r_in && ρ<r_out
            tagPf[i] = 1
        end
    end
    Pfa  = copy(Pf)
    Pta  = copy(Pt)
    σrra = copy(σrr)
    σtta = copy(σtt)
    
    for it=1:nt

        τrr0 .= τrr
        τtt0 .= τtt
        Pt0  .= Pt
        Pf0  .= Pf

        dτVr = Δr^2/G/Δt/4.1

        for iter=1:100000

            Vrc[2:end-1]  .= 0.5 * (Vr[1:end-1] + Vr[2:end])
            divV[2:end-1] .= diff(Vr) ./ Δr .+ Vrc[2:end-1] ./rc[2:end-1] 
            ε̇rr[2:end-1]  .= diff(Vr) ./ Δr .- 1/3*divV[2:end-1]
            ε̇tt           .= Vrc./rc .- 1/3*divV
            τrr           .= τrr0 .+ 2*G*Δt*ε̇rr
            τtt           .= τtt0 .+ 2*G*Δt*ε̇tt
            σrr           .= -Pt .+ τrr
            σtt           .= -Pt .+ τtt
            qDr[2:end-1]  .= -k_muf .* diff(Pf[2:end-1] ) ./ Δr 
            σrri          .= 0.5 .* (σrr[1:end-1] .+ σrr[2:end])
            σtti          .= 0.5 .* (σtt[1:end-1] .+ σtt[2:end])
            Pfv           .= 0.5 .* (Pf[1:end-1]  .+ Pf[2:end])       
            RVr[2:end-1]  .= diff(σrr[2:end-1]) ./ Δr .+ 1 ./ rv[2:end-1].*(σrri[2:end-1] .- σtti[2:end-1])
            RPt[2:end-1]  .= -(divV[2:end-1]   .+ (Pt[2:end-1] - Pf[2:end-1])./K./Δt)
            RPf[2:end-1]  .= -(diff(qDr)./ Δr - 1 ./ rc[2:end-1] .* diff(Pfv)./ Δr  .- two_way*(Pt[2:end-1] - Pf[2:end-1])./K./Δt)

            if mod(iter, 1000)==0
                err = max(mean(abs.(RVr[tagVr.==1])), mean(abs.(RPt[tagPf.==1])), mean(abs.(RPf[tagPf.==1])))
                @show iter, err
                err < 1e-8 && break
            end

            ∂Vr∂τ .= ( RVr .+ (1-dmp).*∂Vr∂τ)
            ∂Pt∂τ .= ( RPt .+ (1-dmp).*∂Pt∂τ)
            ∂Pf∂τ .= ( RPf .+ (1-dmp).*∂Pf∂τ)
            Vr   .+= dτVr.*∂Vr∂τ.*tagVr 
            Pt   .+= dτVr.*∂Pt∂τ.*tagPf 
            Pf   .+= dτVr.*∂Pf∂τ.*tagPf 

        end


        # This definition allows to match numerical Pt
        σΘΘ = -Pt .+ (-τrr.-τtt)
        P   = -1/3*(σrr+σtt+σΘΘ)

        # This definition allows to match analytical Pt
        # P = -1/2*(σrr+σtt)

        p1 = plot(xlabel="ρ", ylabel="Pf", title=@sprintf("%1.4e", mean(abs.(Pf .- Pfa))), legend=:bottomright)
        p1 = plot!(rc, Pf, label="num.")
        p1 = plot!(rc, Pfa, label="anal.")
        p2 = plot(xlabel="ρ", ylabel="Pt", title=@sprintf("%1.4e", mean(abs.(Pt .- Pta))), legend=:bottomright)
        p2 = plot!(rc, Pt, label="num.")
        p2 = plot!(rc, P, label="num. 2")
        p2 = plot!(rc, Pta, label="anal.")
        p3 = plot(xlabel="ρ", ylabel="Ur", title=@sprintf("%1.4e", mean(abs.(Vr .- Vra))), legend=:bottomright)
        p3 = plot!(rv, Vr, label="num.")
        p3 = plot!(rv, Vra, label="anal.")
        p4 = plot(xlabel="ρ", ylabel="σrr", title=@sprintf("%1.4e", mean(abs.(σrr .- σrra))), legend=:bottomright)
        p4 = plot!(rc, σrr, label="num.")
        p4 = plot!(rc, σrra, label="anal.")
        p5 = plot(xlabel="ρ", ylabel="σtt", title=@sprintf("%1.4e", mean(abs.(σtt .- σtta))), legend=:bottomright)
        p5 = plot!(rc, σtt, label="num.")
        p5 = plot!(rc, σtta, label="anal.")
        display(plot(p1, p2, p3, p4, p5, layout=(3,2)))
    end
end

main1D()