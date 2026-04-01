using GLMakie

function ElasticShells()

    # 2 shell model - cylindrical
# def ur(A,B,r):  # General solution for cylindrical case
#     return A*r - B/2/r
# def Err(A,B,r): # in cylindrical coordinates Err = d(ur)/d(r)
#     return A + B/2/r**2
# def Epp(A,B,r): # in cylindrical coordinates Epp = ur/r + 1/r*d(up)/d(p)
#     return A - B/2/r**2
# def Srr(r,K,G,A,B,C):
#     return (K+4.0/3.0*G)*Err(A,B,r) + (K-2.0/3.0*G)*Epp(A,B,r) - 2.0*K*C



rmin = 1e-10
rmax = 1.
nr   = 1000
r    = LinRange(rmin, rmax, nr)

ρi   = 2700.0
ρf   = 3200.0
c    = -1/2*(log(ρf) - log(ρi))   

r1 = 0.25
r2 = 1.0
K1 = 4e10
G1 = 3e10
K2 = 4e10
G2 = 3e10
P  = 3e9

ph         = ones(Int64, nr) 
ph[r.>r1] .= 2

C1 = c

# From SphereInHole_cylindrical.ipynb
B1 = 0.
A1 = (-3.0*B1.*G1.*r1.^2 + 3.0*B1.*G1.*r2.^2 - B1.*G2.*r1.^2 - 3.0*B1.*G2.*r2.^2 - 3.0*B1.*K2.*r1.^2 + 6.0*C1.*K1.*r1.^4 - 6.0*C1.*K1.*r1.^2 .*r2.^2)./(2.0*G1.*r1.^4 - 2.0*G1.*r1.^2 .*r2.^2 - 2.0*G2.*r1.^4 - 6.0*G2.*r1.^2 .*r2.^2 + 6.0*K1.*r1.^4 - 6.0*K1.*r1.^2 .*r2.^2 - 6.0*K2.*r1.^4);
A2 = (-4.0*B1.*G1 - 3.0*B1.*K1 + 6.0*C1.*K1.*r1.^2)./(2.0*G1.*r1.^2 - 2.0*G1.*r2.^2 - 2.0*G2.*r1.^2 - 6.0*G2.*r2.^2 + 6.0*K1.*r1.^2 - 6.0*K1.*r2.^2 - 6.0*K2.*r1.^2);
B2 = (-4.0*B1.*G1.*r2.^2 - 3.0*B1.*K1.*r2.^2 + 6.0*C1.*K1.*r1.^2 .*r2.^2)./(G1.*r1.^2 - G1.*r2.^2 - G2.*r1.^2 - 3.0*G2.*r2.^2 + 3.0*K1.*r1.^2 - 3.0*K1.*r2.^2 - 3.0*K2.*r1.^2);

Atab = [A1, A2]
Btab = [0 , B2]
Ktab = [K1, K2]
Gtab = [G1, G2]
Ctab = [ c,  0]
A    = Atab[ph]
B    = Btab[ph]
K    = Ktab[ph]
G    = Gtab[ph]
C    = Ctab[ph]
b    = 1 ./r;
c    = -1 ./r.^2;
ur1  = r;
ur2  = -1/2 ./r;
ur   = A.*ur1 + B.*ur2;

Err = A + 0.5*B./r.^2
Epp = (A.*r - 0.5*B./r)./r
Ezz = 0.0
srr = -2*C.*K + ( 4/3*G + K).*Err + (-2/3*G + K).*Epp + (-2/3*G + K).*Ezz
spp = -2*C.*K + (-2/3*G + K).*Err + ( 4/3*G + K).*Epp + (-2/3*G + K).*Ezz
szz = -2*C.*K + (-2/3*G + K).*Err + (-2/3*G + K).*Epp + ( 4/3*G + K).*Ezz
p    = -1/3*(srr+spp+szz);

τ    = sqrt.(1/2*((p .- srr).^2 + (p .- spp).^2 + (p .- szz).^2))

f = Figure()
ax = Axis(f[1,1], title="uᵣ")
lines!(ax, r, ur)
ax = Axis(f[2,1], title="P MPa]")
lines!(ax, r, (p .- 0*p[1])/1e9)
ax = Axis(f[3,1], title="σᵣᵣ [MPa]")
lines!(ax, r, (srr .- 0*srr[1])/1e9)
display(f)

# load('CheckWithAnal.mat')
# subplot(311),
# hold on, plot(xg_plot',Vx_sol,'r.');
# xlim([0,1]);

# subplot(312),
# hold on,plot(xc_plot,P_sol./1e9,'r.');
# xlim([0,1])

# subplot(313),
# hold on, plot(xc_plot',Sxx_sol./1e9,'r.');
# xlim([0,1])

end

ElasticShells()