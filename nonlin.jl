using DoubleFloats, LinearAlgebra
using CSV,DataFrames
using Plots
# using BenchmarkTools

# To include custom module
include("cheb.jl")
include("TimeIntegrator.jl")
include("Maxwell_recon.jl")
include("SpheHarmonics.jl")

## 1.set parameters

# mode coupling (l1,m1)(l2,m2) → (l3,m3)
l1 = 2; l2 = l1; l3 = 4; 
m1 = 2; m2 = -2; m3 = m1-m2; 

# the start time of the tail phase for ϕ₂
Tend1 = 160;    # when l1 = 1;
# Tend1 = 190;  # when l1 = 2; 
# Tend1 = 260;  # when l1 = 3;

Tend2 = 1500;
N = 180; 
k = 0; b = 1; # parameters for Gaussian wave packet
s1 = -1; s2 = -2; # spin weight

# 
G20 = Gaunt(-2,0 ,2,l1,l2,l3,m1,-m2,-m3);
G11 = Gaunt(-1,-1,2,l1,l2,l3,m1,-m2,-m3);
G31 = Gaunt(-3,1 ,2,l1,l2,l3,m1,-m2,-m3);

# default parameters
M = 1; L = 1; 
dT = 1; # the time interval for storing data
T1 = 0:dT:Tend1;

## 2.set grids points and equation coefficients
# We compactify the coordinate r to R via R = L^2/r.

# horizon
rH = 2*M; 
RH = L^2/rH; # RH equals 1/2 by default, which is an exact float.
# Cheb grids
D,R = chebQ(N); R = (R .+ 1) * (RH/2); D = D / (RH/2); D2 = D^2;
# set T step
dt = Double64(2^-10);  # 9.765625e-4
# internal loop
tN = Int(ceil(dT/dt));
# external loop
TN = Int(ceil(Tend1/dT + 1));

# eqcoefs
cTTFun(R) = 16*M^2*(1 .+ 2*M*R/L^2);
cTRFun(R) = -2*(L^2 .- 8*M^2*R.^2/L^2);
cRRFun(R) = -(L^2 .- 2*M*R) .* R.^2/L^2; 
cTFun(R,s)  = 4M*(-s .+ (2+s)*2*M*R/L^2);
cRFun(R,s)  = 2R.*(-(1+s) .+ (s+3)*M*R/L^2) ;
cFun(R,l,s) = 2*(1+s)*M*R/L^2 .+ (l-s)*(l+s+1);

cTT = cTTFun(R); cTR =  cTRFun(R); cRR = cRRFun(R);
cT1 = cTFun(R,s1); cR1 = cRFun(R,s1); 
c1_l1 = cFun(R,l1,s1); c1_l2 = cFun(R,l2,s1); 
cT2 = cTFun(R,s2); cR2 = cRFun(R,s2); c2_l3 = cFun(R,l3,s2);
# Matries
Ł_1_l1 = Ł(N,cTT,cTR,cRR,cT1,cR1,c1_l1,D,D2);
Ł_1_l2 = Ł(N,cTT,cTR,cRR,cT1,cR1,c1_l2,D,D2); 
Ł_2_l3 = Ł(N,cTT,cTR,cRR,cT2,cR2,c2_l3,D,D2);   

A_1_l1 = ( I(2*(N+1)) - dt/2 * Ł_1_l1 * ( I(2*(N+1)) - dt/6 *Ł_1_l1 ) ) \ (dt*Ł_1_l1);
A_2_l3 = ( I(2*(N+1)) - dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6 *Ł_2_l3 ) ) \ (dt*Ł_2_l3);
A_Sn   = ( I(2*(N+1)) - dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6 *Ł_2_l3 ) ) \ ( Double64(2//3)*dt*Ł_2_l3 * ( I(2*(N+1)) - dt/8*Ł_2_l3) );
A_Snp1 = ( I(2*(N+1)) - dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6 *Ł_2_l3 ) ) \ ( Double64(1//3)*dt*Ł_2_l3 * ( I(2*(N+1)) - dt/4*Ł_2_l3) );
A_Ṡ    = ( I(2*(N+1)) - dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6 *Ł_2_l3 ) ) \ ( dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6*Ł_2_l3 ) );

A_Sn   = A_Sn[:,N+2:end];
A_Snp1 = A_Snp1[:,N+2:end];
A_Ṡ    = A_Ṡ[:,N+2:end];

## 3.initial data
center = Double64(RH)/2; w = Double64(RH)/10;

# initial data for ϕ₂
# stationary init
ψ̂1init = k * exp.( -((R .- center)/w).^2 ) .+ b;
ψ̇1init = 0*ψ̂1init; # time derivative of ψ

# non stationary init
# ψ̇1init = k * exp.( -((R .- center)/w).^2 ) .+ b;
# ψ̂1init = 0*ψ̇1init; 

# ingoing init
# nᵀ =  2 .+ 4*M*R/L^2; nᴿ = R.^2/L^2; 
# ψ̇1init = - nᴿ/nᵀ .* D * ψ̂1init; 

# outgoing init
# lᵀ =  4*M^2; lᴿ = (2*M*R .- L^2)/2;
# ψ̇1init = - lᴿ/lᵀ .* D * ψ̂1init; 

# An auxiliary variable P is defined to reduce the 2nd-order equation.
P1init = cTT .* ψ̇1init + cTR .* D * ψ̂1init + cT1 .* ψ̂1init;
vl1 = [ψ̂1init;P1init];

# initial data for Ψ₄
# zero initial data 
u = 0*vl1;

# ψ̂2init = 1 * exp.( -((R .- center)/w).^2 ) .+ 0;
# ψ̇2init = 0*ψ̂2init;
# P2init = cTT .* ψ̇2init + cTR .* D * ψ̂2init + cT2 .* ψ̂2init;
# u = [ψ̂2init;P2init];

sol1 = zeros(Double64,2*(N+1),TN);
sol1[:,1] = vl1;
sol2 = zeros(Double64,2*(N+1),TN);
sol2[:,1] = u;

## 4.Matries needed to reconstruct the Source term S and ∂S_∂v
∂Δϕ2_∂v = zeros(Double64, N+1, 2(N+1))
#∂Δϕ2_∂ψ
∂Δϕ2_∂v[:,1:N+1] = Diagonal( -R/(2M) ) + (R.*(L^4 .- 4M^2*R.^2)/(4*L^2*M^2)) .* D;
#∂Δϕ2_∂P
∂Δϕ2_∂v[:,N+2:2(N+1)] = Diagonal( R/(8*M^2) );

# ∂Δ²ϕ2_∂v is the part which does not depend on l
∂Δ²ϕ2_∂v = zeros(Double64, N+1, 2(N+1));
∂Δ²ϕ2_∂v[:,1:N+1] = Diagonal( R/(4*M^2) ) - R .* (3*L^4 .+ 6*L^2*M*R -16*M^2*R.^2)/(8*L^2*M^3) .*D + 
                    R.*(L^4 .- 4*M^2*R.^2).^2/(16*L^4*M^4) .* D2;
∂Δ²ϕ2_∂v[:,N+2:2(N+1)] = Diagonal( R.*(8*L^2*M^2*R .- 4M*L^4)/(32*L^4*M^4) ) + L^2*R/(32*M^4) .* D;

∂Δ²ϕ2_∂v_l1 = ∂Δ²ϕ2_∂v + R.*(L^2 .+ 2M*R) ./ (4*L^2*M^2) .*Ł_1_l1[N+2:end,:]

# 
∂ϕ̂1_∂v = zeros(Double64, N+1, 2(N+1));
∂ϕ̂1_∂v[:,1:N+1] = ( 2*sqrt(df64"2")*M^2*R.^2 ./ (L^2*(L^2 .+ 2M*R)) ) .* D;
∂ϕ̂1_∂v[:,N+2:2(N+1)] = Diagonal( -1 ./ (2*sqrt(df64"2")*(L^2 .+ 2M*R)) );
∂ϕ̂1_∂v_fun(l) = 1/sqrt(Double64(l*(l+1))) * ∂ϕ̂1_∂v;
∂ϕ̂1_∂v_l2 = ∂ϕ̂1_∂v_fun(l2);

# Δ̂ϕ̂1 = Δϕ1/R²
∂Δ̂ϕ̂1_∂v = zeros(Double64, N+1, 2(N+1));
∂Δ̂ϕ̂1_∂v[:,1:N+1] = (4*sqrt(df64"2")*M^2*R.^3)./(L^4*(L^2 .+ 2M*R)) .* D;
∂Δ̂ϕ̂1_∂v[:,N+2:2(N+1)] = Diagonal( -R ./ (sqrt(df64"2")*L^2*(L^2 .+ 2M*R)) );
∂Δ̂ϕ̂1_∂v_fun(l) = 1/sqrt(Double64(l*(l+1))) * ∂Δ̂ϕ̂1_∂v + sqrt( Double64(l*(l+1))/2 )/L^2 * [I(N+1) zeros(Double64,N+1,N+1)];;
∂Δ̂ϕ̂1_∂v_l2 = ∂Δ̂ϕ̂1_∂v_fun(l2);
#
∂ϕ̂0_∂v_l2 = ∂ϕ̂0_∂v_fun(l2,N,L,M,R,D,D2);
∂Δ̂ϕ̂0_∂v_l2 = ∂Δ̂ϕ̂0_∂v_fun(l2,N,L,M,R,D,D2);
∂Δ̂²ϕ̂0_∂v_l2 = ∂Δ̂²ϕ̂0_∂v_fun(l2,N,L,M,R,D,D2);
# Source function
Sl1l2(vl1,vl2) = Source_fun(vl1,vl2,l1,l2,m2,∂Δϕ2_∂v,∂Δ²ϕ2_∂v_l1,∂ϕ̂1_∂v_l2,∂Δ̂ϕ̂1_∂v_l2,∂ϕ̂0_∂v_l2,∂Δ̂ϕ̂0_∂v_l2,∂Δ̂²ϕ̂0_∂v_l2,Ł_1_l1,Ł_1_l2,N,R,L,G20,G11,G31);

## 5. evolution in cheb grids
print("Start\n")
time_symmetric_integrate_nonlinear!(sol1,sol2,TN,tN,dt,A_1_l1,A_2_l3,A_Sn,A_Snp1,A_Ṡ,vl1,u,Sl1l2)

ψ̂1sol1 = sol1[1:N+1,:]; P1sol1 = sol1[N+2:end,:];
ψ̂2sol1 = sol2[1:N+1,:]; P2sol1 = sol2[N+2:end,:];

R2x(R) = 2*R/RH .- 1;
ψ̂1old = ψ̂1sol1[:,end]; P1old = P1sol1[:,end];
ψ̂2old = ψ̂2sol1[:,end]; P2old = P2sol1[:,end];

κ = abs( log( abs(ψ̂1old[end]/(D[end,:]' * ψ̂1old)) )); # Note that D[end,:] is of size(N+1,1) in Julia.

########################################
## 6. transform into AnMR grids
T2 = Tend1:dT:Tend2; TN = Int(ceil( (Tend2-Tend1)/dT + 1)); # update TN
N = 180; # update N
Dnew,xnew = chebQ(N);

x2R(x) = RH * sinh.( κ*(x .+ 1)/2 ) / sinh(κ)
∂R∂x= RH * κ/2 * cosh.(κ*(xnew .+ 1)/2) / sinh(κ);

Rnew = x2R(xnew);
Dnew = Dnew ./ ∂R∂x;
D2new = Dnew^2;

# update init
ψ̂1init = cheb_interp(real_to_chebQ(ψ̂1old),R2x(Rnew));
P1init = cheb_interp(real_to_chebQ(P1old),R2x(Rnew));
ψ̂2init = cheb_interp(real_to_chebQ(ψ̂2old),R2x(Rnew));
P2init = cheb_interp(real_to_chebQ(P2old),R2x(Rnew));
vl1 = [ψ̂1init;P1init];
u   = [ψ̂2init;P2init];

sol12 = zeros(Double64,2*(N+1),TN);
sol12[:,1] = vl1;
sol22 = zeros(Double64,2*(N+1),TN);
sol22[:,1] = u;

# update eqcoefs R→Rnew, D→Dnew, D2→D2new
cTT = cTTFun(Rnew); cTR =  cTRFun(Rnew); cRR = cRRFun(Rnew);

cT1 = cTFun(Rnew,s1); cR1 = cRFun(Rnew,s1); 
cT2 = cTFun(Rnew,s2); cR2 = cRFun(Rnew,s2); 
c1_l1 = cFun(Rnew,l1,s1); c1_l2 = cFun(Rnew,l2,s1); 
c2_l3 = cFun(Rnew,l3,s2);
# Matries
Ł_1_l1 = Ł(N,cTT,cTR,cRR,cT1,cR1,c1_l1,Dnew,D2new);
Ł_1_l2 = Ł(N,cTT,cTR,cRR,cT1,cR1,c1_l2,Dnew,D2new); 
Ł_2_l3 = Ł(N,cTT,cTR,cRR,cT2,cR2,c2_l3,Dnew,D2new);   

A_1_l1 = ( I(2*(N+1)) - dt/2 * Ł_1_l1 * ( I(2*(N+1)) - dt/6 *Ł_1_l1 ) ) \ (dt*Ł_1_l1);
A_2_l3 = ( I(2*(N+1)) - dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6 *Ł_2_l3 ) ) \ (dt*Ł_2_l3);
A_Sn   = ( I(2*(N+1)) - dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6 *Ł_2_l3 ) ) \ ( Double64(2//3)*dt*Ł_2_l3 * ( I(2*(N+1)) - dt/8*Ł_2_l3) );
A_Snp1 = ( I(2*(N+1)) - dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6 *Ł_2_l3 ) ) \ ( Double64(1//3)*dt*Ł_2_l3 * ( I(2*(N+1)) - dt/4*Ł_2_l3) );
A_Ṡ    = ( I(2*(N+1)) - dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6 *Ł_2_l3 ) ) \ ( dt/2 * Ł_2_l3 * ( I(2*(N+1)) - dt/6*Ł_2_l3 ) );

A_Sn   = A_Sn[:,N+2:end];
A_Snp1 = A_Snp1[:,N+2:end];
A_Ṡ    = A_Ṡ[:,N+2:end];

# Matries needed to reconstruct the Source term S and ∂S_∂v
∂Δϕ2_∂v = zeros(Double64, N+1, 2(N+1))
#∂Δϕ2_∂ψ
∂Δϕ2_∂v[:,1:N+1] = Diagonal( -Rnew/(2M) ) + (Rnew.*(L^4 .- 4M^2*Rnew.^2)/(4*L^2*M^2)) .* Dnew;
#∂Δϕ2_∂P
∂Δϕ2_∂v[:,N+2:2(N+1)] = Diagonal( Rnew/(8*M^2) );

# ∂Δ²ϕ2_∂v is the part which does not depend on l
∂Δ²ϕ2_∂v = zeros(Double64, N+1, 2(N+1));
∂Δ²ϕ2_∂v[:,1:N+1] = Diagonal( Rnew/(4*M^2) ) - Rnew .* (3*L^4 .+ 6*L^2*M*Rnew -16*M^2*Rnew.^2)/(8*L^2*M^3) .*Dnew + 
                    Rnew.*(L^4 .- 4*M^2*Rnew.^2).^2/(16*L^4*M^4) .* D2new;
∂Δ²ϕ2_∂v[:,N+2:2(N+1)] = Diagonal( Rnew.*(8*L^2*M^2*Rnew .- 4M*L^4)/(32*L^4*M^4) ) + L^2*Rnew/(32*M^4) .* Dnew;

∂Δ²ϕ2_∂v_l1 = ∂Δ²ϕ2_∂v + Rnew.*(L^2 .+ 2M*Rnew) ./ (4*L^2*M^2) .*Ł_1_l1[N+2:end,:]

# 
∂ϕ̂1_∂v = zeros(Double64, N+1, 2(N+1));
∂ϕ̂1_∂v[:,1:N+1] = ( 2*sqrt(df64"2")*M^2*Rnew.^2 ./ (L^2*(L^2 .+ 2M*Rnew)) ) .* Dnew;
∂ϕ̂1_∂v[:,N+2:2(N+1)] = Diagonal( -1 ./ (2*sqrt(df64"2")*(L^2 .+ 2M*Rnew)) );
∂ϕ̂1_∂v_fun(l) = 1/sqrt(Double64(l*(l+1))) * ∂ϕ̂1_∂v;
∂ϕ̂1_∂v_l2 = ∂ϕ̂1_∂v_fun(l2);

# Δ̂ϕ̂1 = Δϕ1/R²
∂Δ̂ϕ̂1_∂v = zeros(Double64, N+1, 2(N+1));
∂Δ̂ϕ̂1_∂v[:,1:N+1] = (4*sqrt(df64"2")*M^2*Rnew.^3)./(L^4*(L^2 .+ 2M*Rnew)) .* Dnew;
∂Δ̂ϕ̂1_∂v[:,N+2:2(N+1)] = Diagonal( -Rnew ./ (sqrt(df64"2")*L^2*(L^2 .+ 2M*Rnew)) );
∂Δ̂ϕ̂1_∂v_fun(l) = 1/sqrt(Double64(l*(l+1))) * ∂Δ̂ϕ̂1_∂v + sqrt( Double64(l*(l+1))/2 )/L^2 * [I(N+1) zeros(Double64,N+1,N+1)];
∂Δ̂ϕ̂1_∂v_l2 = ∂Δ̂ϕ̂1_∂v_fun(l2);
#
∂ϕ̂0_∂v_l2 = ∂ϕ̂0_∂v_fun(l2,N,L,M,Rnew,Dnew,D2new);
∂Δ̂ϕ̂0_∂v_l2 = ∂Δ̂ϕ̂0_∂v_fun(l2,N,L,M,Rnew,Dnew,D2new);
∂Δ̂²ϕ̂0_∂v_l2 = ∂Δ̂²ϕ̂0_∂v_fun(l2,N,L,M,Rnew,Dnew,D2new);
# Source function
Sl1l2(vl1,vl2) = Source_fun(vl1,vl2,l1,l2,m2,∂Δϕ2_∂v,∂Δ²ϕ2_∂v_l1,∂ϕ̂1_∂v_l2,∂Δ̂ϕ̂1_∂v_l2,∂ϕ̂0_∂v_l2,∂Δ̂ϕ̂0_∂v_l2,∂Δ̂²ϕ̂0_∂v_l2,Ł_1_l1,Ł_1_l2,N,Rnew,L,G20,G11,G31);
########################################

## 7. evolution in AnMR grids
print("Start_nonlinear\n")
time_symmetric_integrate_nonlinear!(sol12,sol22,TN,tN,dt,A_1_l1,A_2_l3,A_Sn,A_Snp1,A_Ṡ,vl1,u,Sl1l2)

ψ̂1sol2 = sol12[1:N+1,:]; P1sol2 = sol12[N+2:end,:];
ψ̂2sol2 = sol22[1:N+1,:]; P2sol2 = sol22[N+2:end,:];

ψ̇1sol2 = ( P1sol2 - cTR.*(Dnew * ψ̂1sol2) - cT1.*ψ̂1sol2 )./cTT;
ψ̇2sol2 = ( P2sol2 - cTR.*(Dnew * ψ̂2sol2) - cT2.*ψ̂2sol2 )./cTT;

LPI1 = T2' .* ψ̇1sol2 ./ ψ̂1sol2;
LPI2 = T2' .* ψ̇2sol2 ./ ψ̂2sol2;

## save data
CSV.write("Rnew.csv", Rnew);
df = DataFrame(LPI1, :auto); CSV.write("LPI1.csv", df);
df = DataFrame(LPI2, :auto); CSV.write("LPI2.csv", df);
df = DataFrame(ψ̂1sol2, :auto); CSV.write("phi2sol2.csv", df);
df = DataFrame(ψ̂2sol2, :auto); CSV.write("psi4sol2.csv", df);
df = DataFrame(ψ̂1sol1, :auto); CSV.write("phi2sol1.csv", df);
df = DataFrame(ψ̂2sol1, :auto); CSV.write("psi4sol1.csv", df);
print("End")

# plot(T2,LPI2[1:N:end,:]',xlims=(T2[1],T2[end]),ylims=(-(2*l3+4),0))


ϕ̂1sol2 = ∂ϕ̂1_∂v_l2 * sol12;
Δ̂ϕ̂1sol2 = ∂Δ̂ϕ̂1_∂v_l2 * sol12;
ϕ̇1sol2 = (Δ̂ϕ̂1sol2 - 2*Rnew/L^2 .* ϕ̂1sol2 - Rnew.^2/L^2 .* Dnew* ϕ̂1sol2)./(2 .+ 4*M*Rnew/L^2);
LPIϕ̂1 = T2' .* ϕ̇1sol2 ./ ϕ̂1sol2;
# plot(T2,LPIϕ̂1[1,:],xlims=(T2[1],T2[end]),ylims=(-10,0))

ϕ̂0sol2 = ∂ϕ̂0_∂v_l2 * sol12;
Δ̂ϕ̂0sol2 = ∂Δ̂ϕ̂0_∂v_l2 * sol12;
ϕ̇0sol2 = (Δ̂ϕ̂0sol2 - 3*Rnew/L^2 .* ϕ̂0sol2 - Rnew.^2/L^2 .* Dnew * ϕ̂0sol2)./(2 .+4*M*Rnew/L^2);
LPIϕ̂0 = T2' .* ϕ̇0sol2 ./ ϕ̂0sol2;
# plot(T2,LPIϕ̂0[1,:],xlims=(T2[1],T2[end]),ylims=(-10,0))

df = DataFrame(LPIϕ̂1, :auto); CSV.write("LPIphi1.csv", df);
df = DataFrame(LPIϕ̂0, :auto); CSV.write("LPIphi0.csv", df);

# plot(T2,log10.(abs.(ψ̂2sol2[1,:])))

df = DataFrame(real_to_chebQ(ψ̂1old), :auto); CSV.write("phi2old_spec.csv", df);
df = DataFrame(real_to_chebQ(ψ̂1init), :auto); CSV.write("phi2new_spec.csv", df);
df = DataFrame(real_to_chebQ(ψ̂1sol2[:,end]), :auto); CSV.write("phi2end_spec.csv", df);

df = DataFrame(real_to_chebQ(ψ̂2old), :auto); CSV.write("psi4old_spec.csv", df);
df = DataFrame(real_to_chebQ(ψ̂2init), :auto); CSV.write("psi4new_spec.csv", df);
df = DataFrame(real_to_chebQ(ψ̂2sol2[:,end]), :auto); CSV.write("psi4end_spec.csv", df);

Send,Ṡend = Sl1l2(sol12[:,end],sol12[:,end]);
df = DataFrame(Send, :auto); CSV.write("SourceEnd.csv", df);
df = DataFrame(real_to_chebQ(Send), :auto); CSV.write("SourceEnd_spec.csv", df);

ψ̂2H = [ψ̂2sol1[1,:];ψ̂2sol2[1,2:end]];
# plot(0:1000,log10.(abs.(ψ̂2H)))
ψ̂2I = [ψ̂2sol1[end,:];ψ̂2sol2[end,2:end]];

ψ̂1H = [ψ̂1sol1[1,:];ψ̂1sol2[1,2:end]];
# plot(0:1000,log10.(abs.(ψ̂2H)))
ψ̂1I = [ψ̂1sol1[end,:];ψ̂1sol2[end,2:end]];

CSV.write("psi4_H.csv", ψ̂2H);
CSV.write("psi4_I.csv", ψ̂2I);
CSV.write("phi2_H.csv", ψ̂1H);
CSV.write("phi2_I.csv", ψ̂1I);

Source2 = 0*ψ̂2sol2;
Ṡource2 = 0*ψ̂2sol2;
for ii = 1:length(T2)
    Source2[:,ii],Ṡource2[:,ii] = Sl1l2(sol12[:,ii],sol12[:,ii]);
end

df = DataFrame(Source2, :auto); CSV.write("Source2.csv", df);
df = DataFrame(Ṡource2, :auto); CSV.write("Source2_dot.csv", df);