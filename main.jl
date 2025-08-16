using DoubleFloats, LinearAlgebra

using BenchmarkTools
using CSV,DataFrames
include("cheb.jl")
include("TimeIntegrator.jl")
# promote_type(Int, Double64) == Double64

## set parameters
s = -2; 
l = 3;
Tend = 1000; 
N = 72;
# default parameters
M = 1; 
L = 1;
dT = 1;
T1 = 0:dT:Tend;
## set grids points and equation coefficients
# We compactify the coordinate r to R via R = L^2/r.

# horizon
rH = 2*M; 
RH = L^2/rH; # RH equals 1/2 by default, which is an exact float.
# grids
D,R = chebQ(N);
R = (R .+ 1) * (RH/2);
D = D / (RH/2);
D2 = D^2;

cTTFun(R) = 16 * M^2 * ( 1 .+ 2*M*R/L^2 );
cTRFun(R) = -2 * ( L^2 .- 8*M^2*R.^2/L^2 );
cRRFun(R) = -(L^2 .- 2*M*R ) .* R.^2/L^2; 
cTFun(R)  = 4*M*( -s .+ (2+s)*2*M*R/L^2 );
cRFun(R)  = 2*R.*(-(1+s) .+ (s+3)*M*R/L^2 ) ;
cFun(R)   = 2*(1+s)*M*R/L^2 .+ (l-s)*(l+s+1);

cTT = cTTFun(R);
cTR =  cTRFun(R);
cRR = cRRFun(R);

cT = cTFun(R);
cR = cRFun(R);
c = cFun(R);

LM = zeros(Double64,2*(N+1),2*(N+1)) ;
LM[1:N+1,1:N+1] = -(cTR .* D + Diagonal(cT) ) ./ cTT;
LM[1:N+1,N+2:end] = Diagonal(1 ./ cTT);
LM[N+2:end,1:N+1] = -(cRR .* D2 + cR .* D + Diagonal(c) );

# initial data
k = 1; center = RH/2; w = RH/10; b=0;
ψ̂init = k * exp.( -((R .- center)/w).^2 ) .+ 1;
ψ̇init = 1*ψ̂init; # time derivative of \psi
# An auxiliary variable P is defined to reduce the 2nd-order equation.
Pinit = cTT .* ψ̇init + cTR .* D * ψ̂init + cT .* ψ̂init;

x = [ψ̂init;Pinit];

##
# set T step
dt = Double64(2^-11);  #9.765625e-4
# internal loop
tN = Int(ceil(dT/dt));
# external loop
TN = Int(ceil(Tend/dT + 1));

A = ( I(2*(N+1)) - dt/2 * LM * ( I(2*(N+1)) - dt/6*LM ) ) \ (dt*LM) ;

sol = zeros(Double64,2*(N+1),TN);
sol[:,1] = x;

print("start1\n")
# @btime time_symmetric_integrate!(sol, TN, tN, A, x)
time_symmetric_integrate!(sol, TN, tN, A, x);
print("End\n")
ψ̂sol = sol[1:N+1,:];
Psol = sol[N+2:end,:];

R2x(R) = 2*R/RH .- 1;
ψ̂old = ψ̂sol[:,end];
Pold = Psol[:,end];

κ = abs( log( abs(ψ̂old[end]/(D[end,:]' * ψ̂old) ))); # Note that D[end,:] is of size(N+1,1) in Julia.

ψ̇sol = ( Psol - cTR.*(D * ψ̂sol) - cT.*ψ̂sol )./cTT;
LPI1 = T1' .* ψ̇sol ./ ψ̂sol;

plot(T1,LPI1[1:N:end,:]',xlims=(T1[1],T1[end]),ylims=(-(2*l+4),-l))
########################################
# new grids
# Tend2 = 1000;
# T2 = Tend:dT:Tend2;
# TN = Int(ceil( (Tend2-Tend)/dT + 1)); # update TN

# N = 112; # update N
# Dnew,xnew = chebQ(N);

# x2R(x) = RH * sinh.( κ*(x .+ 1)/2 ) / sinh(κ)
# ∂R∂x= RH * κ/2 * cosh.(κ*(xnew .+ 1)/2) / sinh(κ);

# Rnew = x2R(xnew);
# Dnew = Dnew ./ ∂R∂x;
# D2new = Dnew^2;

# # update init
# ψ̂init = cheb_interp(real_to_chebQ(ψ̂old),R2x(Rnew));
# Pinit = cheb_interp(real_to_chebQ(Pold),R2x(Rnew));
# x = [ψ̂init;Pinit];

# # update eqcoefs
# cTT = cTTFun(Rnew);
# cTR =  cTRFun(Rnew);
# cRR = cRRFun(Rnew);

# cT = cTFun(Rnew);
# cR = cRFun(Rnew);
# c = cFun(Rnew);

# LM = zeros(Double64,2*(N+1),2*(N+1)) ;
# LM[1:N+1,1:N+1] = -(cTR .* Dnew + Diagonal(cT) ) ./ cTT;
# LM[1:N+1,N+2:end] = Diagonal(1 ./ cTT);
# LM[N+2:end,1:N+1] = -(cRR .* D2new + cR .* Dnew + Diagonal(c) );

# A = ( I(2*(N+1)) - dt/2 * LM * ( I(2*(N+1)) - dt/6*LM ) ) \ (dt*LM) ;

# sol2 = zeros(Double64,2*(N+1),TN);
# sol2[:,1] = x;

# print("start2")
# # @btime time_symmetric_integrate!(sol, TN, tN, A, x)
# time_symmetric_integrate!(sol2, TN, tN, A, x);
# ψ̂sol2 = sol2[1:N+1,:];
# Psol2 = sol2[N+2:end,:];

# ψ̇sol2 = ( Psol2 - cTR.*(Dnew * ψ̂sol2) - cT.*ψ̂sol2 )./cTT;
# LPI = T2' .* ψ̇sol2 ./ ψ̂sol2;
# plot(T2,LPI[1:N:end,:]',xlims=(T2[1],T2[end]),ylims=(-(2*l+4),0))