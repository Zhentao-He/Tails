using Plots
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

##########
# add time
Tend3 = 2000;

R2x(R) = 2/κ * asinh.(R*sinh(κ)/RH) .- 1;
ψ̂1old2 = ψ̂1sol2[:,end]; P1old2 = P1sol2[:,end];
ψ̂2old2 = ψ̂2sol2[:,end]; P2old2 = P2sol2[:,end];
κ2 = max(abs( log( abs(ψ̂2old2[end]/(Dnew[end,:]' * ψ̂2old2)) )),abs( log( abs(ψ̂1old2[end]/(Dnew[end,:]' * ψ̂1old2)) )) ); # Note that D[end,:] is of size(N+1,1) in Julia.

# new grids
T3 = Tend2:dT:Tend3; 
TN = Int(ceil( (Tend3-Tend2)/dT + 1)); # update TN
N = 180; # update N
Dnew,xnew = chebQ(N);

x2R(x) = RH * sinh.( κ2*(x .+ 1)/2 ) / sinh(κ2)
∂R∂x= RH * κ2/2 * cosh.(κ2*(xnew .+ 1)/2) / sinh(κ2);

Rnew = x2R(xnew);
Dnew = Dnew ./ ∂R∂x;
D2new = Dnew^2;

# update init
ψ̂1init = cheb_interp(real_to_chebQ(ψ̂1old2),R2x(Rnew));
P1init = cheb_interp(real_to_chebQ(P1old2),R2x(Rnew));
ψ̂2init = cheb_interp(real_to_chebQ(ψ̂2old2),R2x(Rnew));
P2init = cheb_interp(real_to_chebQ(P2old2),R2x(Rnew));

vl1 = [ψ̂1init;P1init];
u   = [ψ̂2init;P2init];

sol13 = zeros(Double64,2*(N+1),TN);
sol13[:,1] = vl1;
sol23 = zeros(Double64,2*(N+1),TN);
sol23[:,1] = u;

########################################
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

print("Start_nonlinear\n")
time_symmetric_integrate_nonlinear!(sol13,sol23,TN,tN,dt,A_1_l1,A_2_l3,A_Sn,A_Snp1,A_Ṡ,vl1,u,Sl1l2)

ψ̂1sol3 = sol13[1:N+1,:]; P1sol3 = sol13[N+2:end,:];
ψ̂2sol3 = sol23[1:N+1,:]; P2sol3 = sol23[N+2:end,:];

ψ̇1sol3 = ( P1sol3 - cTR.*(Dnew * ψ̂1sol3) - cT1.*ψ̂1sol3 )./cTT;
ψ̇2sol3 = ( P2sol3 - cTR.*(Dnew * ψ̂2sol3) - cT2.*ψ̂2sol3 )./cTT;
LPI1_add = T3' .* ψ̇1sol3 ./ ψ̂1sol3;
LPI2_add = T3' .* ψ̇2sol3 ./ ψ̂2sol3;

CSV.write("Rnew_add.csv", Rnew);
df = DataFrame(LPI1_add, :auto); CSV.write("LPI1_add.csv", df);
df = DataFrame(LPI2_add, :auto); CSV.write("LPI2_add.csv", df);
df = DataFrame(ψ̂1sol3, :auto); CSV.write("phi2sol3.csv", df);
df = DataFrame(ψ̂2sol3, :auto); CSV.write("psi4sol3.csv", df);
print("End")

