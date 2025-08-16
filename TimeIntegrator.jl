using DoubleFloats, LinearAlgebra

# The operator Ł of the linear part of the ODE
function Ł(N,cTT,cTR,cRR,cT,cR,c,D,D2)
    LM = zeros(Double64,2*(N+1),2*(N+1)) ;
    # ψ̇ = ( P - cTR ψ̂′ - cT ψ̂ )/cTT 
    LM[1:N+1,1:N+1] = -(cTR .* D + Diagonal(cT) ) ./ cTT;
    LM[1:N+1,N+2:end] = Diagonal(1 ./ cTT);
    # Ṗ = -(cRR ψ̂′′ + cR ψ̂′ + c ψ̂)
    LM[N+2:end,1:N+1] = - (cRR.*D2 + cR.*D + Diagonal(c));
    return LM
end
function time_symmetric_integrate!(sol, TN, tN, A, x)
    for ii = 2:TN
        for jj = 1:tN
            x = x + A*x;
        end
        sol[:,ii] = x;
    end
end

function time_symmetric_integrate_nonlinear!(sol1,sol2,TN,tN,dt,A_1_l1,A_2_l3,A_Sn,A_Snp1,A_Ṡ,vl1,u,Sl1l2)
    Sn,Ṡn = Sl1l2(vl1,vl1);
    Ṡn = expfilter!(Ṡn);
    for ii = 2:TN
        for jj = 1:tN
            vl1 = vl1 + A_1_l1*vl1;
            
            Snp1,Ṡnp1 = Sl1l2(vl1,vl1);
            # Ṡnp1 = expfilter!(Ṡnp1);
            Ṡtmp = (Ṡn - Ṡnp1);
            u = u + A_2_l3*u + dt/2*( [zeros(Double64,N+1); Sn] + A_Sn*Sn + 
                                    [zeros(Double64,N+1);Snp1] + A_Snp1*Snp1 + 
                                        dt/6*([zeros(Double64,N+1);Ṡtmp] + A_Ṡ*Ṡtmp));
            
            Sn = Snp1; Ṡn = Ṡnp1;
        end
        sol1[:,ii] = vl1;
        sol2[:,ii] = u;
    end
end
