using DoubleFloats, LinearAlgebra

function ∂ϕ̂0_∂v_fun(l,N,L,M,R,D,D2)
    ∂ϕ̂0_∂v = zeros(Double64, N+1, 2(N+1));
    ∂ϕ̂0_∂v[:,1:N+1] = Diagonal( -2*M^2 ./ (L^2*(L^2 .+ 2M*R)) ) +
                        4*M^2*R .*(L^2 .+ M*R) ./ (l*(l+1)*(L^2 .+ 2M*R).^3) .* D +
                        2*M^2*R.^2 ./(l*(l+1)*(L^2 .+ 2M*R).^2) .* D2;

    ∂ϕ̂0_∂v[:,N+2:2(N+1)] =  Diagonal( L^2*M ./(2*l*(l+1)*(L^2 .+ 2M*R).^3) ) -
                            L^2 ./ (4*l*(l+1)*(L^2 .+ 2M*R).^2) .* D; 
    return ∂ϕ̂0_∂v
end

function ∂Δ̂ϕ̂0_∂v_fun(l,N,L,M,R,D,D2)
    ∂Δ̂ϕ̂0_∂v = zeros(Double64, N+1, 2(N+1));
    ∂Δ̂ϕ̂0_∂v[:,1:N+1] = Diagonal(-2*M^2*R ./ (L^4*(L^2 .+ 2M*R))) + 
                    2*M^2*R.^2 .*( (2+l+l^2)*L^4 .+ 2*(1+2l*(l+1))*L^2*M*R + 4l*(l+1)*M^2*R.^2 ) ./ (l*(l+1)*L^4*(L^2 .+ 2M*R).^3) .* D + 
                    (2*M^2*R.^3) ./ (l*(l+1)*L^2*(L^2 .+ 2M*R).^2) .* D2;
    
    ∂Δ̂ϕ̂0_∂v[:,N+2:2(N+1)] = Diagonal( M*R./(2*l*(l+1)*(L^2 .+ 2M*R).^3) - 1 ./(4*L^2*(L^2 .+ 2M*R))) -
                            R./(4*l*(l+1)*(L^2 .+ 2M*R).^2) .* D
    
    return ∂Δ̂ϕ̂0_∂v
end

function ∂Δ̂²ϕ̂0_∂v_fun(l,N,L,M,R,D,D2)
    ∂Δ̂²ϕ̂0_∂v = zeros(Double64, N+1, 2(N+1));
    ∂Δ̂²ϕ̂0_∂v[:,1:N+1] = Diagonal( Double64(l*(l+1)//(2*L^4)) .- 4*M^2*R.^2 ./ (L^6*(L^2 .+ 2M*R))) +
                        8*M^2*R.^3 .* ( (1+l*(l+1))*L^4 .+ (L+2l*L)^2*M*R + 4l*(l+1)*M^2*R.^2 ) ./(l*(l+1)*L^6*(L^2 .+ 2M*R).^3) .*D +
                        4*M^2*R.^4 ./ (l*(l+1)*L^4*(L^2 .+ 2M*R).^2) .* D2;
    ∂Δ̂²ϕ̂0_∂v
    return ∂Δ̂²ϕ̂0_∂v
end
function Source_fun(vl1,vl2,l1,l2,m2,∂Δϕ2_∂v,∂Δ²ϕ2_∂v_l1,∂ϕ̂1_∂v_l2,∂Δ̂ϕ̂1_∂v_l2,∂ϕ̂0_∂v_l2,∂Δ̂ϕ̂0_∂v_l2,∂Δ̂²ϕ̂0_∂v_l2,Ł_1_l1,Ł_1_l2,N,R,L,G20,G11,G31)
    # vl1,vl2 are evolving
    # other inputs are constant during the evolution

    # reconstruction
    # l1 
    ψ̂l1 =  vl1[1:N+1,:]; Δϕ2l1 = ∂Δϕ2_∂v * vl1; Δ²ϕ2l1 = ∂Δ²ϕ2_∂v_l1 * vl1; ϕ2l1 = R .* ψ̂l1;
    # l2
    ψ̂l2 =  vl2[1:N+1,:];
    ϕ̂1l2 = ∂ϕ̂1_∂v_l2 * vl2; Δ̂ϕ̂1l2 = 1/L^2 * ( 2*R.*ϕ̂1l2 + sqrt(l2*(l2+1)/df64"2")*ψ̂l2);
    ϕ̂0l2 = ∂ϕ̂0_∂v_l2 * vl2; Δ̂ϕ̂0l2 = 1/L^2 * (   R.*ϕ̂0l2 + sqrt(l2*(l2+1)/df64"2")*ϕ̂1l2); Δ̂²ϕ̂0l2 = 1/L^2 *(2*R.*Δ̂ϕ̂0l2 + sqrt(l2*(l2+1)/df64"2")*Δ̂ϕ̂1l2); 

    # Source caculation
    # S1
    # use factor to save space
    factor =  sqrt(df64"2") * L^2 * (-1)^m2 * ( sqrt( Double64((l1-1)*(l1+2)) )*G20 + sqrt(Double64((l2+1)*l2))*G11 );
    # use tmp's to save time by space
    tmp = (Δϕ2l1 - 3*R.^2/L^2 .* ψ̂l1); 
    S1 =      factor*2*( ϕ̂1l2 .* tmp + ϕ2l1 .* Δ̂ϕ̂1l2 );
    ∂S1∂vl1 = factor*2*( ϕ̂1l2 .* (∂Δϕ2_∂v - [Diagonal(3*R.^2/L^2) zeros(Double64,N+1,N+1)]) + Δ̂ϕ̂1l2 .* [Diagonal(R) zeros(Double64,N+1,N+1)]);
    ∂S1∂vl2 = factor*2*( tmp .* ∂ϕ̂1_∂v_l2 + ϕ2l1 .* ∂Δ̂ϕ̂1_∂v_l2 );
    # S2
    factor = 2*L^4*(-1)^m2 * G11 ;
    # save time by space
    tmp = 2*(Δ̂ϕ̂0l2 - 3*R/L^2 .* ϕ̂0l2);
    tmp2  = Δ̂²ϕ̂0l2 - 6*R/L^2 .* Δ̂ϕ̂0l2 + 4*R.^2/L^4 .* ϕ̂0l2 ;
    S2 = factor*( Δ²ϕ2l1 .* ϕ̂0l2 + Δϕ2l1 .* tmp + ϕ2l1 .* tmp2 );
    ∂S2∂vl1 = factor*( ϕ̂0l2.* ∂Δ²ϕ2_∂v_l1 + tmp .* ∂Δϕ2_∂v + tmp2.* [Diagonal(R) zeros(Double64,N+1,N+1)] );
    # S2 = factor*(  (Δ²ϕ2l1 - Δϕ2l1.* 6*R/L^2 + ϕ2l1 .* 4*R.^2/L^4) .* ϕ̂0l2
    # +   2* (Δϕ2l1  -  ϕ2l1 .* 3*R/L^2 ).* Δ̂ϕ̂0l2
    # +  ϕ2l1 .* Δ̂²ϕ̂0l2   )
    ∂S2∂vl2 = factor*( (Δ²ϕ2l1 - Δϕ2l1 .* 6 .*R/L^2 + ϕ2l1 .* 4 .*R.^2/L^4) .* ∂ϕ̂0_∂v_l2 +
                        2*(Δϕ2l1  -  ϕ2l1 .* 3 .*R/L^2).* ∂Δ̂ϕ̂0_∂v_l2 +
                        ϕ2l1 .* ∂Δ̂²ϕ̂0_∂v_l2 );
    # S3
    factor = (-1)^m2 *( l2*(l2+1)*G11 + sqrt(Double64((l1-1)*(l1+2))) * 
                        ( 2*sqrt(Double64(l2*(l2+1)))*G20 + sqrt(Double64((l1-2)*(l1+3))*G31^2)) );
    S3 = factor * ϕ2l1 .* ψ̂l2;
    ∂S3∂vl1 = factor * ψ̂l2  .* [Diagonal(R) zeros(Double64,N+1,N+1)];
    ∂S3∂vl2 = factor * ϕ2l1 .* [I(N+1) zeros(Double64,N+1,N+1)];

    S = S1 - S2 - S3;
    ∂S∂vl1 = ∂S1∂vl1 - ∂S2∂vl1 - ∂S3∂vl1;
    ∂S∂vl2 = ∂S1∂vl2 - ∂S2∂vl2 - ∂S3∂vl2;
    Ṡ = ∂S∂vl1 * Ł_1_l1 * vl1 +  ∂S∂vl2 * Ł_1_l2 * vl2; 
    # Note that both S,Ṡ are of size(N+1,)
    # exponential filter
    # S = expfilter!(S);
    return S,Ṡ
end