using DoubleFloats, LinearAlgebra
# For Double64 numerical calculation

function Wigner3j(j1,j2,j3,m1,m2,m3)
    if 2*j1 != floor(2*j1) || 2*j2 != floor(2*j2) || 2*j3 != floor(2*j3) || 2*m1 != floor(2*m1) || 2*m2 != floor(2*m2) || 2*m3 != floor(2*m3)
        error("All arguments must be integers or half-integers.");
    end

    if  j1 - m1 != floor( j1 - m1 ) 
        error("2*j1 and 2*m1 must have the same parity");
    end 

    if j2 - m2 != floor( j2 - m2 )
        error("2*j2 and 2*m2 must have the same parity");
    end

    if j3 - m3 != floor( j3 - m3 )
        error("2*j3 and 2*m3 must have the same parity");
    end

    # The triangular inequalities
    if j3 > j1 + j2 || j3 < abs(j1 - j2)
        print("[Warning::Wigner3j]: j3 is out of bounds.");
        return 0
    end

    if abs(m1) > j1
        # warning("m1 is out of bounds.");
        return 0
    end

    if abs(m2) > j2
        # warning("m2 is out of bounds.");
        return 0
    end

    if abs(m3) > j3
        # warning("m3 is out of bounds.");
        return 0
    end

    if m1 + m2 + m3 != 0
        # warning('m1 + m2 is not equal m3.');
        return 0
    end

    # Integer perimeter rule
    if j1+j2+j3 != floor(j1+j2+j3)
        error("j1+j2+j3 is not an integer.")
    end

    t1 = j2 - m1 - j3;
    t2 = j1 + m2 - j3;
    t3 = j1 + j2 - j3;
    t4 = j1 - m1;
    t5 = j2 + m2;

    tmin = max( 0, max( t1, t2 ) );
    tmax = min( t3, min( t4, t5 ) );
    t = tmin : tmax;

    wigner = Double64( sum( (-1).^t .// ( factorial.(t) .* factorial.(t .- t1) .* factorial.(t .- t2) .* factorial.(t3 .- t) .* factorial.(t4 .- t) .* factorial.(t5 .- t) ) ) ); 
    
    wigner = wigner * (-1)^(j1-j2-m3) * sqrt( Double64( factorial(j1+j2-j3) * factorial(j1-j2+j3) * factorial(-j1+j2+j3) // factorial(j1+j2+j3+1) * factorial(j1+m1) * factorial(j1-m1) * factorial(j2+m2) * factorial(j2-m2) * factorial(j3+m3) * factorial(j3-m3) ));

    return  wigner
end

function Gaunt(s1,s2,s3,l1,l2,l3,m1,m2,m3)
    return sqrt( (2*l1+1)*(2*l2+1)*(2*l3+1)/(4*Double64(π))) * Wigner3j(l1,l2,l3,-s1,-s2,-s3) * Wigner3j(l1,l2,l3,m1,m2,m3);
end
