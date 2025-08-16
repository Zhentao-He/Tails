using DoubleFloats, LinearAlgebra, FFTW, FastTransforms
using Plots
function chebQ(N)

    if N == 0
        D=0;
        x=1;
        return D,x
    end
    x = sin.( Double64(π) / (2*N) * (N .- 2*(0:N)) ); # 0:N is a col vec
    # x = cos.( Double64(π)*(0:N)/N);
    c = [2;ones(Double64,N-1,1);2] .* (-1).^(0:N);
    dx = x .- x';
    D = ( c * (1 ./ c)' ) ./ (dx + I(N+1));
    D = D - Diagonal( sum(D,dims=2)[:] );
    return D,x
end

function cheb(N)

    if N == 0
        D=0;
        x=1;
        return D,x
    end
end

function myFCT1Q(f)
    ## 利用FFT计算 a real data array 'f' 的第一类余弦变换.
    # f 要求为列矢量
    # 系数顺序：模式数从前往后为0到N

    N = length(f) - 1; 
    F = zeros(Double64,N+1,1);
    jj = 0:N-1;

    y = ( f[1:end-1] + f[end:-1:2] )/2 - sin.( jj* Double64(π) /N ) .* ( f[1:end-1] - f[end:-1:2] ); #第二项
    u = fft(y); # 排列顺序为k从0到N-1
    R = real(u);
    I = - imag(u); # Julia FFTW 中FFT的定义为exp(-2*pi*i jk/N)
    # 以下代码默认N为偶数
    kk = 0: Int(N/2);
    F[ 2*kk .+ 1 ] = R[ kk .+ 1 ];#+1是因为Matlab指标从1开始
    j = jj[2:end]; # j从1取到N-1
    F[1+1] = (f[1]-f[end])/2 + sum( f[j .+ 1] .* cos.(j*Double64(π)/N) );
    for k = 1:( Int(N/2)-1 )
        F[2k+1+1] = F[2*k] + I[k+1];
    end

    return F
end

function myFCT1(f)
    ## 利用FFT计算 a real data array 'f' 的第一类余弦变换.
    # f 要求为列矢量
    # 系数顺序：模式数从前往后为0到N

    N = length(f) - 1; 
    F = zeros(Float64,N+1,1);
    jj = 0:N-1;

    y = ( f[1:end-1] + f[end:-1:2] )/2 - sin.( jj* π/N ) .* ( f[1:end-1] - f[end:-1:2] ); #第二项
    u = fft(y); # 排列顺序为k从0到N-1
    R = real(u);
    I = - imag(u); # Julia FFTW 中FFT的定义为exp(-2*pi*i jk/N)
    # 以下代码默认N为偶数
    kk = 0: Int(N/2);
    F[ 2*kk .+ 1 ] = R[ kk .+ 1 ];#+1是因为Matlab指标从1开始
    j = jj[2:end]; # j从1取到N-1
    F[1+1] = (f[1]-f[end])/2 + sum( f[j .+ 1] .* cos.(j*π/N) );
    for k = 1:( Int(N/2)-1 )
        F[2k+1+1] = F[2*k] + I[k+1];
    end

    return F
end

function real_to_chebQ(f)
    N = length(f) - 1; 
    a = myFCT1Q(f);
    a = 2*a/N;
    a[1] = a[1]/2;
    a[end] = a[end]/2;
    return a
end

function cheb_to_realQ(a)
    N = length(a) - 1;
    p = cos.( ( Double64(π)*(0:N)/ N) .* (0:N)' ) * a;
    return p
end

function cheb_testQ(f)
    cheb_to_realQ(real_to_chebQ(f)) - f;
end

function cheb_interp(a,x)
    # x's are interpolation points in the original interval [-1,1]
    # a,x both are col vec
    N = length(a) - 1;
    θ = acos.(x);
    vals = cos.( θ.* (0:N)' ) * a
    return vals

end

function spec_plot(f)
    coefs = real_to_chebQ(f);
    coefsOrder = log10.(abs.(coefs));
    Plots.plot(coefsOrder, seriestype=:scatter)
    # return Plots.plot(coefsOrder, seriestype=:scatter)
end

function expfilter!(f)
    N = length(f) - 1;
    a = real_to_chebQ(f);
    A = 72;  # = ln(eps(df64"1"))
    p = 16;
    a = exp.( -A * ( (0:Double64(N))/N ).^p ) .* a;
    f = cheb_to_realQ(a);
    return f
end