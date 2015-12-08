import cmath
import math
import matplotlib.pyplot as plt
import pylab;
import math
import cmath
import numpy as np



def conv(x, h):
    '''Linear Convolution'''
    y = np.zeros((len(x) + len(h) -1));
    for i in range(len(y)):
        for k in range(len(h)):
            if ((i-k >= 0) and (i-k <= len(x)-1)):
                y[i] = y[i] + x[i-k] * h[k];
    return(y);

def convlong(x, h, Nfft):
    '''Convolute a long signal segment by segment'''
    Nsegment = Nfft - len(h)
    hseq = np.zeros((Nfft), dtype=h.dtype)
    hseq[:len(h)] = h[:]
    Hseq = fft(hseq)
    yiffted = np.zeros((len(x)+Nfft), dtype=x.dtype)
    for s in range(len(x)//Nsegment + 1):
        xseq = np.zeros((Nfft), dtype=x.dtype)
        if (s+1)*Nsegment <= len(x):
            reallen = Nsegment
        else:
            reallen = len(x) - s*Nsegment
        xseq[:reallen] = x[s*Nsegment : s*Nsegment+reallen]
        Xseq = fft(xseq)
        seqiffted = ifft(Hseq*Xseq)
        l = s*Nsegment
        r = l + len(seqiffted)
        yiffted[l:r] = yiffted[l:r] + seqiffted[:]
    return(yiffted)

def convint(x, h):
    '''Convolution to int result'''
    y = np.zeros((len(x) + len(h) -1), dtype=int);
    for i in range(len(y)):
        acc = 0
        for k in range(len(h)):
            if ((i-k >= 0) and (i-k <= len(x)-1)):
                acc = acc + x[i-k] * h[k];
        y[i] = int(acc)
    return(y); 
      
def cconv(x, h):
    '''Circular Convolution'''
    y = np.zeros((len(x)), dtype=x.dtype)
    for i in range(len(x)):
        for k in range(len(h)):
            y[i] = y[i] + x[(i-k)%len(h)] * h[k]
    return y

def conv2d(x2d, h2d):
    '''2D Convolution, assume x is much bigger than h, h size odd in both dimensions'''
    (Nxr, Nxc) = x2d.shape
    (Nhr, Nhc) = h2d.shape
    Nyr = Nxr
    Nyc = Nxc 
    y2d = np.zeros((Nyr, Nyc), dtype=float)
    for i in range(Nyr):
        for j in range(Nyc):
            y2d[i,j] = 0
            for l in range(Nhr):
                for k in range(Nhc):
                    xrindex = i + Nhr//2 - l
                    xcindex = j + Nhc//2 - k
                    if (xrindex >=0 and xrindex < Nxr and xcindex >= 0 and xcindex < Nxc):
                        y2d[i,j] = y2d[i,j] + h2d[l,k] * x2d[xrindex, xcindex]
    return(y2d);
    
def conv2dsep(x2d, h1d, v1d):
    '''2D Convolution by Separatable Kernel'''
    (Nxr, Nxc) = x2d.shape
    Nh1d = len(h1d)
    Nv1d = len(v1d)
    Nyr = Nxr + Nv1d - 1
    Nyc = Nxc + Nh1d - 1
    y2d = np.zeros((Nyr, Nyc), dtype=float)
    for m in range(Nxr):
        y2d[m, :Nyc] = mm.conv(h1d, x2d[m, :])[:Nyc]
    for n in range(Nyc):
        y2d[:Nyr, n] = mm.conv(v1d, y2d[:, n])[:Nyr]
    return(y2d[Nh1d//2 : Nh1d//2 + Nxr,  Nv1d//2 : Nv1d//2 + Nxc])
    
def real_dft(x):
    '''Real input Discrete Fourier Transform'''
    N = len(x)
    X = np.zeros((N), dtype=complex)
    for m in range(N):
        for n in range(N):
            X[m] = X[m] + x[n] * complex(math.cos(2*math.pi*m*n/N), -1.0*math.sin(2*math.pi*m*n/N))
    return(X)
    
def complex_dft(x):
    '''Complex Input DFT'''
    N = len(x)
    X = np.zeros((N), dtype=complex)
    for m in range(N):
        for n in range(N):
            X[m] = X[m] + x[n] * cmath.exp(-1j*2*math.pi*m*n/N)
    return(X)   

def complex_idft(X):
    '''Complex Input Inverse DFT, Scaled by 1/N'''
    N = len(X)
    x = np.zeros((N), dtype=complex)
    for n in range(N):
        for m in range(N):
            x[n] = x[n] + X[m] * cmath.exp(1j*2*math.pi*m*n/N)
        x[n] = x[n] / N
    return(x)    

def linalg_dft(x):
    '''DFT by Direct Linear Algebra Solver'''
    N = len(x)
    idftmatrix = np.zeros((N,N), dtype=complex)
    for cindex in range(N):
        for rindex in range(N):
            idftmatrix[rindex, cindex] = cmath.exp(1j*2*math.pi*rindex*cindex/N)
    X = np.linalg.solve(idftmatrix, x)
    return(X)

def fft(x):
    '''Radix-2 Fast DFT'''
    N = len(x);
    Half_N = N // 2;
    X = np.zeros((N), dtype=complex);
    if N==1:
        X[0] = x[0]
    else:
        x_even = np.zeros((Half_N), dtype=complex)
        x_odd = np.zeros((Half_N), dtype = complex)
        for i in range(N):
            if i % 2 == 0:
                x_even[ i//2 ] = x[i]
            else:
                x_odd[ (i-1)//2 ] = x[i]
        X_even = fft(x_even)
        X_odd = fft(x_odd)
        for k in range(N):
            X[k] = X_even[k % Half_N] + X_odd[k % Half_N] * cmath.exp(-1j * 2 * math.pi * k / N)
    return X;
   
def unscaled_ifft(X):
    '''Radix-2 Unscaled Inverse FFT'''
    N = len(X);
    Half_N = N // 2;
    x = np.zeros((N), dtype=complex);
    if N==1:
        x[0] = X[0]
    else:
        X_even = np.zeros((Half_N), dtype=complex)
        X_odd = np.zeros((Half_N), dtype = complex)
        for i in range(N):
            if i % 2 == 0:
                X_even[ i//2 ] = X[i]
            else:
                X_odd[ (i-1)//2 ] = X[i]
        x_even = unscaled_ifft(X_even)
        x_odd = unscaled_ifft(X_odd)
        for k in range(N):
            x[k] = x_even[k % Half_N] + x_odd[k % Half_N] * cmath.exp(1j * 2 * math.pi * k / N)
    return x;
    
def ifft(X):
    N = len(X)
    x = unscaled_ifft(X)
    for n in range(N):
        x[n] = x[n] / N
    return x;

def fft2d(f):
    '''2D Forward FFT'''
    (Nr, Nc) = f.shape
    F = np.zeros((Nr, Nc), dtype= complex)
    for m in range(Nr):
        F[m,:] = fft(f[m, :])
    for n in range(Nc):
        F[:, n] = fft(F[:, n])
    return(F)
    
def ifft2d(F):
    '''2D inverse FFT'''
    (Nr, Nc) = F.shape
    f = np.zeros((Nr, Nc), dtype= complex)
    for m in range(Nr):
        f[m,:] = ifft(F[m, :])
    for n in range(Nc):
        f[:, n] = ifft(f[:, n])
    return(f)
    
def dftplot(x, X, figtitle = 'Figure from DFTplot'):
    '''Plot a time domain signal and a frequency domain one'''
    N = len(X);

    Xreal = [0.0] * N;
    Ximag = [0.0] * N;
    Xamplitude = [0.0] * N;
    Xdb = [0.0] * N;
    Xphase = [0.0] * N;

    for i in range(len(X)):
        Xreal[i] = X[i].real;
        Ximag[i] = X[i].imag;
        Xamplitude[i], Xphase[i] = cmath.polar(X[i]);
        if Xamplitude[i] != 0.0:
            Xdb[i] = 20 * cmath.log10(Xamplitude[i])
        if Xamplitude[i] < 1e-10:
            Xphase[i] = 0.0;

   
    # Six axes, returned as a 2-d array
    f, axarr = plt.subplots(3, 2)
    f.suptitle(figtitle)
    axarr[0, 0].plot(x, 'go-')
    axarr[0, 0].set_title('Time domain samples')
    axarr[0, 1].plot(Xdb, 'go-')
    axarr[0, 1].set_title('Magnitude in db')
    axarr[1, 0].plot(Xreal, 'go-')
    axarr[1, 0].set_title('X[m] Real')
    axarr[1, 1].plot(Xamplitude, 'go-')
    axarr[1, 1].set_title('X[m] Amplitude')
    axarr[2, 0].plot(Ximag, 'go-')
    axarr[2, 0].set_title('X[m] Imaginary')
    axarr[2, 1].plot(Xphase, 'go-')
    axarr[2, 1].set_title('X[m] Phase')
    plt.show();
   
def rotate(x, s):
    x = x[s:len(x)] + x[0:s]
    return(x)

def carray_rec2polar(x):
    (Nr, Nc) = x.shape
    xamplitude = np.zeros((Nr, Nc), dtype=float)
    xphase = np.zeros((Nr, Nc), dtype=float)
    for m in range(Nr):
        for n in range(Nc):
            xamplitude[m,n], xphase[m,n] = cmath.polar(x[m,n])
    return(xamplitude, xphase)
    
