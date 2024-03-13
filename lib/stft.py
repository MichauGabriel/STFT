# -*- coding: utf-8 -*-
"""
Created on Mon Feb 5 14:10:16 2016

@author: Gabriel Michau
"""

import numpy as np
import matplotlib.pyplot as plt

 # Fourier Transform
def norm(S):
    """
    Compute value to normalise the spectrogram

    Parameters
    ----------
    S : result of stft

    Returns
    -------
    norm a value to normalise the spectrogram.

    """
    return np.max(np.max(np.power(S,2)))

def conv2db(S,norm):
    """
    Conversion of the Fourier coefficients values to normalised dB

    Parameters
    ----------
    S : result of stft
    norm : result of norm(S) or arbitrarily chosen

    Returns
    -------
    S: normalised Froueri coefficient in dB

    """
    return 10*np.log10(np.power(S,2)/norm)

def stft(x,dt,wlen,hop,nfft=None,window='hamming',spectrum="single"):
    """
    Generate the short-time Fourier transform of the signal x with sampling rate dt [s], over a window of size wlen (in samples),
    and of type 'hamming' or 'rectangle', for a window ever 'hop' samples.

    Parameters
    ----------
    x : 1D array with signals on which to perform the stft 
    dt : FLOAT. Sample rate (in [s])
    wlen : INT. window length for the Fourier transform (number of samples)
    hop : INT. Gap between windows in number of samples
    nfft : INT. Number of Fourier coefficients to compute (default wlen for symmetricla spectrum)
    window : STRING. window type. The default is 'hamming', can be set to rectangle, other window types can also be implemented.
    spectrum : STRING. Single to get real valued, single sided spectrum, anything else to get the complex valued spectrum
         The default is "single".

    Returns
    -------
    S: 2D array with the resulting stft.
    t: the timestamp of each column in S
    f: the frequency center of each row in S 

    """
    if window.lower() != 'hamming' and window.lower() != 'rectangle':
        raise TypeError("Error window type not implemented. Use 'hamming' or 'rectangle'.")
    if spectrum.lower() != 'single' and spectrum.lower() != 'complex':
        raise TypeError("Spectrum not understood.\nSpectrum can only be 'single' for single sided real spectrum or 'complex'.")
    if nfft is None or nfft<=0:
        nfft = wlen
    if window.lower() == 'hamming':
        conv = np.hamming(wlen)
    else:
        conv = np.ones(wlen)

    xlen = x.__len__()
    lentout = np.int64(1+np.floor((xlen-wlen)/hop))
    lenfout = np.ceil(nfft/2+0.1).astype(int)

    max_freq = 1.0/(2.0*dt)             # max frequency identifiable with FFT
    step = max_freq/(lenfout-1)         # frequency step in FFT output (around 11Hz/step for windows 99)

    t = np.arange(np.floor(wlen/2),np.floor(wlen/2)+(lentout)*hop, hop)*dt
    f = np.arange(0, lenfout*step,step)

    S = np.zeros((lenfout,lentout),dtype=np.complex_)
    indx = 0
    for ii in range(lentout):
        S[:,ii] = np.fft.rfft(np.multiply(x[indx:indx+wlen],conv),nfft)#[0:lenfout]
        indx    = indx + hop
    if spectrum=="single":
        return 2.0/wlen * np.abs(S), t, f
    else:
        return S, t, f

def spectroDB(S, factor=0.75, normS=None, thr=None):
    """
    Convert the spectrogram into dB relavtive to NormS and cut off very low negative coefficients so that 
    when plotting the final color map, it has contrast where the coefficients have menaingfull values
    If doing several spectrograms => Compute the threshold once and use the same threshold for all spectrogamms.
    Otherwise, you don't have to specify the threshold (thr), just the contrast factor.

    Parameters
    ----------
    S : 2D array with the resulting stft.
    factor : FLOAT
        Used to compute the threshold under which coefficients are shut down (set to min value). All coeeficient below factor*(avg(S)-std(S)) will appear with same value.
        The default is 0.75.
    normS : FLOAT, optional
        Value to be used to normalise the spectrogram. The default is None, in which case the function norm will be used
    thr : FLOAT, optional
        Threshold underwhich Fourier coeeficients will be set to min value. The default is None, in which case
        the factor will be used as explained above.

    Returns
    -------
    Sdb : ARRAY of same size as S but in db relative to normS.
    thr : FLOAT. The computed threshold on the Fourier Coefficients

    """
    if normS is None:
        normS = norm(np.abs(S))
    Sdb = conv2db(np.abs(S),normS) # | | even if squared as S might be complex
    if thr is None:
        thr = factor*(np.nanmean(Sdb[:])-np.nanstd(Sdb[:]))
    Sdb[Sdb<thr]=thr
    return Sdb,thr

def spectroplot(Sdb,t,f,fig=None):
    """
    Helper to plot the db spectrogram

    Parameters
    ----------
    Sdb : 2D ARRAY
        DESCRIPTION.
    t : 1D array
        the timestamp of each column in S 
    f : 1D array
        the frequency center of each row in S.
    fig : Matplotlib figure, optional
        If a specific figure object already existing should be used. The default is None, in which case a new figure object is created.

    Returns
    -------
    fig : matplotlib figure

    """
    if not fig:
        fig = plt.figure()
    ax=fig.gca()
    im=ax.imshow(Sdb[::-1,:], interpolation='none',extent=[0,t[-1],0,f[-1]], cmap=plt.cm.jet)
    ax.set_position([0.1,0.1,0.75,0.8])
    ax.set_aspect('auto', adjustable='box')
    box = ax.get_position()
    cax = fig.add_axes([box.x0+1.02*box.width,box.y0, 0.05*(box.width),box.height])
    fig.colorbar(im, cax=cax, label='db/Hz')
    ax.set_ylabel('Hz')
    ax.set_xlabel('Seconds')
    return fig

def istft(S, dt,  wlen, hop, nfft=None, window='hamming'):
    """ Perform ISTFT (via IFFT and Weighted-OLA)"""
    if window.lower() != 'hamming' and window.lower() != 'rectangle':
        raise TypeError("Error window type not implemented. Use 'hamming' or 'rectangle'.")
    if not np.iscomplexobj(S):
        go = input("The input spectrum is not complex. Go on anyway (y/n)?")
        if go.lower() != 'y':
            print("Programm aborted")
            return

    if nfft is None or nfft<=0:
        nfft = wlen
    coln = S.shape[1]
    xlen = wlen + (coln-1)*hop
    x = np.zeros(xlen)
    idx = 0
    if window.lower() == 'hamming':
        conv = np.hamming(wlen)
    else:
        conv = np.ones(wlen)

    if nfft % 2 == 1: # odd nfft excludes Nyquist point
        for col in range(coln):
            X = np.append(S[:,col],S[-1:0:-1,col].conjugate())
            X = np.real(np.fft.irfft(X,wlen))
            # weighted-OLA
            x[idx:(idx+wlen)] = x[idx:(idx+wlen)] + np.multiply(X,conv).T
            idx += hop
    else:
        for col in range(coln):
            X = np.append(S[:,col],S[-2:0:-1,col].conjugate())
            # even nfft includes Nyquist point
            X = np.real(np.fft.irfft(X,wlen))#[0:wlen])
            # weighted-OLA
            x[idx:(idx+wlen)] = x[idx:(idx+wlen)] + np.multiply(X,conv).T
            idx += hop
    # Scaling
    W = np.power(conv,2).sum()
    x = np.multiply(x,hop/W)
    t = np.arange(0,xlen)*dt

    return x,t

def stfft(x,ids,hamming=False,nfft=-1):
    """
    Function to return the value of the highest Fourier coefficient among some ids specified in ids for a signal x

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    ids : TYPE
        DESCRIPTION.
    hamming : TYPE, optional
        DESCRIPTION. The default is False.
    nfft : TYPE, optional
        DESCRIPTION. The default is -1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if nfft==-1:
        nfft = x.__len__()
    # select among ids the one with maximal value for cefficient
    # multiply by 2/N to get true coeff value (amplitude)
    if hamming:
        return 2.0/x.__len__() * np.abs(np.fft.rfft(x*np.hamming(x.__len__()),nfft)[ids[0]:ids[1]]).max()
    else:
        return 2.0/x.__len__() * np.abs(np.fft.rfft(x,nfft)[ids[0]:ids[1]]).max()
