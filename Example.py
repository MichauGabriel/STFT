# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:17:40 2024

@author: Gabriel Michau
"""

import numpy as np
from lib import stft 

import matplotlib.pyplot as plt

## Lets generate a speed signal with a speed dependent vibration component

## Start speed
f0 = 0 
## End speed
f1 = 100
## Length of the segments
l = 100000
## Sampling rate in [s]
dt = 1/2000 # sampling rate in s

## Generate the speed signal, with constant acceleration to f1, constant speed and then constant decceleration
x = np.concatenate((np.linspace(f0,f1,l),f1*np.ones(l),np.linspace(f1,f0,l)))

## Generate the phase of the vibration component
phase = np.cumsum(2*np.pi*x*dt)

## Generate final signal as speed + vibration at frequency == speed
y = 2*x + np.sin(phase)

## You can check the result by plotting the signal
plt.plot(y)

## Let's make the short time Frouier transfrom. 
S, t, f = stft.stft(y,dt,wlen=1000,hop=1000//2,nfft=None,window='hamming',spectrum="single")

## Let's compute the norm of this spectrogram (if doing several spectrograms, this norm should be chosen once and used for all the spectrograms)
norm = stft.norm(S)
## Convert the spectrogram into dB and cut off very low negative coefficients so that the final color map has contrast where the coefficient have menaingfull values
## If doing several spectrograms => Compute the threshold once and use the same threshold for all spectrogamms.
## Otherwise, you don't have to specify the threshold (thr), just the contrast factor 
Sdb,thr = stft.spectroDB(S, factor=0.75, normS=norm, thr=None)

## Get the plot
fig = stft.spectroplot(Sdb,t,f,fig=None)
