# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   jupytext_formats: ipynb,py
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 2
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython2
#     version: 2.7.14
# ---

import matplotlib.pyplot as plt
# %matplotlib inline
import math
import numpy as np
import time as ttime

# %autosave 0

# +
srate = 50
ampl = 2
freq = 4
phas = 75
time = np.arange(0,2,1/srate)
sw = ampl*np.sin(2*np.pi*freq*time + phas)

plt.plot(time,sw)

# +
# Create a signal
srate=500
time=np.arange(-1.,1.,1./500)
sw = 2*np.sin(2*np.pi*5*time)

plt.plot(time,sw)
gau_sig = np.exp((-time**2)/0.1)
plt.plot(time,gau_sig)

signal = np.multiply(sw,gau_sig)
# -

print(len(time))

plt.plot(time,signal)

sinefrex = np.arange(2.,10.,.5)
print(len(sinefrex),sinefrex)

dps = np.zeros(len(sinefrex))
for fi in range(1,len(sinefrex)):
    #print(time)
    sinew = np.sin(2*np.pi*sinefrex[fi]*time)
    dps[fi] = np.dot(sinew,signal)/len(time)

plt.stem(sinefrex,dps)

# Excercise Solutions

# Write python code to generate and plot two-second signals that comprise a sum of the following
# sine wave parameters. Test various sampling rates, ranging from 1 Hz to 1000 Hz, to determine
# what—if any—effect that has on the plots.
# a)
# f = 2 p = 0 a = 1
# f = 4.2 p = 3pi/4 a = 1.7
# b)
# f = 200 p = 0 a = 100
# f = 402 p = 0 a = 10
# f = 3.2 p = 1 a = 50

srate = 20
t=np.arange(0,2,1/srate)

s1=1  *np.sin(2*np.pi*2  *t + 0)
s2=1.7*np.sin(2*np.pi*4.2*t + 0)
s12 = s1+s2

plt.subplot(2,2,1)
plt.plot(t,s1)
plt.subplot(2,2,2)
plt.plot(t,s2)
plt.subplot(2,2,3)
plt.plot(t,s12)


# Computing Fourier Coeff for a sample signal

file='/Users/jm186072/Documents/DataScience/Udemy Courses/Fourier Transform/Alesis-Fusion-Clean-Guitar-C3.wav'

# +
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from scipy.io.wavfile import write
spf = wave.open(file,'r')
signal = spf.readframes(-1)
signal = np.fromstring(signal,'Int16')
print('numpy signal shape',signal.shape)

plt.plot(signal)
plt.title("Sound Signal without Echo")
plt.show()
# -

import wave
import sys
from scipy.io.wavfile import write
from scipy.io.wavfile import read as read_wav

# +
sampling_rate, signal=read_wav(file)

print('numpy signal shape',signal.shape, "Sampling Rate:=",sampling_rate)

plt.title("Original Signal")

plt.plot(signal)
plt.show()


# +
## The DTFT in loop-form



# +



# -

print(len(signal),len(fourTime))

fourTime.shape


# +
# create the signal
srate  = 44100 # hz
time   = np.arange(0.,8.92438999,1/srate) # time vector in seconds
pnts   = len(time) # number of time points
#signal = 2.5 * np.sin( 2*np.pi*4*time ) + 1.5 * np.sin( 2*np.pi*6.5*time )
signal = signal/2

# prepare the Fourier transform
fourTime = np.array(range(0,pnts))/pnts
fCoefs   = np.zeros((len(signal)),dtype=complex)


# +
for fi in range(0,pnts):
    
    # create complex sine wave
    csw = np.exp( -1j*2*np.pi*fi*fourTime )
    print(fi)
    # compute dot product between sine wave and signal
    # these are called the Fourier coefficients
    fCoefs[fi] = np.sum( np.multiply(signal,csw) ) / pnts
    
    
# -



# extract amplitudes
ampls = 2*np.abs(fCoefs)

# compute frequencies vector
hz = np.linspace(0,srate/2,num=math.floor(pnts/2.)+1)

plt.stem(hz,ampls[range(0,len(hz))])
plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude (a.u.)')
#plt.xlim(0,10)
#plt.show()


len(ampls)

ampls

len(hz)

ampls[0]

ampls.max()

np.argmax(ampls)

ampls[583]

hz[693]

plt.stem(hz[80:180],ampls[range(80,180)])
plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude (a.u.)')
plt.xlim(9,21)


