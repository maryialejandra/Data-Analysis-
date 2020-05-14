import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt
from scipy.optimize import curve_fit
from scipy.fftpack import fft, ifft,fftfreq, fftshift
col=['t','x','y']
data=pd.read_csv('data.csv',',', names=col)
x=data['x'].values
y=data['y'].values
t=data['t'].values
mplt.plot(t,x)
mplt.xlabel('t')
mplt.ylabel('x')
mplt.grid()
mplt.show()
mplt.plot(t,y)
mplt.xlabel('t')
mplt.ylabel('y')
mplt.grid()
mplt.show()

N=len(y)
xf=fftfreq(N,t[2]-t[1])
xf= fftshift(xf)
yf=fft(y)
yplot=fftshift(yf)

mplt.plot(xf[5000:],(1/N*np.abs(yplot[5000:])))
mplt.show()
d=max(1/N*np.abs(yplot))

def f(t,b,w1,a):
    return a*np.sin(w1*t)+(b*np.sin(d*np.pi*t))#+np.sin(10*t))
def g(t,a,w,theta,b):
    return a*np.sin(w*t+theta)+b
#mplt.plot(t,f(t))
#mplt.show()
#mplt.plot(t,g(t))
#mplt.show()


l=0.44530445
popt, pcov = curve_fit(f, t, x)
mplt.plot(t,x,label='data')
mplt.plot(t, f(t, *popt),label='fit')
mplt.plot(t,4*np.sin(0.8*np.pi*t)*np.cos(10*2*np.pi*t)**2,label='frequency')
mplt.legend()
mplt.grid()
mplt.show()
#mplt.plot(t[:1000],x[:1000],label='data')
mplt.plot(t[:1000],x[:1000],label='data')
#mplt.plot(t, f(t, *popt),label='fit')

mplt.plot(t[:1000],3*np.sin(l*2*np.pi*t[:1000]-0.9)+2*np.sin(6.3*2*np.pi*t[:1000]-1.3)-0.1,label='frequency')
mplt.legend()
mplt.grid()
mplt.show()


#t=np.linspace(0,10,200)
#y=np.sin(4*2*np.pi*t)
#N=len(y)
#xf=fftfreq(N,t[2]-t[1])
#xf= fftshift(xf)
#yf=fft(y)
#yplot=fftshift(yf)

#mplt.plot(xf,(1/N*np.abs(yplot)))
#mplt.show()
#print(max(1/N*np.abs(yplot)))
