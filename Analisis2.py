#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:27:39 2020

@author: maryi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt
from scipy.optimize import curve_fit

col=['t','x','y']
data=pd.read_csv('data.csv',',', names=col)
x=data['x'].values
y=data['y'].values
t=data['t'].values

N=len(x)

def g(t,a,theta,b,w):
    return a*np.sin(w*np.pi*t+theta) +b
def chi2(data,fit):
    d=np.abs((data-fit)**2/data)
    return np.sum(d)


##Learning to minimize


## dos fits... 
popt12, pcov2 = curve_fit(g, t[:700], x[:700])
popt1, pcov = curve_fit(g, t, x)

mplt.plot(t,x,label='Data')
p1=g(t, *popt1)+ g(t, *popt12)+1
mplt.plot(t, p1,label='Scipy fit')
mplt.legend()
mplt.xlabel('t')
mplt.ylabel('x')

mplt.grid()
mplt.savefig('scipyxfit1.png')
mplt.show()
print(popt12)
print(popt1)
chi=chi2(x,p1)
print('chix=',chi)

popt2, pcov2 = curve_fit(g, t[:700], y[:700])
popt, pcov = curve_fit(g, t, y)

mplt.plot(t,y,label='data')
p2=g(t, *popt)+g(t, *popt2)+1
mplt.plot(t, p2,label='Scipy fit')
mplt.legend()
mplt.xlabel('t')
mplt.ylabel('y')
mplt.grid()
mplt.savefig('scipyyfit1.png')
mplt.show()
print(popt2)
print(popt)
chi=chi2(y,p2)
print('chiy=',chi)
mplt.plot(x,y,label='data')
mplt.plot(g(t, *popt1)+g(t, *popt12)+1,g(t, *popt)+g(t, *popt2)+1)
mplt.legend()
mplt.grid()

mplt.show()

####Otros fit



vchi1=0
#mplt.plot(t[:2000], rec(t[:2000], *popt22)+g(t[:2000], *popt12)+1,label='fit')
def rd1(t,a,b):
    return (a*t+b)
def rd2(t,a,b):
    return (-(a)*t+b)
mplt.plot(t,x,label='data')
for i in range(0,5):
    if i%2==0:
        
        popt22, pcov2 = curve_fit(rd1, t[i*2000:(i+1)*2000], x[i*2000:(i+1)*2000])
        print(popt22)
        ln=rd1(t[i*2000:(i+1)*2000], *popt22)+g(t[i*2000:(i+1)*2000], *popt12)+1
        mplt.plot(t[i*2000:(i+1)*2000],ln,color='r')
        vchi1+=chi2(x[i*2000:(i+1)*2000],ln)
    else:
        
        popt22, pcov2 = curve_fit(rd2, t[i*2000:(i+1)*2000], x[i*2000:(i+1)*2000])
        ln=rd2(t[i*2000:(i+1)*2000], *popt22)+g(t[i*2000:(i+1)*2000], *popt12)+1
        mplt.plot(t[i*2000:(i+1)*2000],ln,color='r')
        vchi1+=chi2(x[i*2000:(i+1)*2000],ln)
        print(popt22)
    #print(popt22)
    #mplt.plot(t[i*2000:(i+2)*2000],rd2(t[i*2000:(i+2)*2000], *popt22,i)+g(t[i*2000:(i+2)*2000], *popt12)+1)

mplt.legend()
mplt.grid()
mplt.xlabel('t')
mplt.ylabel('x')
mplt.savefig('scipy_x_md2.png')
#popt2, pcov2 = curve_fit(g, t[:300], y[:300])
mplt.show()
mplt.plot(t,y,label='data')
vchi2=0
for i in range(0,5):
    if i%2==0:
        
        popt22, pcov2 = curve_fit(rd1, t[i*2000:(i+1)*2000], y[i*2000:(i+1)*2000])
        print(popt22)
        ln=rd1(t[i*2000:(i+1)*2000], *popt22)+g(t[i*2000:(i+1)*2000], *popt2)+1
        mplt.plot(t[i*2000:(i+1)*2000],ln,color='r')
        vchi2+=chi2(y[i*2000:(i+1)*2000],ln)
    else:
        
        popt22, pcov2 = curve_fit(rd2, t[i*2000:(i+1)*2000], y[i*2000:(i+1)*2000])
        ln=rd2(t[i*2000:(i+1)*2000], *popt22)+g(t[i*2000:(i+1)*2000], *popt2)+1
        mplt.plot(t[i*2000:(i+1)*2000],ln,color='r')
        vchi2+=chi2(y[i*2000:(i+1)*2000],ln)
        print(popt22)
    #print(popt22)mplt.plot(t[i*2000:(i+2)*2000],rd2(t[i*2000:(i+2)*2000], *popt22,i)+g(t[i*2000:(i+2)*2000], *popt12)+1)
mplt.legend()
mplt.grid()
mplt.xlabel('t')
mplt.ylabel('y')
mplt.savefig('scipy_y_md2.png')
    
mplt.show()
print('chiy',vchi2 )
N=len(x)
w=[]
for i in range(0,N//700):
    #print(i)
    j=x[i*700:(i+1)*700]
    #print(len(j))
    xf=fftfreq(len(j),t[2]-t[1])
    xf= fftshift(xf)
    yf=fft(j)
    #print(yf)
    yplot=fftshift(yf)
    w.append(1/len(j)*max(np.abs(yplot)))
w=np.array(w)

mplt.plot(t[:700],j)
mplt.show()
a=[-1.23,-1.53]
b=[1.5,2]
c=np.arange(-1  ,1,0.1)
w=[0.95,0.96]
#_,param1=minimizing_chi(a,b,c,w,g,x,t)
a=np.arange(1,3,0.1)
b=np.arange(-6,-4,0.1)
c=np.arange(-1.5,0,0.1)
w=np.arange(15,17,0.1)
#_,param2=minimizing_chi(a,b,c,w,g,x[:700],t[:700])
#mplt.plot(t,x,label='data')
#mplt.plot(t, g(t, *param1)+ g(t, *param2)+1  ,label='fit')
#mplt.legend()
#mplt.grid()
#mplt.show()
def minimizing_chi(a,b,c,d,g,x,t):
    chi=[]
    dat1=[]
    dat2=[]
    for i in a:
        for j in b:
            yt=g(t,i,j,c,d)
            chi.append(chi2(x,yt))
            dat1.append(i)
            dat2.append(j)

    return chi,dat1,dat2
chitn,ap,bp=minimizing_chi(a,b,popt12[-2],popt12[-2],g,x[:700],t[:700])

fig = mplt.figure()
ax=fig.add_subplot(111,projection='3d')
mplt.savefig('surface.png')

#mplt.show()
#
ax.scatter(ap,bp,chitn,c='r',marker='o')
mplt.xlabel('a')
mplt.ylabel('b')
mplt.show()
mplt.plot(t,x,label='y')
for i in range(1,5):
    mplt.vlines(x=i,ymin=-4,ymax=4)
mplt.grid()
mplt.xlabel('t')    
mplt.ylabel('x')
mplt.savefig('xdata.png')
mplt.show()