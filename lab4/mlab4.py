import numpy as np
#Note that the next import is required for Matplotlib versions before 3.2.0. 
#For versions 3.2.0 and higher, you can plot 3D plots without importing
#from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import sympy as sp
import math

from sympy.series import O

def Rot(X,Y,Alpha,XC,YC):
    RX = (X-XC)*np.cos(Alpha) - (Y-YC)*np.sin(Alpha)+XC
    RY = (X-XC)*np.sin(Alpha) + (Y-YC)*np.cos(Alpha)+YC
    return RX, RY
def formY(y, t, fV, fOm):
    y1,y2,y3,y4 = y   
    dydt = [y3,y4,fV(y1,y2,y3,y4),fOm(y1,y2,y3,y4)]
    return dydt
def formY2(y, t, fOm):
    y1,y2 = y   
    dydt = [y2,fOm(y1,y2)]
    return dydt

# size of the prism
R=1
#size of the beam 
r = 0.1
#masses of the prism and the beam 
m1 = 5
m2 = 1
g = 9.81
# coefficients

c=0
#defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')

#defining s, phi, V=ds/dt and om=dphi/dt as functions of 't'
s=0;
alpha=sp.Function('alpha')(t)
V=0
om=sp.Function('om')(t)


#constructing the Lagrange equations
#1 defining the kinetic energy
TTR = m1*V**2
#The squared velocity of the center of mass
# Vc2 = V**2+(om**2)*(r/2**2)/4-V*om*r/2*sp.cos(alpha)
Vc2 = m2*V**2/2-V*om*r/2*sp.cos(alpha)
#The moment of inertia
Jc = (m2*r**2)/12
TTr = (m2*Vc2)/2+(Jc*om**2)/2
TT = TTR+TTr
#2 defining the potential energy
Pi1 = -m1*g
Pi2 = m2*g*(-R/2+r*(sp.cos(alpha)))
Pi3=(c*(s-1)**2/2)
Pi = Pi1+Pi2+Pi3
M = -m1*g*sp.cos(alpha);

#Lagrange function
L = TT-Pi

#equations
#ur1 = sp.diff(sp.diff(L,V),t)-sp.diff(L,s)
ur2 = sp.diff(sp.diff(L,om),t)-sp.diff(L,alpha)-M
print(ur2);

#isolating second derivatives(dV/dt and dom/dt) using Kramer's method
# a11 = ur1.coeff(sp.diff(V,t),1)
# a12 = ur1.coeff(sp.diff(om,t),1)
# a21 = ur2.coeff(sp.diff(V,t),1)
a22 = ur2.coeff(sp.diff(om,t),1)
# b1 = -(ur1.coeff(sp.diff(V,t),0)).coeff(sp.diff(om,t),0).subs([(sp.diff(s,t),V), (sp.diff(alpha,t), om)])
b2 = -ur2.coeff(sp.diff(om,t),0).subs(sp.diff(alpha,t), om);

# detA = a11*a22-a12*a21
# detA1 = b1*a22-b2*a21
# detA2 = a11*b2-b1*a21

# dVdt = detA1/detA
domdt = b2/a22
print(domdt)

countOfFrames = 200

# Constructing the system of differential equations
T = np.linspace(0, 12, countOfFrames)
# Pay attention here, the function lambdify translate function from the sympy to numpy and then form arrays much more
# faster then we did using subs in previous lessons!
# fV = sp.lambdify([s,alpha,V,om], dVdt, "numpy")
fOm = sp.lambdify([alpha,om], domdt, "numpy")
y0 = [1.5, 0]
sol = odeint(formY2, y0, T, args = (fOm,))

#sol - our solution
#sol[:,0] - s
#sol[:,1] - phi
#sol[:,2] - ds/dt
#sol[:,3] - dphi/dt

#constructing functions

#Motion of the prism with a spring (translational motion)
# XsprS = sp.lambdify(s, 2+s)
# YsprS = sp.lambdify(1,1)

#Motion of the beam with respect to a spring (A - the farthest point on the beam from the spring)
xASPhi = sp.lambdify(alpha, -r/2*sp.sin(alpha))
yASPhi = sp.lambdify(alpha, 0.9-r/2*sp.cos(alpha))

#constructing corresponding arrays (Very fast, thanks to lambdify)
XC = np.zeros(countOfFrames)

Alpha = sol[:,0]
XA = xASPhi(sol[:,0])
YA = yASPhi(sol[:,0])

#This region remains the same
#here we start to plot
fig = plt.figure(figsize=(100,100))
ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')

#построение линий ОСИ
liney, = ax1.plot([0, 0], [0, 5], 'black')
linex, = ax1.plot([0, 5], [0, 0], 'black')
ArrowX = np.array([-0.2, 0, -0.2])
ArrowY = np.array([0.1, 0, -0.1])
ArrowOY, = ax1.plot(ArrowY, ArrowX+5, 'black')
ArrowOX, = ax1.plot(ArrowX+5, ArrowY, 'black')
#построение соединяющей линии
conline,=ax1.plot([0, XC[0]], [R, R], 'black')

#инфа про движения
P, = ax1.plot(XC[0], R , marker='o', color='black')
#Большая окружность
Phi = np.linspace(0, 2*math.pi, 20)
Circ, = ax1.plot(XC[0]+R*np.cos(Phi), R+R*np.sin(Phi), 'black')
#физический маятний
Mayatnik =ax1.plot(XA[0]+r/2*np.cos(Phi), YA[0]+r*np.sin(Phi), color='black')[0]
#Доп графики
ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(T, sol[:,0])

ax2.set_xlabel('T')
ax2.set_ylabel('V')

ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, sol[:,1])

ax3.set_xlabel('T')
ax3.set_ylabel('Om')

plt.subplots_adjust(wspace=0.3, hspace = 0.7)

def anima(i):
    NewX = []
    NewY = []
    P.set_data(XC[i], R)
    conline.set_data([0, XC[i]], [R, R])
    Circ.set_data(XC[i]+R*np.cos(Phi), R+R*np.sin(Phi))
    for phi in Phi:
        newx, newy = Rot(XC[i] + 0.15 * math.cos(phi), 0.1 + 0.3 * math.sin(phi) + 0.6, (Alpha[i]+60)/2, XC[i], R)
        NewX.append(newx)
        NewY.append(newy)
    Mayatnik.set_data(NewX, NewY)
    return Circ, P, conline, Mayatnik
anim = FuncAnimation(fig, anima, frames=countOfFrames, interval=100, blit=True)
plt.show()

