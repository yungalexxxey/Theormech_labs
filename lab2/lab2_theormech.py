import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math
def Rot(X,Y,Alpha,XC,YC):
    RX = (X-XC)*np.cos(Alpha) - (Y-YC)*np.sin(Alpha)+XC
    RY = (X-XC)*np.sin(Alpha) + (Y-YC)*np.cos(Alpha)+YC
    return RX, RY
#Ввод переменной t и радиусов необходимых окружностей + ввод угла поворота шариков
t = sp.Symbol('t')
R=1
phi=3*sp.sin(t)
#Параметрические уравнения трёх окружностей. 1. Большая 2.Малая 3. Окружность. по которой движется большой шар.
xc = sp.sin(t)+2
xa = xc-0.05*sp.sin(phi)
ya = 0.9+0.05*sp.cos(phi)
Alpha = sp.cos(6*t)/2
Vx = sp.diff(xc, t)
Vy = 0*t
omega = sp.diff(Alpha, t)
Vxa = Vx-omega*R*sp.cos(Alpha)
Vya = Vy-omega*R*sp.sin(Alpha)
#ВВод и заполнение массивов
T = np.linspace(0, 10, 1000)
XC = np.zeros_like(T)
XA = np.zeros_like(T)
YA = np.zeros_like(T)
ALPHA = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
VXA = np.zeros_like(T)
VYA = np.zeros_like(T)
for i in np.arange(len(T)):
    XC[i] = sp.Subs(xc , t , T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    XA[i] = sp.Subs(xa, t, T[i])
    YA[i] = sp.Subs(ya, t, T[i])
    VXA[i] = sp.Subs(Vxa, t, T[i])
    VYA[i] = sp.Subs(Vya, t, T[i])
    ALPHA[i] = sp.Subs(Alpha, t, T[i])
#Построение графика и подграфика с вырвaниванием осей
fig = plt.figure(figsize=(17, 8))
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
Mayatnik =ax1.plot(XA[0]+0.05*np.cos(Phi), YA[0]+0.1*np.sin(Phi), color='black')[0]
#Доп графики
ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(T, VX)
ax2.set_xlabel('T')
ax2.set_ylabel('VX')
ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, VY)
ax3.set_xlabel('T')
ax3.set_ylabel('VY')
ax4 = fig.add_subplot(4, 2, 6)
ax4.plot(T, VXA)
ax4.set_xlabel('T')
ax4.set_ylabel('Vx mayatnik')
ax5 = fig.add_subplot(4, 2, 8)
ax5.plot(T, VYA)
ax5.set_xlabel('T')
ax5.set_ylabel('Vy mayatnik')
def anima(i):
    NewX = []
    NewY = []
    P.set_data(XC[i], R)
    conline.set_data([0, XC[i]], [R, R])
    Circ.set_data(XC[i]+R*np.cos(Phi), R+R*np.sin(Phi))
    for phi in Phi:
        newx, newy = Rot(XC[i] + 0.15 * math.cos(phi), 0.1 + 0.3 * math.sin(phi) + 0.6, ALPHA[i], XC[i], R)
        NewX.append(newx)
        NewY.append(newy)
    Mayatnik.set_data(NewX, NewY)
    return Circ, P, conline, Mayatnik
anim = FuncAnimation(fig, anima, frames=500, interval=60, blit=True)
plt.show()


