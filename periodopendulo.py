import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar

#parte 1

def t(theta, theta0):
    return 1 / (np.cos(theta) - np.cos(theta0))


def calcular(angulo):
    angulo = np.deg2rad(angulo)
    x      = np.linspace(0, angulo, 1000, endpoint=False)
    y      = t(x, angulo)
    return x, y


x1, y1 = calcular(1)
plt.plot(x1, y1, color="red")

x90, y90 = calcular(90)
plt.plot(x90, y90, color="green")

x179, y179 = calcular(179)
plt.plot(x179, y179, color="blue")

plt.legend([r'$\theta_{0}=1^\circ$', r'$\theta_{0}=90^\circ$', r'$\theta_{0}=179^\circ$'], loc=9)
plt.xlabel(r'$\theta(rads)$')
plt.ylabel(r"$t(\theta)$")
plt.ylim(0, 10000)
plt.show()

'''
Com a ajuda do gráfico podemos perceber que quando a função é crescente e a concavidade para cima a regra do trapézio
excede a estimativa do resultado, por isso é melhor utilizar outras regras de integração que possuem uma variação de 
erro menor.
'''

#parte 2

def func_integrate(phi, theta0):
    return 1 / (np.sqrt(1 - (np.sin(theta0 / 2) ** 2) * np.sin(phi) ** 2))


def deltaT(theta0):
    t0 = 2. * np.pi
    t_theta0 = 4. * quad(func_integrate, 0, np.pi / 2, args=theta0)[0]
    dt = 100. * np.abs((t0 - t_theta0) / t_theta0)
    return dt


def func(theta0):
    return deltaT(theta0) - 1


sol = root_scalar(func, bracket=[0, 179.9], method='brentq')
print(f'{sol.root} rad')

vet = np.vectorize(deltaT)
lim_x = np.deg2rad(179.9)
x = np.linspace(0.0, lim_x, 1000)
y = vet(x)

plt.plot(x, y)
plt.xlabel(r'$\theta_{0}(rads)$')
plt.ylabel(r'$\Delta T(\%)$')
plt.show()
