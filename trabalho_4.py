import matplotlib.pyplot as plt
import numpy as np
import math as mt
from scipy.integrate import solve_ivp

#----------------------------------------------------Questao 1----------------------------------------------------------

def energia(w1, w2, theta1, theta2, m = 1, l = 10, g = 9.8):
    theta1 = mt.radians(theta1)
    theta2 = mt.radians(theta2)
    num1 = m * l
    num2 = w1 ** 2 + w2 ** 2 / 2 + w1 * w2 * mt.cos(theta1 - theta2)
    num3 = m * g * (l * ((2 * mt.cos(theta1)) + mt.cos(theta2)))
    return num1 * (num2) - num3


w1 = 0
w2 = 0
theta1 = 90
theta2 = 90

E = energia(w1, w2, theta1, theta2)
print('Questao 1\n')
print(f'Energia: {E:.2f}\n')

#---------------------------------------------------Questao 2-----------------------------------------------------------


def energia2(params, m, l, g):
    theta1 = params[0]
    theta2 = params[1]
    w1 = params[2]
    w2 = params[3]

    num1 = m * l
    num2 = w1 ** 2 + w2 ** 2 / 2 + w1 * w2 * np.cos(theta1 - theta2)
    num3 = m * g * (l * ((2 * np.cos(theta1)) + np.cos(theta2)))
    return num1 * (num2) - num3


def derivada_w1(w1, w2, theta1, theta2, g, l):
    num1 =  (w1**2) * np.sin((2*theta1)-(2*theta2))
    num2 = 2 * (w2 **2) * np.sin(theta1 - theta2)
    num3 = (g/l)*((np.sin(theta1 - 2* theta2)) + 3*np.sin(theta1))
    den = 3 - np.cos((2*theta1) - (2*theta2))
    return -(num1 + num2 + num3)/den


def derivada_w2(w1, w2, theta1, theta2, g, l):
    num1 = 4 * w1**2 * np.sin(theta1 - theta2)
    num2 = w2**2* np.sin(2*theta1 - 2*theta2)
    num3 = 2*(g/l)*(np.sin(2*theta1-theta2) - np.sin(theta2))
    den = 3 - np.cos(2*theta1 - 2*theta2)
    return (num1 + num2 + num3)/den


def func(t, params, l, g):
    theta1 = params[0]
    theta2 = params[1]
    w1     = params[2]
    w2     = params[3]

    w1_dt = derivada_w1(w1, w2, theta1, theta2, g, l)
    w2_dt = derivada_w2(w1, w2, theta1, theta2, g, l)
    return [w1, w2, w1_dt, w2_dt]


m  = 1
l  = 1
g  = 9.8
w1 = 0
w2 = 0
theta1 = np.deg2rad(90)
theta2 = np.deg2rad(90)

t = np.arange(0, 1001, 1)
t_func = (t.min(), t.max())
params = np.array([theta1, theta2, w1, w2])

print('Questao 2\n\nMetodo RK45\n')

sol1 = solve_ivp(func, t_span=t_func, y0=params, method='RK45', t_eval=t, rtol=1.e-10, atol=1.e-10, args=(l, g))
print(f'Duas ultimas posicoes angulares (theta 1): {sol1.y[0][-2:]}')
print(f'Duas ultimas posicoes angulares (theta 2): {sol1.y[1][-2:]}')

E1 = energia2(sol1.y, m, l, g)
print(f'Energia: {E1}\n')

plt.plot(t, E1)
plt.title('Método RK45')
plt.xlabel("Tempo(s)")
plt.ylabel("Energia Mecânica(J)")
plt.show()

print('Metodo DOP853\n')

sol2 = solve_ivp(func, t_span=t_func, y0=params, method='DOP853', t_eval=t, rtol=1.e-10, atol=1.e-10, args=(l, g))
print(f'Duas ultimas posicoes angulares (theta 1): {sol2.y[0][-2:]}')
print(f'Duas ultimas posicoes angulares (theta 2): {sol2.y[1][-2:]}')

E2 = energia2(sol2.y, m, l, g)
print(f'Energia: {E2}\n')

plt.plot(t, E2)
plt.title('Método DOP853')
plt.xlabel("Tempo(s)")
plt.ylabel("Energia Mecânica(J)")
plt.show()

'''
A energia não é conservada. O método mais acurado é DOP853, pois a ordem dele é maior, com isso, obtém resultados mais 
acurados.
'''

#---------------------------------------------------Questao 3-----------------------------------------------------------

'''
O que o Dan errou no código foi a fórmula na equação em que ele corrigiu utilizando parênteses e também a troca das duas
linhas de código de aceleração e velocidade dos ângulos em que parece que o erro foi consertado ao mudar a ordem da 
aceleração e velocidade para velocidade e aceleração do ângulo. Não sei explicar o que ocorreu, mas depois de ver os
comentários eu consegui entender que da maneira incorreta ele estava colocando mais energia no sistema por conta do erro
não ser limitado pelo método de Euler que coloca a posição como primeiro. Se colocar a velocidade primeiro é o método de
Verlet que tem o erro limitado, com isso a energia flutua dentro do parâmetro, entendi pela explicação do comentário de 
5_inch C. Dan não estava errado inicialmente, pois ele escolheu um método que não serve para aquilo que ele queria 
apresentar. Eu não sei dizer o método utilizado pelo Dan, mas de acordo com os comentários depois que ele corrigiu 
parece que é o método de Verlet.
'''