import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc
from scipy import optimize

# Jéssica Beatriz e Savio Carvalho


def posForm(f0, gamma, var, x):
    pos = (lambda t:(((f0/gamma)-var)*(t+(np.exp(-gamma*t)-1)/gamma)))
    inverPos = inversefunc(pos)
    return inverPos(x)


def veloForm(f0, gamma, var, t):
    return ((f0/gamma)-var)*(1-np.exp(-gamma*t))


def xQuad(params, t, x, t_reaction, v_ar):
    f0 = params[0]
    gamma = params[1]
    t_model = posForm(f0, gamma, v_ar, x) + t_reaction
    dt = (t - t_model) ** 2
    return np.sum(dt)


def paramBolt(t, x, x0, t_reaction,  v_ar):
    res = optimize.minimize(xQuad, x0, args=(t, x, t_reaction, v_ar), method='Nelder-Mead', tol=1e-8)
    return res.fun, res.x[0], res.x[1]


def veloCansadoForm(a, b, c, t):
    return a * (1 - np.exp(-c * t)) - (b * t)


def posCansadoForm(a, b, c, x):
    posCansado      = (lambda t: (a * t) - (b * t * t / 2) - ((a / c) * (1 - np.exp(-c * t))))
    inverPosCansado = inversefunc(posCansado)
    return inverPosCansado(x)


def xQuadCansado(params, tBeijing,tBerlin, x):
    f0      = params[0]
    gamma   = params[1]
    cansado = params[2]

    t_model_beijing = posCansadoForm(f0, gamma, cansado, x)
    t_model_berlin  = posCansadoForm(f0, gamma, cansado, x)

    dt1 = (tBeijing- t_model_beijing) ** 2
    dt2 = (tBerlin - t_model_berlin) ** 2

    soma1 = np.sum(dt1)
    soma2 = np.sum(dt2)

    return soma1+soma2/len(tBeijing)+len(tBerlin)-3


def paramCorredor(t1, t2, x, x0):
    res = optimize.minimize(xQuadCansado, x0, args=(t1, t2, x), method='Nelder-Mead', tol=1e-8)
    return res.x[0], res.x[1], res.x[2]


#Questão 1
x     = np.linspace(10, 100, 10)
f0    = 8.466
gamma = 0.686
v_ar  = 0.

print('Questao 1:\n')
t = posForm(f0, gamma, v_ar, x)
print('Tempo na posição 100 sem velocidade do ar:', t[-1],"s")

v_ar = 2
t_ar = posForm(f0, gamma, v_ar, x)
print('Tempo na posição 100 com velocidade do ar:', t_ar[-1],"s", '\n')

v_sem_var = veloForm(f0, gamma, 0, t)
v_com_var = veloForm(f0, gamma, 2, t_ar)

plt.plot(x, t, color="red")
plt.plot(x, t_ar, color="blue")
plt.title('Questão 1')
plt.legend([r'Sem Ar', r'Com Ar'], loc=9)
plt.xlabel(r'$Posição(m)$')
plt.ylabel(r"$Tempo(s)$")
plt.show()

plt.plot(x, v_sem_var, color="red")
plt.plot(x, v_com_var, color="blue")
plt.title('Questão 1')
plt.legend([r'Sem Ar', r'Com Ar'], loc='lower right')
plt.xlabel(r'$Posição(m)$')
plt.ylabel(r"$Velocidade(\frac{m}{s})$")
plt.show()

#Questão 2
print('Questao 2:\n')
xBolt = np.linspace(10., 100., 10)
tBolt = np.array([1.88, 2.88, 3.78, 4.64, 5.47, 6.29, 7.10, 7.92, 8.74, 9.58])
chute = np.array([10, 1])
tReaction = 0.146

min, f0, gamma = paramBolt(tBolt, xBolt, chute, tReaction, 0)
print('Minimo da funcao chi-quadrado sem velocidade do ar =', min, '\nParametros sem velocidade do ar: F0 = ', f0, 'm/s^2', 'gamma = ', gamma,'s^-1')

minAr, f0Ar, gammaAr = paramBolt(tBolt, xBolt, chute, tReaction, 2)
print('Minimo da funcao chi-quadrado com velocidade do ar =', minAr, '\nParametros com velocidade do ar: F0 = ', f0Ar, 'm/s^2', 'gamma = ', gammaAr,'s^-1', '\n')

'''
O modelo teórico baseia-se na percepção de padrões a partir dos dados recebidos
no experimento. Nesse experimento foi utilizado o método dos mínimos quadrados que
funciona quando os dados dependem linearmente dos parâmetros a serem ajustados, como
esse método depende dos parâmetros, não conseguimos chegar no resultado ideal e sim 
um próximo do que desejamos, como por exemplo, faltou levar em conta o cansaço e a massa
que consideramos com 1kg.Com isso, percebemos que esse modelo teórico não é ideal para 
analisar um corredor de 100m rasos.

'''

#Questão 3
print('Questao 3:\n')
x = np.linspace(10, 100, 10)
chute1 = np.array([15.50, 0.1, 1.192])
chute2 = np.array([12, 0.08, 0.8])
chute3 = np.array([9.45, 0.013, 0.529])


tBeijing = np.array([1.85, 2.87, 3.78, 4.65, 5.50, 6.32, 7.14, 7.96, 8.79, 9.69])
tReactionBeijing = 0.165
tBerlin = np.array([1.88, 2.88, 3.78, 4.64, 5.47, 6.29, 7.10, 7.92, 8.74, 9.58])
tReactionBerlin = 0.146

tempoCansado1 = posCansadoForm(chute1[0], chute1[1], chute1[2], x)
tempoCansado2 = posCansadoForm(chute2[0], chute2[1], chute2[2], x)
tempoCansado3 = posCansadoForm(chute3[0], chute3[1], chute3[2], x)

a, b, c = paramCorredor(tBeijing, tBerlin, x, chute1)
print ("Melhor ajuste: a = %2.3f m/s, b = %2.3f m/s^2 e c = %2.3f/s " % (a, b, c))

a, b, c = paramCorredor(tBeijing, tBerlin, x, chute2)
print("Melhor ajuste: a = %2.3f m/s, b = %2.3f m/s^2 e c = %2.3f/s " % (a, b, c))

a, b, c = paramCorredor(tBeijing, tBerlin, x, chute3)
print("Melhor ajuste: a = %2.3f m/s, b = %2.3f m/s^2 e c = %2.3f/s\n " % (a, b, c))

veloCansado1 = veloCansadoForm(chute1[0], chute1[1], chute1[2], tempoCansado1)
veloCansado2 = veloCansadoForm(chute2[0], chute2[1], chute2[2], tempoCansado2)
veloCansado3 = veloCansadoForm(chute3[0], chute3[1], chute3[2], tempoCansado3)


plt.plot(x, tempoCansado1, color="green")
plt.plot(x, tempoCansado2, color="blue")
plt.plot(x, tempoCansado3, color="purple")
plt.title('Questão 3')
plt.legend([r'Corredor 1', r'Corredor 2',r'Corredor 3'], loc='lower right')
plt.xlabel(r'$Posição(m)$')
plt.ylabel(r"$Tempo(s)$")
plt.show()

plt.plot(x, veloCansado1, color="green")
plt.plot(x, veloCansado2, color="blue")
plt.plot(x, veloCansado3, color="purple")
plt.title('Questão 3')
plt.legend([r'Corredor 1', r'Corredor 2',r'Corredor 3'], loc='lower right')
plt.xlabel(r'$Posição(m)$')
plt.ylabel(r"$Velocidade(\frac{m}{s})$")
plt.show()
'''
Como podemos ver, esse parâmetro adicional aumenta a força do corredor em relação
ao da questão 1, pois ao longo da corrida a força irá ser diminuida e na questão 1
a força é constante.
'''