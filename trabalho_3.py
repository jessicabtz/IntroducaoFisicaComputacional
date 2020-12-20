import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad

#---------------------------------------- Questao 1 -----------------------------------------------------
#Monte Carlo: Método Aceita-Rejeita
def func(x, n):
    arr = np.zeros(n, dtype=np.int8)
    r2 = x**2
    soma = r2.sum(axis=1)
    arr[soma < 1] = 1
    return arr

rng  = np.random.default_rng()
n    = 100000
ndim = 10

x = 2 * rng.random((n, ndim)) - 1

arr = func(x, n)
integral = 2 ** ndim * np.sum(arr) / n
print("Questao 1:")
print("A volume da esfera de 10 dimensoes e raio 1 e : ", integral,'\n')

'''
#Cálculo com a 'nquad' da Scipy

dim = 4
lims = [[0, 1]] * dim
func2 = lambda x0, x1, x2, x3: (1 if (x0**2 + x1**2 + x2**2+ x3**2 < 1) else 0)
integral1, er1 = nquad(func2, lims, opts=[dict(epsrel=1e-4, epsabs=1e-4)] * dim)
integral1 = integral1 * (2 ** dim)
print(integral1)
 
'''
'''
Como podemos ver no cálculo de até 4 dimensões as respostas são 
semelhantes. No entanto quando calculamos para mais de 5 dimensões, 
os métodos tradicionalmente utilizados (função scipy.integrate.nquad)
são inviáveis, pois há uma demora muito grande para calcular. Todavia, pelo
método do Monte Carlo a complexidade cresce linearmente, desse modo
é mais prático e rápido calcular.
'''

#---------------------------------------- Questao 2 -----------------------------------------------------
# Andar de 4 bebados
def andar(direcao, nPassos, nBebados):
    bebado = rng.choice(direcao, size=(nPassos, nBebados))
    bebado[0, :] = 0
    bebado = np.cumsum(bebado, axis=0)
    return bebado

print("Questao 2")
rng = np.random.default_rng()

nBebados = 4
nPassosA = int(input("Insira o numero de passos que os bebados darao: "))  # Numero de passos .
direcao = np.array([-1, 1])

caminhoBebadoX = andar(direcao, nPassosA, nBebados)
caminhoBebadoY = andar(direcao, nPassosA, nBebados)

plt.plot(caminhoBebadoX[:, 0], caminhoBebadoY[:, 0], "g--", caminhoBebadoX[-1, 0], caminhoBebadoY[-1, 0], 'go')
plt.plot(caminhoBebadoX[:, 1], caminhoBebadoY[:, 1], "r--", caminhoBebadoX[-1, 1], caminhoBebadoY[-1, 1], 'ro')
plt.plot(caminhoBebadoX[:, 2], caminhoBebadoY[:, 2], "b--", caminhoBebadoX[-1, 2], caminhoBebadoY[-1, 2], 'bo')
plt.plot(caminhoBebadoX[:, 3], caminhoBebadoY[:, 3], "y--", caminhoBebadoX[-1, 3], caminhoBebadoY[-1, 3], 'yo')
plt.title('Questão 2 A')
plt.legend(['Bêbado 1','Posição Final 1', 'Bêbado 2', 'Posição Final 2', 'Bêbado 3', 'Posição Final 3', 'Bêbado 4',
            'Posição Final 4'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.xlabel("Caminho do Bêbado em X")
plt.ylabel("Caminho do Bêbado em Y")
plt.show()

# Distancia final ao quadrado de 100 mil bebados
def distanciaRQuad(x, y):
    return (x**2)+(y**2)


def media(n):
    passo = distanciaRQuad(caminhoBebadoXB[n], caminhoBebadoYB[n])
    return np.mean(passo)

nPassosB = 300
caminhoBebadoXB = andar(direcao, nPassosB, 100000)
caminhoBebadoYB = andar(direcao, nPassosB, 100000)

vet       = np.array([49, 99, 149, 199, 249, 299])
vet_media = np.zeros(6)
for i in range(6):
    vet_media[i] = media(vet[i])

N = vet + 1
plt.plot(N, vet_media)
plt.title('Questão 2 B')
plt.xlabel("N(Passos)")
plt.ylabel(r'$r^2_{rms}$')
plt.show()

"""
Como podemos ver, Rrms = raiz da média de r^2 = raiz de 2N, com isso
percebemos que Rrms^2 = média de r^2 = 2N que da como resultado uma reta.
"""