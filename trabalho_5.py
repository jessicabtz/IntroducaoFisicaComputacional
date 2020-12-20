import numpy as np

#--------------------------------------- Questão 2 ---------------------------------------------------------
def func(x1, x2, ni):
    arr = []
    for n in ni:
        f1 = 4 * x1 * (1 + x1) * np.cos(2 * np.pi * n * x1)
        f2 = 4 * x2 * (1 - x2) * np.cos(2 * np.pi * n * x2)
        a1 = np.trapz(f1, x1)
        a2 = np.trapz(f2, x2)
        an = a1 + a2

        f3 = 4 * x1 * (1 + x1) * np.sin(np.pi * n * x1)
        f4 = 4 * x2 * (1 - x2) * np.sin(np.pi * n * x2)
        b1 = np.trapz(f3, x1)
        b2 = np.trapz(f4, x2)
        bn = b1 + b2

        bn_analitico = -16 * (((-1) ** n) - 1) / ((np.pi ** 3) * (n ** 3))

        dif = bn_analitico / bn
        if n % 2 == 1:
            if abs(dif) >= 2 or abs(dif) <= 0.02:
                arr.append(n)

        print(f'{an:.5f} \t{bn:.15f} \t{bn_analitico:.15f}')

    return arr


a = -1
b = 0
c = 1

div = 100
ni = np.arange(1, 101, 1)

x1 = np.linspace(a, b, div)
x2 = np.linspace(b, c, div)
print('an numerico \tbn numerico \tbn analitico')
arr = func(x1, x2, ni)
print(f"Os coeficientes diferem em mais de 100% ou menos de 1% em: {arr}")
