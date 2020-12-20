import numpy as np

#Dupla : Jéssica Beatriz e Savio Carvalho


def d2_func(vet, niter, d2):
    if niter == 0:
        return vet
    else:
        vet.append(d2)
        d2 = 2. - 2. * np.sqrt(1. - d2 / 4)
        d2_func(vet, niter - 1, d2)
    return vet


niter = 28

iterable = (i for i in range(1, niter + 2))
it = np.fromiter(iterable, int)

nd = np.power(2, it)
d2 = 2
vet = []
res = d2_func(vet, niter+1, d2)

pi_arch = nd * np.sqrt(res)
error = np.abs(pi_arch / np.pi - 1.)


print("O valor nd  =", nd[-1],  "|| O valor PI =", pi_arch[-1], "|| O valor Error = ", error[-1])
