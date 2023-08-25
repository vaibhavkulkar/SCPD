import numpy as np

x = [1,2,3,4]
k = 5
poly = np.ones((len(x), k+1), np.int)

for xi in range (0,len(x)):
    for ki in range(0, k+1):
        xpower = np.power(x[xi], ki)
        poly[xi, ki] = xpower
print poly