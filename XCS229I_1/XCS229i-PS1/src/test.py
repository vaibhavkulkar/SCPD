import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = np.poly1d(x,1)
factor = 2.0
#print(mymodel)
plot_x = np.ones([1000, 1])
plot_x[:, 0] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
plt.scatter(plot_x,plot_x)
plt.show()
print (plot_x)

#plt.scatter(x, y)
#plt.plot(myline, mymodel(myline))
#plt.show()