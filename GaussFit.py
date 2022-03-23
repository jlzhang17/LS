import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

a = np.loadtxt('1.txt')
c = []
A = sum(a[:, 1])
for num in a[:, 1]:
    c.append(float(num/A))
x = [i for i in range(0, 4096)]
plt.bar(x, c, width = 1)

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-((x - mu) / np.sqrt(2) / sigma)**2)
popt, pcov = curve_fit(gaussian, x, c, bounds=([500, -np.inf, -np.inf], [3500, np.inf, np.inf]))

p_sigma = np.sqrt(np.diag(pcov))
print(round(popt[0],3),round(popt[1],3),round(p_sigma[0],3) )
#print(round(popt[0],3),p_sigma[0])
plt.plot(x,gaussian(x,*popt),'r-')
plt.show()
