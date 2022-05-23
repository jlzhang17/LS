from typing import TextIO, List

import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.optimize import curve_fit
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

filename = 'NJ66_2021-11-23.txt'
xdata, ydata, erry, ADCx, chisq= [], [], [], [], []
with open(filename, 'r') as f:
    lines = f.readlines()
    pedestal = float(lines[0].split()[1])
    for i in lines[1:]:
        value = [float(s) for s in i.split()]
        xdata.append(value[0])
        ydata.append(value[1] - pedestal)
        erry.append(value[2])


# print(xdata)
# print(ydata)
# print(erry)
# print(pedestal)


def func(x, L, ADC_0):
    return ADC_0 * np.exp(-x / L)


popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [70., 4096.]))
popt, pcov
p_sigma = np.sqrt(np.diag(pcov))
print(popt)

print(p_sigma[0])



x = np.linspace(0, 1.05, 100)
y = func(x, *popt)
for i in range(len(xdata)):
    chisq.append(((ydata[i] - func(xdata[i], *popt))/(erry[i]+0.0027*func(xdata[i], *popt)))**2)
chisq = sum(chisq)    
print(chisq)
p2, = plt.plot(x, y, '-', color='#00FF00', label='Fit')
p1 = plt.errorbar(x=xdata, y=ydata, yerr=erry, fmt='o', ms=2.8, ecolor='black', color='b', elinewidth=0.5,capsize=1, label='ADC')
#p1 = plt.plot(xdata, ydata, '.', ms=6, color='b', label='ADC')

plt.xlim(0.05, 1.05)
plt.title("Attenuation Length", fontdict={'family': 'Arial', 'size': 16})
plt.xlabel('Liquid Thickness x (m)', fontdict={'family': 'Arial'})
plt.ylabel('ADC Channels', fontdict={'family': 'Arial'})
# l1 = \
label = fr'$L_\lambda$ =$ {round(popt[0], 2)}$ $\pm{round(p_sigma[0], 2)}$ (m)'
p3, = plt.plot([], [], ' ')
label2 = fr'$\chi^2/ndf$ $= {round(chisq,2)}/8$ '
plt.legend([p1,p2,p3],["ADC", label,label2], prop={'family': 'Arial'}, title='Data:', loc=1)
#plt.legend(["$L_\lambda$="], title='Attenuation Length:', prop={'family' : 'Arial'},loc = 1)
# plt.gca().add_artist(l1)


my_x_ticks = np.arange(0.5, 0.85, 0.1)
plt.xticks(my_x_ticks, fontproperties='Arial')
plt.yticks(fontproperties='Arial')
plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.tick_params(which='minor', direction='in')
plt.tick_params(which='major', direction='in')
plt.grid(linestyle='--', linewidth=0.6)

fig = plt.gcf()
fig.savefig('1_Attenuation_Length_WLS.pdf', dpi=600, format='pdf')
plt.show()

# r'$L_\lambda$ = 10$^{{{}}}$ '.format(round(popt[0],2))
