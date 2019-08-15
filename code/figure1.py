# Computes statistics and convergence results associated with the random
# variable 4 F_M.

import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt

from utils import generate_4FM

imgsave = True

N = 1000 # Ensemble size for statistics

Ms = np.logspace(1, 4, 100, dtype=int)

qs = [90., 50., 10.]
quantiles = np.zeros((Ms.size, len(qs)))

for ind, M in enumerate(Ms):

    F = generate_4FM(N, M)

    # Can use numpy.quantile instead if have a recent enough version
    quantiles[ind,:] = np.percentile(np.abs(F - np.pi), qs)

# Compute a line of slope -1/2 to include in the plot
linex = np.logspace(3.1, 3.4, 10)
liney = 5*linex**(-0.5)

fontprops = {'fontsize': 16}
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'semibold'
rcParams['mathtext.fontset'] = 'dejavuserif'

plt.loglog(Ms, quantiles)
plt.legend(('90% quantile', 'median', '10% quantile'))
plt.xlabel('$M$', **fontprops)
plt.ylabel('$|4 F_M - \pi|$', **fontprops)

plt.plot(linex, liney, 'k--')
plt.gca().annotate(r'Slope$=-1/2$', xy=(10**3.35, 1.4*np.min(liney)), horizontalalignment='left')

if imgsave:
    plt.savefig('FM_stats.pdf')
else:
    plt.show()
