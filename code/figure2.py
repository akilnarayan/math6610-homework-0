# Plots histogram of deviation of 4F_M from the mean and compares to
# what is predicted by the CLT.

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from utils import generate_4FM

imgsave = True

N = 3000 # Ensemble size for histogram

Ms = [100, 1000, 10000]

# This array contains scaled and shifted realizations of F
Gs = np.zeros((N, len(Ms)))

muf = np.pi/4
sigmaf = np.sqrt( np.pi/4 - (np.pi/4)**2 )

for ind, M in enumerate(Ms):

    Gs[:,ind] = generate_4FM(N, M)/4

    # Scale and shift according to what is predicted by the CLT
    Gs[:,ind] = np.sqrt(M)/sigmaf * (Gs[:,ind] - muf)

fontprops = {'fontsize': 16}

Nbins = 30
bin_edges = np.linspace(-4, 4, Nbins+1)
x = np.linspace(np.min(bin_edges), np.max(bin_edges), 100)

for ind, M in enumerate(Ms):
    plt.subplot(131 + ind)
    plt.hist(Gs[:,ind], bin_edges, density=True)
    plt.plot(x, norm.pdf(x), 'k')
    plt.xlabel('$x$', **fontprops)
    plt.title('$M={0:d}$'.format(M), **fontprops)
    if ind == 0:
        plt.legend(('$\\varphi(x)$', 'Histogram'))

plt.gcf().set_size_inches(13, 4.2)

if imgsave:
    plt.savefig('CLT.pdf')
else:
    plt.show()
