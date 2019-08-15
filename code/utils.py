import numpy as np


def generate_X(N,M):
    """
    Generates N * M realizations of a random variable X that is uniformly
    distributed on [-1,1]^2.

    Output: an N x M x 2 array.
    """

    return np.random.rand(N, M, 2)

def generate_4FM(N,M):
    """
    Generates N realizations of a random variable 4 F_M that is defined as

        4 \sum_{m=1}^M f(X_m),

    where X_m is uniformly distributed on [-1,1]^2, and f(.) is the
    indicator function on the unit ball centered at the origin.
    """

    X = generate_X(N,M)
    return 4*np.sum(np.sum(X**2, axis=2) <= 1., axis=1)/M
