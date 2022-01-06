# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy import sqrt, exp, log, pi
import pandas_datareader.data as web


def bootstrap(data, n_obs: int, replacement=None) -> list:
    """bootstrap."""
    n = len(data)

    if n < n_obs:
        raise(Exception('n is less than n_obs'))

    if replacement is None:
        y = np.random.permutation(data)

        return y[:n_obs]

    else:
        for _ in range(n_obs):
            k = np.random.permutation(data)
            y.append(k[0])

        return y
