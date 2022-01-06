# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy import sqrt, exp, log, pi
import pandas_datareader.data as web
from tqdm import tqdm


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


class AnnualReturnsDistribution:
    """Annual Return Distribution."""

    def __init__(self, ticker: int, begin_date: str,
                 end_date: str, n_simulation: int, source: str = 'yahoo'):
        self.data = web.DataReader(name=ticker, data_source=source,
                                   start=begin_date, end=end_date)
        self.n_simulation = n_simulation

    def __call__(self):
        self._get_annual_returns()
        self._get_estimated_distribution_with_replacement()
        self._show_mean_return_distriibution()

    def _get_annual_returns(self):
        returns_dataframe = np.log((self.data['Adj Close']
                                    .pct_change()
                                    .dropna()
                                    .add(1)
                                    ).to_frame(name='log_returns'))
        returns_dataframe.index = (date.year for date in returns_dataframe.index)

        self.annual_returns = np.exp((returns_dataframe
                                      .groupby(returns_dataframe.index)
                                      .sum())) - 1
        self.annual_returns.columns = ['annual_returns']

    def _get_estimated_distribution_with_replacement(self):
        n_obs = len(self.annual_returns)
        self.estimated_returns = np.zeros(n_obs)

        for i in range(n_obs):
            x = np.random.uniform(low=0, high=n_obs, size=n_obs)
            y = []

            for j in range(n_obs):
                y.append(int(x[j]))
                z = np.array(self.annual_returns)[y]

            self.estimated_returns[i] = np.mean(z)

    def _show_mean_return_distriibution(self):
        plt.title(f'Mean return distribution: number of simulation={self.n_simulation}')
        plt.xlabel('Mean returns')
        plt.ylabel('Frequency')
        mean_annual = round(np.mean(np.array(self.annual_returns)), 4)
        plt.figtext(0.63, 0.8, f'mean annual={mean_annual}')
        plt.hist(self.estimated_returns, 50)
        plt.show()


class StockPriceMovement:
    """

    StockPriceMovement class is to simulate stock price movement.

    Below is an explanation for arguments in this class.

    stock_price_today: stock price at time zero
    T: maturity date (in years)
    n_steps: number of steps
    mu: expected annual return
    sigma: annualized volatility
    n_simulation: number of simulations

    """

    def __init__(self, stock_price_today: float, T: float, n_steps: int,
                 mu: float, sigma: float, n_simulations: int):
        self.stock_price_today = stock_price_today
        self.T = T
        self.n_steps = n_steps
        self.mu = mu
        self.sigma = sigma
        self.n_simulations = n_simulations

    def __call__(self, graph='time_series'):
        self._get_simulated_stock_price()
        self._show_simulated_stock_price(graph)

    def _get_simulated_stock_price(self):
        dt = self.T / self.n_steps
        self.x = range(self.n_steps)
        
        self.S = [[] for _ in range(self.n_simulations)]

        for n in range(self.n_simulations):
            self.S[n] = np.zeros([self.n_steps])
            self.S[n][0] = self.stock_price_today
            for i in self.x[:-1]:
                e = np.random.normal()
                self.S[n][i + 1] = (self.S[n][i] +
                                    self.S[n][i] * (self.mu -
                                                    0.5 * (self.sigma ** 2)
                                                    ) * dt +
                                    self.sigma * self.S[n][i] * np.sqrt(dt) * e)

    def _show_simulated_stock_price(self, graph):
        if graph == 'time_series':
            for i in range(self.n_simulations):
                plt.plot(self.x, self.S[i])
            plt.figtext(0.2, 0.8, f'S0={self.S[i][0]}, mu={self.mu}, sigma={self.sigma}')
            plt.figtext(0.2, 0.76, f'T={self.T}, steps={self.n_steps}')
            plt.title(f'Stock price (number of simulation = {self.n_simulations})')
            plt.xlabel(f'Total number of steps = {self.n_steps}')
            plt.ylabel('stock price')

        elif graph == 'hist':
            plt.title(f'Histogram of terminal price')
            plt.xlabel(f'Number of frequencies')
            plt.ylabel('Terminal price')
            plt.figtext(0.5, 0.8, f'S0={self.S[0][0]}, mu={self.mu}, sigma={self.sigma}')
            plt.figtext(0.5, 0.76, f'T={self.T}, steps={self.n_steps}')
            plt.figtext(0.5, 0.72, f'Number of terminal prices={self.n_simulations}')
            plt.hist([x[-1] for x in self.S])

        plt.show()


class BlackScholesMerton:
    """

    BlackScholesMerton class is to estimate call price.

    Below is an explanation for arguments in this class.

    S0: stock price at time zero
    X: exercise price
    T: maturity date (in years)
    r: risk-free rate
    sigma: annualized volatility
    n_steps: number of steps
    n_simulations: number of simulation

    """

    def __init__(self, S0: float, X: float, T: float, r: float,
                 sigma: float, n_steps: int, n_simulations: int):
        self.S0 = S0
        self.X = X
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.n_simulations = n_simulations

    def get_estimated_call_price(self) -> float:
        dt = self.T / self.n_steps
        call = np.zeros(self.n_simulations)

        for n in tqdm(range(self.n_simulations)):
            s_T = self.S0
            for _ in range(self.n_steps - 1):
                e = np.random.normal()
                s_T *= np.exp(
                    (self.r - 0.5 * (self.sigma ** 2)) * dt +
                    self.sigma * e * np.sqrt(dt)
                    )
                call[n] = max(s_T - self.X, 0)

        call_price = call.mean() * np.exp(-self.r * self.T)

        return call_price
