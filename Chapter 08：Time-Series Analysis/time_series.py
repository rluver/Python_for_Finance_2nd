# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 00:40:52 2022

@author: MJH
"""
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import scipy as sp
import statsmodels.api as sm
from scipy import stats


x = fdr.DataReader("IBM", start='2016-1-1', end='2016-1-21')
print(x[0:4])
GDP = pd.read_csv('data/usGDPquarterly.csv')
ff = pd.read_pickle('data/ffMonthly.pkl').reset_index().rename(columns={'index':'DATE'})
ff.DATE = ff.DATE.map(str).apply(lambda x: datetime.strptime(x, '%Y%m'))


MSFT = web.DataReader('MSFT', 'yahoo', '2012-1-1', '2021-12-31')
GDP = web.DataReader('GDP', 'fred', '2012-1-1', '2021-12-31')

date_range = pd.date_range('1/1/2016', periods=252)
n = len(date_range)
returns = sp.random.normal(0.1, 0.2, n)
data = pd.DataFrame(returns, index=date_range, columns = ['returns'])


def roll_spread(data: any([pd.DataFrame, list]), col_names=None) -> float:
    if type(data) == pd.DataFrame:
        data = data[col_names]
    returns = np.diff(data)

    cov_matrix = np.cov(returns[:-1], returns[1:])
    if cov_matrix[0, 1] > 0:
        cov = round(2 * np.sqrt(cov_matrix[0, 1]), 3)
    else:
        cov = -round(2 * np.sqrt(-cov_matrix[0, 1]), 3)

    return cov


class Liquidity:

    @staticmethod
    def amihud_illiquidity(data: pd.DataFrame) -> float:
        closing_price = data['Adj Close'].to_numpy()
        volume = data.Volume.to_numpy() * closing_price
        returns = np.diff(closing_price) / closing_price[1:]

        illiquidity = np.divide(np.abs(returns), volume[1:]).mean()

        return illiquidity

    @staticmethod
    def pastor_stambaugh_liquidity(data: pd.DataFrame, ff):
        '''
        y_t = alpha + beta_1 * x_(1, t-1) = beta_2 * x_(2, t_2) + epsilon

        y_t  = R_t - R_f (excess stock return on day t)
               R_t: the return for the stock
               R_f: the risk-free rate

        x_(1, t) is the market return
        x_(2, t) is the signed dollar trading volume

        x_(2, t) = sign(R_t - R_(f, t) * P_t * volume)
                   P_t: the stock price
                   
        beta2: Pastor and Stambaugh's liquidity'
        '''
        returns = np.array([x/y - 1 for x, y in zip(MSFT['Adj Close'][1:], MSFT['Adj Close'][:-1])])
        dollar_volume = np.array([x * y for x, y in zip(MSFT['Adj Close'][1:], MSFT['Volume'])])
        dates = data.index

        tt = pd.DataFrame(returns, index=dates[1:], columns=['returns'])
        tt2 = pd.DataFrame(dollar_volume, index=dates[1:], columns=['dollar_volume'])

        ff_dates = ff.DATE.copy()
        ff = pd.DataFrame(ff.iloc[:, 1:])
        ff.index = ff_dates

        tt3 = pd.merge(tt, tt2, left_index=True, right_index=True)
        final = pd.merge(tt3, ff, left_index=True, right_index=True)

        y = [x - y for x, y in zip(final.returns[1:], final.RF[1:])]
        x1 = final.MKT_RF[:-1]
        x2 = np.sign(np.array(final.returns[:-1] - final.RF[:-1])) * np.array(final.dollar_volume[:-1])
        x3 = [x1, x2]

        n = np.size(x3)
        x = np.reshape(x3, [int(n/2), 2])
        x = sm.add_constant(x)

        results = sm.OLS(y, x).fit()

        return results
