# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 00:40:52 2022

@author: MJH
"""
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import scipy as sp
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

def amihud_illiquidity(data: pd.DataFrame) -> float:
    closing_price = data['Adj Close'].to_numpy()
    volume = data.Volume.to_numpy() * closing_price
    returns = np.diff(closing_price) / closing_price[1:]

    illiquidity = np.divide(np.abs(returns), volume[1:]).mean()

    return illiquidity
