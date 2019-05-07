# -*- coding: utf-8 -*-
"""
http://nbviewer.ipython.org/github/twiecki/financial-analysis-python-tutorial/blob/master/1.%20Pandas%20Basics.ipynb
Modified by Avi Thaker for Educational Purposes
"""

# Imports
import datetime
import pandas as pd
import pandas.io.data
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))

# Get Data
aapl = pd.io.data.get_data_yahoo('AAPL', 
        start=datetime.datetime(2006, 10, 1),
        end=datetime.datetime(2015, 4, 7) )
aapl.head()
# Do some time series manipulation
aapl['SMA50'] = pd.rolling_mean(aapl['Adj Close'],50)
aapl['EMA50'] = pd.ewma(aapl['Adj Close'], 50)

# Plot
plt.figure()
plot(aapl.index, aapl['Adj Close'])
plot(aapl.index, aapl['SMA50'])
plot(aapl.index, aapl['EMA50'])
plt.legend(('ADJ Close', 'SMA50', 'EMA50'))


df = pd.io.data.get_data_yahoo(symbols=['AAPL', 'GE', 'GOOG', 'IBM', 'KO', 'MSFT', 'PEP'])['Adj Close']
rets = df.pct_change()
a = plt.figure()
plt.scatter(rets.PEP, rets.KO)
plt.xlabel('Returns PEP')
plt.ylabel('Returns KO')


pd.scatter_matrix(rets, diagonal='kde', figsize=(10, 10));

corr = rets.corr()
plt.figure()
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)

plt.figure()

plt.scatter(rets.mean(), rets.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        