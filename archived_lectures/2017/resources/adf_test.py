# adf_test.py
# Modified by: Avi Thaker
# March 20, 2017
# Modified from https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing

# Import the Time Series library
import statsmodels.tsa.stattools as ts

# Import Datetime and the Pandas DataReader
from datetime import datetime
from pandas_datareader import data

# Download the Google OHLCV data from 1/1/2000 to 1/1/2013
goog = data.get_data_yahoo("GOOG", datetime(2000,1,1), datetime(2013,1,1))

# Output the results of the Augmented Dickey-Fuller test for Google
# with a lag order value of 1
out = ts.adfuller(goog['Adj Close'], 1)
print(out)