# calculate_hurst.py
# Modified by: Avi Thaker
# March 20, 2017
# Modified from https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing

from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from datetime import datetime
from pandas_datareader import data


def hurst(ts):
	"""Returns the Hurst Exponent of the time series vector ts"""
	# Create the range of lag values
	lags = range(2, 100)

	# Calculate the array of the variances of the lagged differences
	tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

	# Use a linear fit to estimate the Hurst Exponent
	poly = polyfit(log(lags), log(tau), 1)

	# Return the Hurst exponent from the polyfit output
	return poly[0]*2.0

# Create a Gometric Brownian Motion, Mean-Reverting and Trending Series
gbm = log(cumsum(randn(100000))+1000)
mr = log(randn(100000)+1000)
tr = log(cumsum(randn(100000)+1)+1000)


goog = data.get_data_yahoo("GOOG", datetime(2000,1,1), datetime(2013,1,1))
#goog = DataReader("GOOG", "yahoo", datetime(2000,1,1), datetime(2013,1,1))


# Output the Hurst Exponent for each of the above series
# and the price of Google (the Adjusted Close price) for 
# the ADF test given above in the article
print("Hurst(GBM):   %s" % hurst(gbm))
print("Hurst(MR):    %s" % hurst(mr))
print("Hurst(TR):    %s" % hurst(tr))

# Assuming you have run the above code to obtain 'goog'!
print("Hurst(GOOG):  %s" % hurst(goog['Adj Close']))