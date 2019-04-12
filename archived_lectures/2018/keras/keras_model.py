import itertools
import pandas as pd
import tensorflow as tf
import datetime
import os


def compute_td_pct(djw, index, days):
    """ Computes a percentage change between a given day and some timedelta (days)
    Args:
        djw(PandasDataframe): contains index of prices and dates
        index(datetime): day to search
        days(int): numbers of days to search back
    Returns:
        (pct, int): percent change, and direction (1 positive, 0 negative)
    """
    pct = None
    ntd = djw.truncate(after=index).iloc[-1]["Closing Value"]
    if days > 0:
        pct = (djw[index:index + datetime.timedelta(days=1)].iloc[-1]["Closing Value"] - ntd) / \
              djw[index:index + datetime.timedelta(days=days)].iloc[-1]["Closing Value"]
    else:
        pct = (ntd - djw[index + datetime.timedelta(days=days):index].iloc[0]["Closing Value"]) / ntd
    if pct > 0.0:
        return pct, 1
    else:
        return pct, 0


djw = pd.read_csv("djw.csv")  # Dow Jones Industrial Average Prices by Day
djw = djw.set_index(pd.to_datetime(djw["Date"]))  # Set the Datetime as index
data = pd.read_csv("data.csv")  # Read in 3rd party handlabeled data
data = data.set_index(pd.to_datetime(data["date_elected"]))  # Set the datetime as the index
data = data[1:]  # We remove the first index to make sure we have enough data to look backwards

# This could have been done in a list of lists but was made explicit for demonstration purposes
day_before_1 = []  # 1 day before the election
day_before_7 = []  # 7 days before the election
day_before_30 = []  # 30 days before the election
day_before_60 = []  # 60 days before the election
day_before_180 = []  # 180 days before the election
day_before_365 = []  # 365 days before the election
day_before_730 = []  # 730 days before the election
day_after_1 = []  # 1 day after the election
day_after_7 = []  # 7 days after the election
day_after_30 = []  # 30 days after the election
day_after_60 = []  # 60 days after the election
day_after_180 = []  # 180 days after the election
day_after_365 = []  # 365 days after the election
for index, row in data.iterrows():
    day_after_1.append(
        compute_td_pct(djw, index, 1)[1])  # Note here we are just getting the direction instead of the market change
    day_after_7.append(compute_td_pct(djw, index, 7)[0])
    day_after_30.append(compute_td_pct(djw, index, 30)[0])
    day_after_60.append(compute_td_pct(djw, index, 60)[0])
    day_after_180.append(compute_td_pct(djw, index, 180)[0])
    day_after_365.append(compute_td_pct(djw, index, 365)[0])
    day_before_1.append(compute_td_pct(djw, index, -1)[0])
    day_before_7.append(compute_td_pct(djw, index, -7)[0])
    day_before_30.append(compute_td_pct(djw, index, -30)[0])
    day_before_60.append(compute_td_pct(djw, index, -60)[0])
    day_before_180.append(compute_td_pct(djw, index, -180)[0])
    day_before_365.append(compute_td_pct(djw, index, -365)[0])
    day_before_730.append(compute_td_pct(djw, index, -730)[0])

# Finally construct a DataFrame containing all of the data and add column labels and concat
# the market data to the third party data
market_data_cols = [day_before_1, day_before_7, day_before_30, day_before_60, day_before_180, day_before_365,
                    day_before_730, day_after_1, day_after_7, day_after_30, day_after_60, day_after_180, day_after_365]
market_data_col_names = ["day_before_1", "day_before_7", "day_before_30", "day_before_60", "day_before_180",
                         "day_before_365", "day_before_730", "day_after_1", "day_after_7", "day_after_30",
                         "day_after_60", "day_after_180", "day_after_365"]
market_data = pd.DataFrame(market_data_cols).transpose()
market_data.columns = market_data_col_names
market_data = market_data.set_index(data.index)  # this operation is not inplace, use existing dataframe's index
frames = [data, market_data]  # Pandas has some quirks unlike sql when concatenating
combined_df = pd.concat(frames, axis=1)  # Axis 0 is after, 1 is next-to



# our goal is to predict the party that will win the election
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Softmax
from sklearn.model_selection import train_test_split

X = combined_df[['party', 'prev_held_office', 'previous_party_1',
       'previous_party_2', 'was_vp_or_vp_runner', 'day_before_1',
       'day_before_7', 'day_before_30', 'day_before_60', 'day_before_180',
       'day_before_365', 'day_before_730']]
y = combined_df[["party", "day_after_1", "day_after_7", "day_after_30",
                         "day_after_60", "day_after_180", "day_after_365"]]

#X_train,  X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train = X.iloc[:-1]
y_train = y.iloc[:-1]
X_test  = X.iloc[-1:]
y_test  = y.iloc[-1:]

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = Sequential()
model.add(Dense(10, input_dim=len(X.columns)))
model.add(Activation("relu"))
model.add(Dense(len(y.columns)))
model.add(Flatten())


model.compile(loss="mean_squared_error",optimizer=sgd,metrics=["mae"])
hist = model.fit(X_train, y_train, epochs=1000, verbose=0, validation_split=0.2)
scores = model.evaluate(X_test, y_test)
print(pd.DataFrame(model.predict(X_test[0:5])).transpose())
print(y_test[0:5].iloc[0])
from tensorflow.keras.utils import plot_model
plot_model(model, to_file="/Users/athaker/Desktop/model.png")
