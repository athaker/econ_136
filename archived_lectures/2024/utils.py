import pandas as pd
from datetime import datetime, timedelta

# Load the data from the files
data_df = pd.read_csv("updated_data.csv", index_col=0, parse_dates=[0])
djw_df = pd.read_csv("djw.csv", index_col=0, parse_dates=[0])


# Function to calculate the performance
def calculate_performance(date, s, e):
    start_date = date - pd.tseries.offsets.DateOffset(months=s)
    end_date = date - pd.tseries.offsets.DateOffset(months=e)
    period_data = djw_df.loc[start_date:end_date]

    if not period_data.empty:
        return (
            period_data["close"].iloc[-1] - period_data["close"].iloc[0]
        ) / period_data["close"].iloc[0]
    else:
        return None


def calculate_performance_after(date):
    # Adjust to start from 1 day after the given date
    start_date = date + pd.DateOffset(days=1)
    end_date = date + pd.DateOffset(months=1)

    # Slice the dataframe for the period from start_date to end_date
    period_data = djw_df.loc[start_date:end_date]

    # Check if the data is not empty
    if not period_data.empty:
        return (
            period_data["close"].iloc[-1] - period_data["close"].iloc[0]
        ) / period_data["close"].iloc[0]
    else:
        return None


def compute_td_pct(djw, index, days):
    ntd = djw.truncate(after=index).iloc[-1]["close"]
    if days > 0:
        pct = (djw[index : index + timedelta(days=1)].iloc[-1]["close"] - ntd) / djw[
            index : index + timedelta(days=days)
        ].iloc[-1]["close"]
    else:
        pct = (ntd - djw[index + timedelta(days=days) : index].iloc[0]["close"]) / ntd
    return pct, 1 if pct > 0 else 0


data_df["3-6_month_performance"] = data_df.index.map(
    lambda x: calculate_performance(x, 6, 3)
)
data_df["6-12_month_performance"] = data_df.index.map(
    lambda x: calculate_performance(x, 12, 6)
)
data_df["12-18_month_performance"] = data_df.index.map(
    lambda x: calculate_performance(x, 18, 12)
)
data_df["1_after"] = data_df.index.map(calculate_performance_after)
# data_df["day_after"] = data_df.index.map(lambda x: compute_td_pct(djw_df, x, 1)[1])
