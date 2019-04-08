"""
import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
import datetime

def fuzzy_search(input_df, search_date, days, search_col):
    """ Finds the percent change and the direction for a given input

    Args:
        input_df: dataframe to search
        search_date: date to look
        days: days to search (can be positive or negative)
        search_col: column to search through

    Returns:
        pct_change: percentage change
        direction: 1, if positive, 0 if negative
    """
    search_date_after = search_date + datetime.timedelta(days=days)
    if days > 0:
        start = input_df.truncate(after=search_date).iloc[-1][search_col]
        end = input_df.truncate(after=search_date_after).iloc[-1][search_col]
    else:  # days < 0
        search_date_after = search_date + datetime.timedelta(days=days)
        start = input_df.truncate(after=search_date_after).iloc[-1][search_col]
        end = input_df.truncate(after=search_date).iloc[-1][search_col]

    pct_change = (end - start) / start
    direction = 1 if end > start else 0

    return pct_change, direction

data_dir = "data"
only_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
ids = list(map(str, range(len(only_files))))
df = pd.DataFrame([ids, only_files]).transpose()
df.columns = ["id", "name"]

df["Date"] = pd.to_datetime(df["name"].str.split("_").apply(lambda x: x[0]))



# Read in the images one-by-one
img_arrs = []
for i in range(len(only_files)):
    img_arr = img_to_array(load_img(os.path.join(data_dir, only_files[i]), target_size=(224,224))) / 255
    img_arrs.append(img_arr)

# Load up a pandas dataframe of the stock data
fut_data = pd.read_csv("futures_data/CHRIS-CME_W1.csv", sep=",")
fut_data["Date"] = pd.to_datetime(fut_data["Date"])
fut_data.set_index("Date", inplace=True)
fut_data.sort_index(inplace=True)

# Compute the return after some amount of days
n_days = [-180, -60, -30, -14, -7, -1, 1, 7, 14, 30, 60, 180]
n_days = [-20, 20]

for duration in n_days:
    col_name_pct = "pct_before_" + str(duration) if duration>0 else "pct_after_" + str(duration)
    col_name_direction = "direction_before_" + str(duration) if duration > 0 else "direction_after_" + str(duration)
    col_pct = []
    col_direction = []
    for index, row in df.iterrows():
        percent_change, direction = fuzzy_search(fut_data, row.Date.date(), duration, search_col="Last")
        col_pct.append(percent_change)
        col_direction.append(direction)

    df[col_name_pct] = col_pct
    df[col_name_direction] = col_direction

# now build the keras model that can take in a conv2d input




#datagen = ImageDataGenerator(rescale=1./255)
#data_generator = datagen.flow_from_directory(
#        "/home/athaker/gdrive/learning/presentations/E136/2019/air/images/",
#        target_size=(224, 224),
#        batch_size=1,
#        class_mode=None)

# Get the images into a parsable format using a generator, only go for the length of the data
# This is an infinite generator so cannot apply a list to it.
#img_data = []
#for i in range(len(data_generator)):
#    img_data.append(data_generator.next())

# Now that we have the image data, load up the data from the