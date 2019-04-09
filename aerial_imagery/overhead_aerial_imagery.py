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
import numpy as np

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
    img_arr = img_to_array(load_img(os.path.join(data_dir, only_files[i]), target_size=(64,64))) / 255
    img_arrs.append(img_arr)

img_arrs = np.array(img_arrs)

# Load up a pandas dataframe of the stock data
fut_data = pd.read_csv("futures_data/CHRIS-CME_W1.csv", sep=",")
fut_data["Date"] = pd.to_datetime(fut_data["Date"])
fut_data.set_index("Date", inplace=True)
fut_data.sort_index(inplace=True)

# Compute the return after some amount of days
n_days = [-100, -60, -30, -14, -7, -1, 1, 7, 14, 30, 60, 180]


for duration in n_days:
    col_name_pct = "pct_after_" + str(abs(duration)) if duration>0 else "pct_before_" + str(abs(duration))
    col_name_direction = "direction_after_" + str(abs(duration)) if duration > 0 else "direction_before_" + str(abs(duration))
    col_pct = []
    col_direction = []
    for index, row in df.iterrows():
        percent_change, direction = fuzzy_search(fut_data, row.Date.date(), duration, search_col="Last")
        col_pct.append(percent_change)
        col_direction.append(direction)

    df[col_name_pct] = col_pct
    df[col_name_direction] = col_direction

# now build the keras model that can take in a conv2d input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Softmax
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Concatenate, Average





picture_input = Input(shape=(64,64,3), dtype="float32", name="picture_input")
ts_input =      Input(shape=(2,), dtype="float32", name="ts_input")
ddd = Dense(10)(ts_input)

hidden_1 = Conv2D(32, (3,3), activation="relu")(picture_input)
hidden_2 = MaxPooling2D(pool_size=(2,2))(hidden_1)
hidden_3 = Dropout(0.25)(hidden_2)
flatten_layer = Flatten()(hidden_3)
dd = Dense(10)(flatten_layer)


merged = Average()([dd, ddd])
dense = Dense(1)(merged)
model = Model(inputs=[picture_input, ts_input], outputs=dense)

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', metrics=['accuracy', 'mae'], optimizer=sgd)
# binary_crossentropy

hist = model.fit([img_arrs, df[["direction_before_14", "direction_before_30"]].values], np.array(df["direction_after_30"].to_list()), epochs=50, verbose=1, validation_split=0.2)

#from tensorflow.keras.utils import plot_model

#plot_model(model)
import matplotlib.pyplot as plt

"""

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))df.direction_before_20.to_list()
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])



ts_input = Input(shape=)

"""

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