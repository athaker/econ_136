
# coding: utf-8

# # Predicting the Trump Election: An Introduction to Tensorflow
# This notebook demonstrates an introduction to Tensorflow for predicting the Trump victory for the 2016 presidential election, using stock market and 3rd party data. Through this guide you will utilize [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) a library in Tensorflow that allows you to quickly build a fully connected neural network and train a model. DNNRegressor is chosen over a more standard classification pipeline as we do not have to 1-hot encode the features. This will allow for our input vectors (i.e. timeseries data), to be more dense.
# 
# The following steps are performed:
# 1. Files
# 2. Preprocess timeseries features
# 3. Play/Visualize the data
# 4. Model: Predicting Trump Election
# 5. Model: Predicting the market returns

# ### Overview 
# The goal of this is to predict the winner of the 2016 Presidential Election using publicly available data at the time.  Instead of predicting the person to win we will phrase this problem as a binary classification task: predicting the political party that will win the election (Republican or Democratic). This will give us more data to sample from, hopefully improving the model performance. After that, we will use similar features to train another neural network which will be used to predict the market return after the election date. 
# 
# We have a small amount of data, overfitting and biases are a major problem. In practice you will have much more data. Using a GPU to accelerate the training time will be beneficial. A common GPU is the [Tesla K80 GPU](https://www.nvidia.com/en-us/data-center/tesla-k80/), and old but powerful and expensive GPU. 

# In[1]:


# Imports
import os
import pandas as pd
import datetime


# ### 1. Files Files
# Here you will find two files: data.csv, djw.csv
# 
# * **data.csv** contains historical data about presidential elections dating back until 1900.
# * **djw.csv** contains historical data about the [Dow Jones Industrial Average](https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average) for the daily close prices. 
# Although this is an index, it can closely approximate the DIA ETF and other ETF's that track "the market". This was chosen due the large dataset size. 

# ### 2. Preprocess Data
# In this section we preprocess the data. Here we have to be careful that we do not inject forward looking bias. Therefore, we need to use the ```truncate``` command to get the nearest day before. The helper function below shows how to get the percentage return of the market given a dataframe of prices. 

# In[2]:


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
        pct = (djw[index:index+datetime.timedelta(days=1)].iloc[-1]["Closing Value"] - ntd) / djw[index:index+datetime.timedelta(days=days)].iloc[-1]["Closing Value"]
    else:
        pct = (ntd - djw[index+datetime.timedelta(days=days):index].iloc[0]["Closing Value"]) / ntd
    if pct > 0.0:
        return pct, 1
    else:
        return pct, 0


# We need to convert times to datetimes for easier processing. Pandas has great built in libraries that allow for quick data parsing. Pandas include a nice helper function called ```.to_datetime()``` which will automatically convert and figure out datetimes for you.

# In[3]:


djw = pd.read_csv("djw.csv") # Dow Jones Industrial Average Prices by Day
djw = djw.set_index(pd.to_datetime(djw["Date"])) # Set the Datetime as index
data = pd.read_csv("data.csv") # Read in 3rd party handlabeled data
data = data.set_index(pd.to_datetime(data["date_elected"])) # Set the datetime as the index
data = data[1:] # We remove the first index to make sure we have enough data to look backwards


# Label out the features to sample. Here we believe that the market or some combination of the market features may predict the election. Id est: smart money might know where the election may go and invest accordingly. 

# In[4]:


# This could have been done in a list of lists but was made explicit for demonstration purposes
day_before_1 = []   # 1 day before the election 
day_before_7 = []   # 7 days before the election
day_before_30 = []  # 30 days before the election
day_before_60 = []  # 60 days before the election
day_before_180 = [] # 180 days before the election
day_before_365 = [] # 365 days before the election
day_before_730 = [] # 730 days before the election
day_after_1 = []    # 1 day after the election
day_after_7 = []    # 7 days after the election
day_after_30 = []   # 30 days after the election
day_after_60 = []   # 60 days after the election
day_after_180 = []  # 180 days after the election
day_after_365 = []  # 365 days after the election
for index, row in data.iterrows():
    day_after_1.append(compute_td_pct(djw, index, 1)[1]) # Note here we are just getting the direction instead of the market change
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
market_data_cols = [day_before_1, day_before_7, day_before_30, day_before_60, day_before_180, day_before_365, day_before_730, day_after_1, day_after_7, day_after_30, day_after_60, day_after_180, day_after_365]
market_data_col_names = ["day_before_1","day_before_7","day_before_30","day_before_60","day_before_180","day_before_365","day_before_730","day_after_1","day_after_7","day_after_30","day_after_60","day_after_180","day_after_365"]
market_data = pd.DataFrame(market_data_cols).transpose()
market_data.columns = market_data_col_names
market_data = market_data.set_index(data.index) # this operation is not inplace, use existing dataframe's index
frames = [data, market_data] # Pandas has some quirks unlike sql when concatenating
combined_df = pd.concat(frames, axis=1) # Axis 0 is after, 1 is next-to


# ### 3. Play/Visualize Data
# Now that we have preprocessed the data, take a look at the data and get a feel for how it is structured. You will note that there is not that much data, as it is hard to find reliable stock data in the early 1900's. 
# 
# Examine the features to get a sense of what they mean. 
# * Party - 1 if Republican, 0 if Democratic
# * Previously Held Office - 1 if true
# * Previous Party - the party that was previously in power (goes back 2 terms), 1 if Republican, 0 if Democratic
# * Was VP or VP Runner - 1 if held the position of VP before the current election
# * day_before_n - percentage or direction of the market for a given number of days before the current election cycle but not including the day
# * day_after_n - percentage or direction of the market for a given number of days after the current election cycle
# 
# When I actually did the prediction, I had much more data than just the above. I used [Google Trends](trends.google.com) to add more feature data. Furthermore, I added a "sentiment analysis" by looking through social media and other documents to get a feeling for the expected outcome. I strongly reccomend you include more features and more data than the 20+ elements we have here. More data the better. High quality data is important. 

# In[5]:


combined_df.head() # gives the top 5, can use tail to give the last 5


# In[6]:


combined_df.describe() # statistics about the dataframe


# ### 4. Model: Predicting the Trump Election
# Here we will train a DNN that aims to predict the 2016 Presidential Election. The features will be the features explored above (except for the forward looking ones). You do not need to fully understand how a [neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) works, however it can be thought of mapping inputs to outputs and the network will figure out everything inbetween. The aim is to not have the best network architecture possible, but to leverage neural network's ability to find patterns among data that otherwise would be difficult or timeconsuming to find by pure inspection. 
# 
# We are using Deep Learning to figure out the useful features and generate a model based upon those useful features to predict upon. 
# 
# Tensorflow is the selected Deep Learning framework, as it tends to be the most popular in industry. There are many others and each has a different purpose and use. Use what is best to get the job done.
# * CNTK (Microsoft Cognitive Toolkit)
# * Keras - this actually is a high level API that has general calls to other frameworks
# * Theano
# * Torch
# * Caffe/Caffe2
# * Scikit learn

# In[7]:


# Import statements
import itertools
import pandas as pd
import tensorflow as tf


# In[8]:


# Set the logging level, API Doc - https://www.tensorflow.org/api_docs/python/tf/logging/set_verbosity)
# This is a low-level setting so more information than you would like may appear
tf.logging.set_verbosity(tf.logging.INFO)


# In[9]:


def get_input_fn(data_set, label, features, num_epochs=None, shuffle=True):
    """
    Args:
        dataset(DataFrame): Pandas DataFrame containing the dataset
        num_epochs(int): number of epochs to train
        shuffle(bool): shuffle the dataset randomly before training
    """
    return tf.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame({k: data_set[k].values for k in features}),
            y=pd.Series(data_set[label].values),
            num_epochs=num_epochs,
            shuffle=shuffle)


# Take the relevant features and label we are trying to predict; market prices and presidential data. 

# In[10]:


# Labeled Data
COLUMNS = combined_df.columns[1:13] # Features and wanted predicted label
FEATURES = combined_df.columns[2:13] # Features only without predicted label
LABEL = "party" # Trying to predict the party that will win the election
# Feature Columns
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]


# Split up the testing and training set. Generally **(almost always!) you would have a test set, and multiple validation sets.** However, given the small sample size we will only train the model, and only predict on the most recent election. Additonally, be sure to randomize your data but consider the time component. In practice there is not always enough data and more data tends to give better performance. After a model has been validated to "work" it is retrained on the entire set to hopefully improve the model's performance.
# 
# __Try for yourself:__ Split up the training_set, test_set, and prediction_set and see how the model performs given different dataset sizes. Predict on other elections, how does the model perform on other elections? 

# In[11]:


# Test, Train, Prediction Sets
training_set = combined_df[COLUMNS].iloc[:-1] # remove the most recent election results
test_set = None # This is not good in practice but doing for now due to low amount of data
prediction_set = combined_df[COLUMNS].iloc[-1:] # the most recent election results


# Here we actually build the model, using [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor) in Tensorflow. We will build a fully connected network with 4 hidden layers with dimensions 44,22,11,11. (This would be considered a small deep network, look at Resnet 152 - a network 152 layers deep used for image classification tasks). 
# 
# The larger the model the better performance is expected (unless you are overfitting), but runtime will be significantly longer. For very large models, with giant datasets many GPU's are used in parallel to train upon (I had a project which used maybe 1000 at once). 
# 
# An important note is hyperparameter selection. These can be the network architecture, initial starting weights, activation functions, regularization, dropout rate, and many other factors. Right now we are using mainly defaults for simplicity. In practice you would train many models across a large variety of parameters, and select the best one (be careful of overfit). A large portion of a data scientests time is finding the best selection of hyperparameters. 
# 
# Now with model's becoming more common, AutoML/AutoDL is becoming a practice where Deep Networks are used to predict the best parameters to initially use.
# 
# **__Note__: you will need to delete the model_dir after use or use another directory as this defaults to checkpointing from the last training epoch**, if you change the model params without deleting you may encounter an error, or keep training from the point you left off!
# 
# Play with the evaluation part (commented out) on a test set. 

# In[12]:


# Build a Fully Connected DNN with 11, 11, units respectively
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[44,22,11,11], model_dir="./pres")

# Train the network
regressor.train(input_fn=get_input_fn(training_set, LABEL, FEATURES), steps=2000)

# Evaluate the loss over one epoch of the test_set
# ev = regressor.evaluate(input_fn=get_input_fn(test_set,num_epochs=1, shuffle=False))


# #### Evaluation
# Look at that loss! That is an incredibly low loss given the size of the network with respect to the size of the data. We might have overfit our data. 
# 
# Now predict the output to see how we did!!! Note due to randomness of parameters upon initialization it is expected to get slighlty different results than others. 

# In[13]:


#Print out predictions over a slice of the prediction set
y = regressor.predict(input_fn=get_input_fn(prediction_set, LABEL, FEATURES, num_epochs=1, shuffle=False))

# .predict() returns an iterator of dicts; convert to a list and print the predictions
predictions = list(p["predictions"] for p in itertools.islice(y, len(prediction_set)))
print("Predictions: {}".format(str(predictions)))
if predictions[0] > 0.5:
    print("Predicting a TRUMP (Republican) victory")
else:
    print("Predicting a CLINTON (Democratic) victory")


# If you did not change parameters you should get roughly 0.9, meaning we expect the Republican party to win the election, and therefore a Trump victory. **We correctly predicted the election!** However, with an output of 0.9 we are pretty confident, thus may be in a situation where we are overpredicting (depending on the parameters and given that this is a regression the coefficients in the network can lead to predicting over a value of 1). Do not take this value as a likelyhood as this process can be thought of as a regression. You can use a softmax to get predicted probabilities, but that is out of the scope of this workbook. 

# ### 5. Predicting the Market return after the Trump Election
# We will follow a similar process to the above. Furthermore, we will use similar features, but take out the party as that would inject some forward looking bias. 
# 
# Here we will predict a direction instead of a size (1 - market up from election, 0 - market down from election). In section 2 we labeled the direction of the market return. You can play with that data column and return instead the market value, see how it changes the performance! Furthermore, play with the other values and add it to the labels to predict upon! See if you can classify the direction of 1 year out!

# In[14]:


# Now lets build a DNN to predict the expected market response, this follows a similar process as above
M_COLUMNS = combined_df.columns[2:14]
M_FEATURES = combined_df.columns[2:13]
M_LABEL = "day_after_1"

# Test, Train, Prediction Sets
m_training_set = combined_df[M_COLUMNS].iloc[:-1]
m_test_set = None # This is not good in practice but doing for now due to low amount of data
m_prediction_set = combined_df[M_COLUMNS].iloc[-1:] # What is the expected return

# Feature Columns
m_feature_cols = [tf.feature_column.numeric_column(k) for k in M_FEATURES]

# Build a Fully Connected DNN with 11, 11, units respectively
m_regressor = tf.estimator.DNNRegressor(feature_columns=m_feature_cols, hidden_units=[44,22,11,11], model_dir="./market")

# Train the network
m_regressor.train(input_fn=get_input_fn(m_training_set, M_LABEL, M_FEATURES), steps=2000)

# Evaluate the loss over one epoch of the test_set
#ev = regressor.evaluate(input_fn=get_input_fn(test_set,num_epochs=1, shuffle=False))

#Print out predictions over a slice of the prediction set
m_y = m_regressor.predict(input_fn=get_input_fn(m_prediction_set, M_LABEL, M_FEATURES, num_epochs=1, shuffle=False))

# .predict() returns an iterator of dicts; convert to a list and print the predictions
m_predictions = list(p["predictions"] for p in itertools.islice(m_y, len(m_prediction_set)))
print("Predictions: {}".format(str(m_predictions)))
print("Actual: " + str(m_prediction_set.iloc[0].day_after_1))

if round(m_predictions[0][0]) ==  m_prediction_set.iloc[0].day_after_1:
    print("Correctly predicted the market direction!")


# Again it looks like we are in a situation were we may have overpredicted! But we did indeed predict the direction of the market return after Trump's victory! 
# 
# Congrats you have trained your first (or maybe not) neural network!

# ### Closing Remarks
# This serves as only an introduction to how to use deep learning to predict the market. I would not use the model as is to predict the next election. There is much to be said about data engineering and bias that we can do to improve this model. 
# 
# What are some problems with the process that we have done?
# 
# What can be done to improve the model?

# In[15]:


import shutil
shutil.rmtree("./market") # Need to update to remove when folders are not empty
shutil.rmtree("./pres") # Need to update to remove when folders are not empty

