# Imports
import os
import pandas as pd
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
import numpy as np


# Define Dataset Class
class ElectionDataset(Dataset):
    def __init__(self, dataframe, features, labels):
        self.dataframe = dataframe
        self.features = dataframe[features].values
        self.labels = dataframe[labels].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx, :].astype(np.float32)), torch.tensor(
            self.labels[idx].astype(np.float32)
        )


# Function to compute percentage change
def compute_td_pct(djw, index, days):
    ntd = djw.truncate(after=index).iloc[-1]["Closing Value"]
    if days > 0:
        pct = (
            djw[index : index + timedelta(days=1)].iloc[-1]["Closing Value"] - ntd
        ) / djw[index : index + timedelta(days=days)].iloc[-1]["Closing Value"]
    else:
        pct = (
            ntd - djw[index + timedelta(days=days) : index].iloc[0]["Closing Value"]
        ) / ntd
    return pct, 1 if pct > 0 else 0


# Add a picture of what the neural network looks like
# TL;dr this is a fully connected neural network with 3 hidden layers
# put in shaply feature attribution map


# Define a simple neural network
class DNNRegressor(nn.Module):
    def __init__(self, input_size):
        super(DNNRegressor, self).__init__()
        self.layer1 = nn.Linear(input_size, 44)
        self.layer2 = nn.Linear(44, 22)
        self.layer3 = nn.Linear(22, 11)
        self.output_layer = nn.Linear(11, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    # Load and prepare data
    djw = pd.read_csv("djw.csv")
    djw.set_index(pd.to_datetime(djw["date"]), inplace=True)
    data = pd.read_csv("output_data.csv")
    data.set_index(pd.to_datetime(data["date_elected"]), inplace=True)
    # data.dropna(inplace=True)

    # Create features and labels
    features = [
        "prev_held_office_democratic",
        "prev_held_office_republican",
        "previous_party_1",
        "previous_party_2",
        "previous_party_3",
        "3-6_month_market_direction",
        "6-12_month_market_direction",
        "12-18_month_direction",
        "party",
    ]

    label = "1_after"  # Replace with your actual label column

    # Now to features we need to put stock market data from djw.csv
    # We will use the closing value of the Dow Jones World Index
    features_closing_diffs = []

    # encode each month return before the election, for election not done yet use 0
    # do 1 year as another feature too
    # do 1, 5, 7 , 1 month after election prediction

    # Split data into training and test sets
    train_size = int(0.99 * len(data))
    train_set, test_set = data[:train_size], data[train_size:]

    # Create datasets
    train_dataset = ElectionDataset(train_set, features, label)
    test_dataset = ElectionDataset(test_set, features, label)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Model, loss, and optimizer
    model = DNNRegressor(input_size=len(features))
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()

        # Print epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Evaluate the model
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            print(outputs)
            loss = criterion(outputs, targets.view(-1, 1))
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader)}")

    # Now we want to predict the very last row in the test set
    inputs, targets = test_dataset[-1]
    inputs = inputs.unsqueeze(0)
    inputs[torch.isnan(inputs)] = 0
    outputs = model(inputs)
    print(f"Predicted: {outputs.item()}, Actual: {targets.item()}")
    if outputs.item() > 0.5:
        print("Predicted: up")
    else:
        print("Predicted: down")

    # Now do the same training but include the expected winner as a feature and predict the market direction for the next month
