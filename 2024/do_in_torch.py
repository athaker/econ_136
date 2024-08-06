# Imports
import os
import pandas as pd
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
import numpy as np


torch.manual_seed(0)  # set seed for reproducability


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
    """
    Computes the percent return of `djw ` for a specified number of`days` before
    the passed in `index` date being very careful to never choose a future date
    that could create a forward looking bias
    """
    if (index + timedelta(days=1)) > djw.index[-1]:
        return 0

    ntd = djw.truncate(after=index).iloc[-1]["close"]

    if days > 0:
        n_days_after = djw[index : index + timedelta(days=days)].iloc[-1]["close"]
        pct = (n_days_after - ntd) / n_days_after
    else:
        n_days_before_price = djw[index + timedelta(days=days) : index].iloc[0]["close"]
        pct = (ntd - n_days_before_price) / ntd
    return pct


# TODO: shaply feature attribution map


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


def run_model(train_loader, test_loader):
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

    # Evaluate the model
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            total_loss += loss.item()
    return model


if __name__ == "__main__":
    # Load and prepare data
    djw = pd.read_csv("djw.csv")
    djw.set_index(pd.to_datetime(djw["date"]), inplace=True)
    data = pd.read_csv("output_data.csv")
    data.set_index(pd.to_datetime(data["date_elected"]), inplace=True)

    # Create features
    features = [
        "prev_held_office_democratic",
        "prev_held_office_republican",
        "previous_party_1",
        "previous_party_2",
        "previous_party_3",
        "3-6_month_market_direction",
        "6-12_month_market_direction",
        "12-18_month_direction",
        "sentiment",
        "day_before_7",
        "day_before_30",
        "day_before_150",
        "day_before_210",
        "day_before_365",
    ]

    # encode stock pct returns leading up to election
    day_before_7 = []
    day_before_30 = []
    day_before_150 = []
    day_before_210 = []
    day_before_365 = []
    for index, row in data.iterrows():
        day_before_7.append(compute_td_pct(djw, index, -7))
        day_before_30.append(compute_td_pct(djw, index, -30))
        day_before_150.append(compute_td_pct(djw, index, -150))
        day_before_210.append(compute_td_pct(djw, index, -210))
        day_before_365.append(compute_td_pct(djw, index, -365))

    data["day_before_7"] = day_before_7
    data["day_before_30"] = day_before_30
    data["day_before_150"] = day_before_150
    data["day_before_210"] = day_before_210
    data["day_before_365"] = day_before_365

    label = "party"  # what we will predict

    # Split data into training and test sets
    train_size = int(0.99 * len(data))
    train_set, test_set = data[:train_size], data[train_size:]

    # Create datasets
    train_dataset = ElectionDataset(train_set, features, label)
    test_dataset = ElectionDataset(test_set, features, label)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # run the training loop and evaluate the model
    model = run_model(train_loader, test_loader)

    # Now we want to predict the very last row in the test set
    inputs, targets = test_dataset[-1]
    inputs = inputs.unsqueeze(0)
    outputs = model(inputs)
    print(f"Predicted: {outputs.item()}")
    if outputs.item() > 0.5:
        print("Predicted: Republican")
        data.loc[data.index[-1], "party"] = 1.0
    else:
        print("Predicted: Democratic")
        data.loc[data.index[-1], "party"] = 0.0

    torch.onnx.export(model, inputs, "party_model.onnx")

    # Now do the same training but include the expected winner as a feature and predict the market direction for the next month
    features.append("party")
    label = "1_after"  # this time we will predict market direction, up or down

    # set up the datasets again with the new feature and label
    train_set, test_set = data[:train_size], data[train_size:]
    train_dataset = ElectionDataset(train_set, features, label)
    test_dataset = ElectionDataset(test_set, features, label)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # run the training loop and evaluate the model
    model = run_model(train_loader, test_loader)

    # Now we want to predict the very last row in the test set
    inputs, targets = test_dataset[-1]
    inputs = inputs.unsqueeze(0)
    outputs = model(inputs)
    print(f"Predicted: {outputs.item()}")
    if outputs.item() > 0.5:
        print("Predicted: up")
        data.iloc[-1] = 1.0
    else:
        print("Predicted: down")
        data.iloc[-1] = 0.0
    torch.onnx.export(model, inputs, "market_model.onnx")
