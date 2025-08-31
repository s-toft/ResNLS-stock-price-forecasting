'''
Baseline ResNLS (Close-only) — LEAKAGE-FREE VERSION
Structure:
1) data preprocessing (train-only scaling; 5-day overlap at split)
2) model definition (original baseline architecture)
3) model training
4) model validation + save clean baseline arrays
'''

##################### data preprocessing #####################

# !pip install baostock

import math
import numpy as np
import pandas as pd
import baostock as bs
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# download CSI dataset
lg = bs.login()
fields= "date,open,high,low,close"
rs = bs.query_history_k_data_plus("sh.000001", fields, start_date="2012-01-01", end_date="2022-12-31", frequency="d", adjustflag="2")
data_list = []
while (rs.error_code == "0") & rs.next():
    data_list.append(rs.get_row_data())
df = pd.DataFrame(data_list, columns=rs.fields)
df.index = pd.to_datetime(df.date)
bs.logout()

# close-only series
data = pd.DataFrame(pd.to_numeric(df["close"]))
dataset = np.reshape(data.values, (df.shape[0], 1))

def split_data(dataset, train_day, predict_day):
    x, y = [], []
    for i in range(train_day, len(dataset)-predict_day+1):
        x.append(dataset[i-train_day : i, 0])
        y.append(dataset[i+predict_day-1, 0])
    return x, y

def reshape_data(train_data, test_data, days):
    x_train, y_train = split_data(train_data, days, 1)
    x_test, y_test   = split_data(test_data,  days, 1)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test,  y_test  = np.array(x_test),  np.array(y_test)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test  = np.reshape(x_test,  (x_test.shape[0],  1, x_test.shape[1]))
    return x_train, y_train, x_test, y_test

# --- Leakage-free scaling: fit scaler on TRAIN ONLY, then transform chronologically ---
training_data_len = math.ceil(len(dataset) * 0.9087)

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(dataset[:training_data_len])                 # fit on training portion only

# transform train and test with the fitted scaler (keep 5-day overlap for test windows)
train_data = scaler.transform(dataset[:training_data_len])
test_data  = scaler.transform(dataset[training_data_len-5:])

# windows
x_train_5, y_train_5, x_test_5, y_test_5 = reshape_data(train_data, test_data, 5)
print("when sequence length is 5, data shape:", x_train_5.shape, y_train_5.shape, x_test_5.shape, y_test_5.shape)


##################### model definition #####################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_input = 5; n_hidden = 64

class ResNLS(nn.Module):
    def __init__(self):
        super(ResNLS, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1)).to(device)  # residual scalar

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden, eps=1e-5),
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_hidden, eps=1e-5),

            nn.Flatten(),
            nn.Linear(n_input * n_hidden, n_input)
        )

        self.lstm  = nn.LSTM(n_input, n_hidden, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):
        cnn_output = self.cnn(x)
        cnn_output = cnn_output.view(-1, 1, n_input)

        residuals = x + self.weight * cnn_output  # (kept as in original baseline)
        # NOTE: original baseline feeds x to LSTM (not residuals); kept for parity:
        _, (h_n, _)  = self.lstm(x)
        y_hat = self.linear(h_n[0,:,:])
        return y_hat


##################### model training #####################

val_input = torch.tensor(x_test_5, dtype=torch.float).to(device)
val_target = torch.tensor(y_test_5, dtype=torch.float).to(device)

epochs = 50; batch_size = 64
model = ResNLS().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if x_train_5.shape[0] % batch_size == 0:
    batch_num = int(x_train_5.shape[0] / batch_size)
else:
    batch_num = int(x_train_5.shape[0] / batch_size) + 1

for epoch in range(epochs):
    for j in range(batch_num):
        train_input = torch.tensor(x_train_5[j * batch_size : (j+1) * batch_size], dtype=torch.float).to(device)
        train_targe = torch.tensor(y_train_5[j * batch_size : (j+1) * batch_size], dtype=torch.float).to(device)

        model.train()
        optimizer.zero_grad()
        train_output = model(train_input)
        train_loss = criterion(train_output, train_targe)
        train_loss.backward()
        optimizer.step()

    if (epoch+1) % (epochs//20 if epochs>=20 else 1) == 0:
        with torch.no_grad():
            model.eval()
            val_output = model(val_input)
            val_loss = criterion(val_output, val_target)
            print("Epoch: {:>3}, train loss: {:.4f}, val loss: {:.4f}".format(epoch+1, train_loss.item(), val_loss.item()))


##################### model validation #####################

from sklearn import metrics
import matplotlib.pyplot as plt

# predictions (inverse-transform with the TRAIN-FIT scaler)
predictions = model(val_input)
predictions = scaler.inverse_transform(predictions.cpu().detach().numpy())

train = data[:training_data_len]
valid = data[training_data_len:]
valid["predictions"] = predictions

y     = np.array(valid["close"])
y_hat = np.array(valid["predictions"])
mae   = metrics.mean_absolute_error(y_hat, y)
mse   = metrics.mean_squared_error(y_hat, y)
rmse  = metrics.mean_squared_error(y_hat, y) ** 0.5
print("[Baseline CLEAN] MAE:{:.2F}   MSE:{:.2f}   RMSE:{:.2F}".format(mae, mse, rmse))

# Save CLEAN baseline arrays for comparisons with CLEAN extended runs
baseline_clean_preds   = predictions.flatten()
baseline_clean_targets = y.flatten()
np.save("baseline_clean_preds.npy", baseline_clean_preds)
np.save("baseline_clean_targets.npy", baseline_clean_targets)
print("[Saved] baseline_clean_preds.npy shape=", baseline_clean_preds.shape,
      " baseline_clean_targets.npy shape=", baseline_clean_targets.shape)

# Plot
plt.figure(figsize=(12,6))
plt.plot(train["close"], label="Training Data")
plt.plot(valid["close"], label="Actual Price", color="blue")
plt.plot(valid["predictions"], label="Predicted Price", color="red", linestyle="--")
plt.title("Stock Price Prediction (ResNLS — close-only, CLEAN scaling)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
