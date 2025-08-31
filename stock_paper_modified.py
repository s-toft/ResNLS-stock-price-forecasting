'''
Extended ResNLS: Multi-feature Stock Price Forecasting (LEAKY VERSION)
Changes made:
1. Added OHLC features (open, high, low and close) instead of only the closing price.
2. Adjusted the data preprocessing functions to handle multivariate input (OHLC).
3. Updated the network definition so that the convolutional and LSTM layers operate on multiple features (n_features > 1).
NOTE: This script intentionally keeps the original (leaky) scaling procedure for parity with the original code.
'''

##################### data preprocessing #####################

import math
import numpy as np
import pandas as pd
import baostock as bs
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
df.index=pd.to_datetime(df.date)

bs.logout()

# Convert numeric columns to float
for col in ["open","high","low","close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# keep only the features we have 
features = df[["open", "high", "low", "close"]]

# Scale all features (for model input)  **LEAKY (fit on full series)**
scaler_all = MinMaxScaler()
scaled_features = scaler_all.fit_transform(features)

# Scale only close price (for prediction target)  **LEAKY (fit on full series)**
scaler_close = MinMaxScaler()
scaled_close = scaler_close.fit_transform(df[["close"]])

# function to split into sequences
def split_data(dataset, scaled_close, train_day, predict_day):
    x, y = [], []
    for i in range(train_day, len(dataset) - predict_day + 1):
        # input: all features (scaled together)
        x.append(dataset[i-train_day:i, :])
        # output: close (scaled separately!)
        y.append(scaled_close[i+predict_day-1, 0])
    return np.array(x), np.array(y)

# reshape helper
def reshape_data(train_data, test_data, scaled_close, days):
    x_train, y_train = split_data(train_data, scaled_close[:len(train_data)], days, 1)
    x_test, y_test = split_data(test_data, scaled_close[len(train_data)-days:], days, 1)

    # reshape into (samples, channels, timesteps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    x_test  = np.reshape(x_test,  (x_test.shape[0],  1, x_test.shape[1],  x_test.shape[2]))
    return x_train, y_train, x_test, y_test

# Split into training and test sets
# 90.87% training, 9.13% test (based on original code)
training_data_len = math.ceil(len(scaled_features) * 0.9087)
train_data = scaled_features[0:training_data_len, :]
test_data  = scaled_features[training_data_len-5:, :]

x_train_5, y_train_5, x_test_5, y_test_5 = reshape_data(train_data, test_data, scaled_close, 5)
print("when sequence length is 5, data shape:", x_train_5.shape, y_train_5.shape, x_test_5.shape, y_test_5.shape)

##################### model definition #####################

import torch
import torch.nn as nn
import torch.optim as optim

# initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# infer shapes from training data
n_timesteps = x_train_5.shape[2]   # 5
n_features  = x_train_5.shape[3]   # 4
n_hidden = 64

class ResNLS(nn.Module):
    def __init__(self):
        super(ResNLS, self).__init__()

        # attention/residual weight
        self.weight = nn.Parameter(torch.zeros(1)).to(device)

        # CNN over (timesteps × features)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_hidden, kernel_size=(3, n_features), stride=1, padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_hidden, eps=1e-5),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_hidden, eps=1e-5),

            nn.Flatten(),
            nn.Linear(n_timesteps * n_hidden, n_timesteps)
        )

        # LSTM expects (batch, seq_len, input_size)
        self.lstm = nn.LSTM(n_timesteps, n_hidden, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):
        # x: (batch, 1, timesteps, features)
        cnn_output = self.cnn(x)                # (batch, timesteps)
        cnn_output = cnn_output.view(-1, 1, n_timesteps)

        # residual connection: use last feature (close) as the base, plus weighted CNN signal
        residuals = x[:, :, :, -1]  # last feature (=close)
        residuals = residuals.view(-1, 1, n_timesteps)
        residuals = residuals + self.weight * cnn_output

        # LSTM
        _, (h_n, _) = self.lstm(residuals)
        y_hat = self.linear(h_n[0,:,:])

        return y_hat

# instantiate model
model = ResNLS().to(device)

##################### model training #####################

batch_size = 64
epochs = 50
learning_rate = 0.001

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train_5, dtype=torch.float32).to(device),
                                               torch.tensor(y_train_5, dtype=torch.float32).to(device))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_input = torch.tensor(x_test_5, dtype=torch.float32).to(device)
val_target = torch.tensor(y_test_5, dtype=torch.float32).to(device)

for epoch in range(1, epochs+1):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb).squeeze()
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_output = model(val_input).squeeze()
        val_loss = criterion(val_output, val_target).item()

    print(f"Epoch {epoch:3d}, train loss: {epoch_loss/len(train_loader):.4f}, val loss: {val_loss:.4f}")

##################### model validation #####################

# --- Get predictions on test set ---
predictions = model(val_input).cpu().detach().numpy()

# Inverse transform to original price scale (full-timeline plot arrays)
predictions = scaler_close.inverse_transform(predictions.reshape(-1,1)).flatten()
actuals = scaler_close.inverse_transform(y_test_5.reshape(-1,1)).flatten()

# --- Main evaluation metrics (Extended ResNLS, LEAKY) ---
mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
print(f"[Extended LEAKY] Test MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# --- Plot: Training vs Validation (full timeline) ---
train = df[["close"]][:training_data_len]
valid = df[["close"]][training_data_len:]
valid["predictions"] = predictions

plt.figure(figsize=(12,6))
plt.plot(train["close"], label="Training Data")
plt.plot(valid["close"], label="Actual Price", color="blue")
plt.plot(valid["predictions"], label="Predicted Price", color="red", linestyle="--")
plt.title("Stock Price Prediction (Extended ResNLS, LEAKY)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# --- Test-only zoom (original price scale) ---
# Use the val_output we computed inside the last epoch loop
test_targets_inv = scaler_close.inverse_transform(val_target.cpu().numpy().reshape(-1,1)).flatten()
test_preds_inv   = scaler_close.inverse_transform(val_output.cpu().numpy().reshape(-1,1)).flatten()

plt.figure(figsize=(12,6))
plt.plot(test_targets_inv, label="Actual Close (test)", color="black")
plt.plot(test_preds_inv, label="Predicted Close (test)", color="red", linestyle="--")
plt.title("Test Set Predictions (Original Price Scale) — Extended LEAKY")
plt.xlabel("Time step")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# --- Save extended (leaky) arrays for later comparisons ---
extended_leaky_preds   = np.asarray(test_preds_inv, dtype=np.float64)
extended_leaky_targets = np.asarray(test_targets_inv, dtype=np.float64)
np.save("extended_leaky_preds.npy", extended_leaky_preds)
np.save("extended_leaky_targets.npy", extended_leaky_targets)
print(f"[Saved] extended_leaky_preds.npy shape={extended_leaky_preds.shape}, "
      f"extended_leaky_targets.npy shape={extended_leaky_targets.shape}")

# --- Paired ΔMAE comparison with baseline (LEAKY) ---
# Load baseline LEAKY arrays saved by the close-only baseline script
baseline_preds_inv   = np.load("baseline_leaky_preds.npy")
baseline_targets_inv = np.load("baseline_leaky_targets.npy")

# Sanity check (optional): align lengths if needed
min_len = min(len(test_targets_inv), len(baseline_targets_inv),
              len(test_preds_inv),   len(baseline_preds_inv))
bt = baseline_targets_inv[:min_len]
bp = baseline_preds_inv[:min_len]
et = test_targets_inv[:min_len]
ep = test_preds_inv[:min_len]

# Absolute errors per sample
err_baseline = np.abs(bt - bp)
err_extended = np.abs(et - ep)
delta = err_baseline - err_extended

print(f"Mean ΔMAE (baseline - extended, LEAKY): {np.mean(delta):.2f}")

# Bootstrap CI for ΔMAE
n_boot = 10000
boot_means = []
for _ in range(n_boot):
    sample = np.random.choice(delta, size=len(delta), replace=True)
    boot_means.append(np.mean(sample))
boot_means = np.array(boot_means)
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"ΔMAE mean (LEAKY): {np.mean(delta):.2f}, 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
