'''
Extended ResNLS (OHLC) — LEAKAGE-FREE VERSION
Changes:
1) OHLC inputs (open, high, low, close) instead of close-only
2) Preprocessing adjusted for multivariate input
3) CNN/LSTM operate on multiple features (n_features > 1)
4) Leakage-free scaling: fit scalers on TRAIN ONLY; transform val/test chronologically
'''

##################### data preprocessing #####################

import math
import numpy as np
import pandas as pd
import baostock as bs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# --- Download CSI dataset ---
lg = bs.login()
fields = "date,open,high,low,close"
rs = bs.query_history_k_data_plus(
    "sh.000001", fields,
    start_date="2012-01-01",
    end_date="2022-12-31",
    frequency="d", adjustflag="2"
)

data_list = []
while (rs.error_code == "0") & rs.next():
    data_list.append(rs.get_row_data())
df = pd.DataFrame(data_list, columns=rs.fields)
df.index = pd.to_datetime(df.date)
bs.logout()

# Convert numeric columns to float
for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep only OHLC features
features = df[["open", "high", "low", "close"]].copy()

# ---------------------- LEAKAGE-FREE SCALING ----------------------
WINDOW = 5  # sequence length
n_total = len(features)
training_data_len = math.ceil(n_total * 0.9087)  # ~90.87% train (chronological split)

# Fit scalers on TRAIN ONLY
scaler_all = MinMaxScaler()
scaler_all.fit(features.iloc[:training_data_len])           # fit on training features only

scaler_close = MinMaxScaler()
scaler_close.fit(df[["close"]].iloc[:training_data_len])    # fit on training close only

# Transform with TRAIN-FIT scalers
train_data = scaler_all.transform(features.iloc[:training_data_len].values)
# include WINDOW overlap so test windows have sufficient context
test_data  = scaler_all.transform(features.iloc[training_data_len - WINDOW:].values)

# Close series transformed (safe: fit on train only)
scaled_close_full = scaler_close.transform(df[["close"]].values)
# -----------------------------------------------------------------

# Sequence utilities
def split_data(dataset, scaled_close, train_day, predict_day):
    x, y = [], []
    for i in range(train_day, len(dataset) - predict_day + 1):
        x.append(dataset[i - train_day : i, :])             # inputs: multivariate
        y.append(scaled_close[i + predict_day - 1, 0])      # target: scaled close
    return np.array(x), np.array(y)

def reshape_data(train_data, test_data, scaled_close, days):
    x_train, y_train = split_data(train_data, scaled_close[:len(train_data)], days, 1)
    x_test,  y_test  = split_data(test_data,  scaled_close[len(train_data) - days:], days, 1)
    # reshape to (batch, channels=1, timesteps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    x_test  = np.reshape(x_test,  (x_test.shape[0],  1, x_test.shape[1],  x_test.shape[2]))
    return x_train, y_train, x_test, y_test

x_train_5, y_train_5, x_test_5, y_test_5 = reshape_data(train_data, test_data, scaled_close_full, WINDOW)
print("when sequence length is 5, data shape:",
      x_train_5.shape, y_train_5.shape, x_test_5.shape, y_test_5.shape)

##################### model definition #####################

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# infer shapes from training data
n_timesteps = x_train_5.shape[2]   # 5
n_features  = x_train_5.shape[3]   # 4
n_hidden = 64

class ResNLS(nn.Module):
    def __init__(self):
        super(ResNLS, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1)).to(device)  # residual scalar

        # CNN over (timesteps × features)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_hidden,
                      kernel_size=(3, n_features), stride=1, padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_hidden, eps=1e-5),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=n_hidden, out_channels=n_hidden,
                      kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_hidden, eps=1e-5),

            nn.Flatten(),
            nn.Linear(n_timesteps * n_hidden, n_timesteps)
        )

        # LSTM expects (batch, seq_len, input_size)
        self.lstm  = nn.LSTM(n_timesteps, n_hidden, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):
        # x: (batch, 1, timesteps, features)
        cnn_output = self.cnn(x)                         # -> (batch, timesteps)
        cnn_output = cnn_output.view(-1, 1, n_timesteps)

        # residual on the close channel (last feature)
        residuals = x[:, :, :, -1].view(-1, 1, n_timesteps)
        residuals = residuals + self.weight * cnn_output

        # LSTM + final linear
        _, (h_n, _) = self.lstm(residuals)
        y_hat = self.linear(h_n[0, :, :])
        return y_hat

model = ResNLS().to(device)

##################### model training #####################

batch_size = 64
epochs = 50
learning_rate = 0.001

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(x_train_5, dtype=torch.float32).to(device),
    torch.tensor(y_train_5, dtype=torch.float32).to(device)
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_input  = torch.tensor(x_test_5, dtype=torch.float32).to(device)
val_target = torch.tensor(y_test_5, dtype=torch.float32).to(device)

for epoch in range(1, epochs+1):
    model.train()
    epoch_loss = 0.0
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

# Recompute on val to be explicit
with torch.no_grad():
    val_output = model(val_input).squeeze()

# Predictions -> original price scale (scaler_close was fit on TRAIN only)
predictions = scaler_close.inverse_transform(val_output.cpu().numpy().reshape(-1,1)).flatten()
actuals     = scaler_close.inverse_transform(val_target.cpu().numpy().reshape(-1,1)).flatten()

np.save("extended_clean_preds.npy",   predictions.flatten())
np.save("extended_clean_targets.npy", actuals.flatten())

# Metrics for Extended (CLEAN)
mae  = mean_absolute_error(actuals, predictions)
mse  = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
print(f"[Extended CLEAN] Test MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# --- Plot: Training vs Validation (full timeline) ---
train = df[["close"]][:training_data_len]
valid = df[["close"]][training_data_len:]
valid["predictions"] = predictions

plt.figure(figsize=(12,6))
plt.plot(train["close"], label="Training Data")
plt.plot(valid["close"], label="Actual Price", color="blue")
plt.plot(valid["predictions"], label="Predicted Price", color="red", linestyle="--")
plt.title("Stock Price Prediction (Extended ResNLS — CLEAN scaling)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# --- Plot: Test-only zoom (original price scale) ---
test_targets_inv = actuals
test_preds_inv   = predictions

plt.figure(figsize=(12,6))
plt.plot(test_targets_inv, label="Actual Close (test)", color="black")
plt.plot(test_preds_inv, label="Predicted Close (test)", color="red", linestyle="--")
plt.title("Test Set Predictions (Original Price Scale) — CLEAN")
plt.xlabel("Time step")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# --- Paired ΔMAE comparison with CLEAN baseline ---
# Expect the following files from the CLEAN baseline run:
#   baseline_clean_preds.npy, baseline_clean_targets.npy
try:
    baseline_preds_inv   = np.load("baseline_clean_preds.npy")
    baseline_targets_inv = np.load("baseline_clean_targets.npy")

    # Length alignment safeguard (should match; if not, crop to min)
    m = min(len(test_targets_inv), len(baseline_preds_inv))
    if (len(test_targets_inv) != len(baseline_preds_inv)) or (len(baseline_targets_inv) != len(baseline_preds_inv)):
        print(f"[WARN] Length mismatch; aligning to min length {m}.")
    y_true     = test_targets_inv[:m]
    y_base_hat = baseline_preds_inv[:m]
    y_ext_hat  = test_preds_inv[:m]

    # ΔMAE = |e_base| - |e_ext| (positive => extended better)
    err_baseline = np.abs(y_true - y_base_hat)
    err_extended = np.abs(y_true - y_ext_hat)
    delta = err_baseline - err_extended

    print(f"Mean ΔMAE (baseline - extended): {np.mean(delta):.2f}")

    # Bootstrap CI
    n_boot = 10000
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(delta, size=len(delta), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"ΔMAE mean: {np.mean(delta):.2f}, 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")

    # Also print CLEAN baseline metrics on the same y_true
    mae_base  = mean_absolute_error(y_true, y_base_hat)
    mse_base  = mean_squared_error(y_true, y_base_hat)
    rmse_base = np.sqrt(mse_base)
    print(f"[Baseline CLEAN] Test MAE: {mae_base:.2f}, MSE: {mse_base:.2f}, RMSE: {rmse_base:.2f}")

except FileNotFoundError:
    print("[INFO] baseline_clean_*.npy not found. Run the CLEAN baseline script first to enable ΔMAE comparison.")
