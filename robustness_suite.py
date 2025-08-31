# robustness_suite.py
# Leakage-free seeds × ablations × window lengths for ResNLS, LSTM-only, and Persistence

import math, warnings, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import baostock as bs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------- helpers ----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_csi_ohlc():
    lg = bs.login()
    fields = "date,open,high,low,close"
    rs = bs.query_history_k_data_plus("sh.000001", fields,
                                      start_date="2012-01-01",
                                      end_date="2022-12-31",
                                      frequency="d", adjustflag="2")
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)
    bs.logout()
    df.index = pd.to_datetime(df.date)
    for col in ["open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df

def leakage_free_scale_and_windows(df, feature_cols, L=5, train_ratio=0.9087):
    assert feature_cols[-1] == "close", "Put 'close' last in feature_cols!"
    feats = df[feature_cols].copy()
    n_total = len(feats)
    train_len = math.ceil(n_total * train_ratio)

    scaler_all = MinMaxScaler()
    scaler_all.fit(feats.iloc[:train_len])
    scaler_close = MinMaxScaler()
    scaler_close.fit(df[["close"]].iloc[:train_len])

    train_data = scaler_all.transform(feats.iloc[:train_len].values)
    test_data  = scaler_all.transform(feats.iloc[train_len - L:].values)

    scaled_close_full = scaler_close.transform(df[["close"]].values)

    def split_data(dataset, scaled_close, train_day, predict_day=1):
        x, y = [], []
        for i in range(train_day, len(dataset) - predict_day + 1):
            x.append(dataset[i-train_day:i, :])
            y.append(scaled_close[i + predict_day - 1, 0])
        return np.array(x), np.array(y)

    x_tr, y_tr = split_data(train_data, scaled_close_full[:len(train_data)], L, 1)
    x_te, y_te = split_data(test_data, scaled_close_full[len(train_data)-L:], L, 1)

    x_tr = x_tr.reshape(x_tr.shape[0], 1, x_tr.shape[1], x_tr.shape[2])
    x_te = x_te.reshape(x_te.shape[0], 1, x_te.shape[1], x_te.shape[2])

    return (x_tr, y_tr, x_te, y_te, scaler_close, train_len)

# ---------------------- models ----------------------
class ResNLS(nn.Module):
    def __init__(self, n_timesteps, n_features, n_hidden=64, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1)).to(device)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=(3, n_features), stride=1, padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_hidden, eps=1e-5),
            nn.Dropout(0.1),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_hidden, eps=1e-5),
            nn.Flatten(),
            nn.Linear(n_timesteps * n_hidden, n_timesteps)
        )
        self.lstm   = nn.LSTM(n_timesteps, n_hidden, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):
        n_timesteps = x.shape[2]
        cnn_output = self.cnn(x).view(-1, 1, n_timesteps)
        residuals  = x[:, :, :, -1].view(-1, 1, n_timesteps)
        residuals  = residuals + self.weight * cnn_output
        _, (h_n, _) = self.lstm(residuals)
        return self.linear(h_n[0, :, :])

class LSTMOnly(nn.Module):
    def __init__(self, n_timesteps, n_features, n_hidden=64):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=n_timesteps, hidden_size=n_hidden, batch_first=True)
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):
        B, _, T, F = x.shape
        seq = x.view(B, F, T)
        _, (h_n, _) = self.lstm(seq)
        return self.linear(h_n[0, :, :])

# ---------------------- training/eval ----------------------
def train_torch_model(model, x_tr, y_tr, x_te, y_te, epochs=30, batch_size=64, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ds = torch.utils.data.TensorDataset(torch.tensor(x_tr, dtype=torch.float32),
                                        torch.tensor(y_tr, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    x_te_t = torch.tensor(x_te, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.float32, device=device)

    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_te = model(x_te_t).squeeze().cpu().numpy()
    return preds_te

def metrics_from_raw(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def run_one(df, feature_cols, L, seed, train_ratio=0.9087, epochs=30, device="cpu"):
    set_seed(seed)
    x_tr, y_tr, x_te, y_te, scaler_close, train_len = leakage_free_scale_and_windows(
        df, feature_cols, L=L, train_ratio=train_ratio
    )
    nT, nF = x_tr.shape[2], x_tr.shape[3]

    # ---- ResNLS ----
    resnls = ResNLS(n_timesteps=nT, n_features=nF, n_hidden=64, device=device)
    preds_res_scaled = train_torch_model(resnls, x_tr, y_tr, x_te, y_te, epochs=epochs, device=device)
    y_te_inv   = scaler_close.inverse_transform(y_te.reshape(-1,1)).flatten()
    pred_res_inv = scaler_close.inverse_transform(preds_res_scaled.reshape(-1,1)).flatten()
    mae_r, mse_r, rmse_r = metrics_from_raw(y_te_inv, pred_res_inv)

    # ---- LSTM-only ----
    lstm_only = LSTMOnly(n_timesteps=nT, n_features=nF, n_hidden=64)
    preds_lstm_scaled = train_torch_model(lstm_only, x_tr, y_tr, x_te, y_te, epochs=epochs, device=device)
    pred_lstm_inv = scaler_close.inverse_transform(preds_lstm_scaled.reshape(-1,1)).flatten()
    mae_l, mse_l, rmse_l = metrics_from_raw(y_te_inv, pred_lstm_inv)

    # ---- Persistence ----
    # prediction = last observed close in each input window
    persist_scaled = np.array([x_te[i, 0, -1, -1] for i in range(len(x_te))])
    persist_inv = scaler_close.inverse_transform(persist_scaled.reshape(-1,1)).flatten()
    mae_p, mse_p, rmse_p = metrics_from_raw(y_te_inv, persist_inv)

    return {
        "y_true_inv": y_te_inv,
        "pred_res_inv": pred_res_inv,
        "pred_lstm_inv": pred_lstm_inv,
        "pred_persist_inv": persist_inv,
        "mae_res": mae_r, "mse_res": mse_r, "rmse_res": rmse_r,
        "mae_lstm": mae_l, "mse_lstm": mse_l, "rmse_lstm": rmse_l,
        "mae_pers": mae_p, "mse_pers": mse_p, "rmse_pers": rmse_p
    }

# ---------------------- run the suite ----------------------
def run_suite(seeds=(0,1,2,3,4), L_list=(5,10), feature_sets=("C","CH","CHL","OHLC"),
              train_ratio=0.9087, epochs=30, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = load_csi_ohlc()

    feat_map = {
        "C":   ["close"],
        "CH":  ["high","close"],
        "CHL": ["high","low","close"],
        "OHLC":["open","high","low","close"],
    }

    rows = []
    for L in L_list:
        for fcode in feature_sets:
            fcols = feat_map[fcode]
            for seed in seeds:
                out = run_one(df, fcols, L, seed, train_ratio=train_ratio, epochs=epochs, device=device)
                rows.append({
                    "feature_set": fcode, "L": L, "seed": seed,
                    "MAE_ResNLS": out["mae_res"], "RMSE_ResNLS": out["rmse_res"], "MSE_ResNLS": out["mse_res"],
                    "MAE_LSTM":   out["mae_lstm"], "RMSE_LSTM":   out["rmse_lstm"], "MSE_LSTM":   out["mse_lstm"],
                    "MAE_Persist":out["mae_pers"], "RMSE_Persist":out["rmse_pers"], "MSE_Persist":out["mse_pers"],
                })
    return pd.DataFrame(rows)

def summarize_and_plot(df_res, save_prefix="robust"):
    agg = df_res.groupby(["feature_set","L"]).agg(["mean","std"])
    print("\n=== Mean ± Std across seeds (MAE/RMSE) ===")
    for (fs, L), _ in agg.iterrows():
        print(f"[{fs}, L={L}]  "
              f"ResNLS {agg.loc[(fs,L), ('MAE_ResNLS','mean')]:.2f}±{agg.loc[(fs,L), ('MAE_ResNLS','std')]:.2f} | "
              f"LSTM {agg.loc[(fs,L), ('MAE_LSTM','mean')]:.2f}±{agg.loc[(fs,L), ('MAE_LSTM','std')]:.2f} | "
              f"Persist {agg.loc[(fs,L), ('MAE_Persist','mean')]:.2f}±{agg.loc[(fs,L), ('MAE_Persist','std')]:.2f}")

    # Bar plot for L=5
    L_fix = 5
    sub = df_res[df_res["L"]==L_fix]
    fig, ax = plt.subplots(figsize=(7,4))
    order = ["C","CH","CHL","OHLC"]
    width = 0.25
    x = np.arange(len(order))
    def barvals(col):
        return [sub[sub["feature_set"]==k][col].mean() for k in order], \
               [sub[sub["feature_set"]==k][col].std() for k in order]
    r_m, r_s = barvals("MAE_ResNLS")
    l_m, l_s = barvals("MAE_LSTM")
    p_m, p_s = barvals("MAE_Persist")
    ax.bar(x- width, r_m, width, yerr=r_s, label="ResNLS", alpha=0.8)
    ax.bar(x,       l_m, width, yerr=l_s, label="LSTM-only", alpha=0.8)
    ax.bar(x+ width,p_m, width, yerr=p_s, label="Persistence", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("MAE (price units)")
    ax.set_title(f"MAE vs. input set (L={L_fix})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_ablation_bars.png", dpi=300)

    # Line plot for window length
    fig, ax = plt.subplots(figsize=(7,4))
    for fs, label in [("C","Close-only"), ("OHLC","OHLC")]:
        m = df_res[df_res["feature_set"]==fs].groupby("L")["MAE_ResNLS"].agg(["mean","std"])
        ax.plot(m.index, m["mean"], marker="o", label=f"ResNLS ({label})")
        ax.fill_between(m.index, m["mean"]-m["std"], m["mean"]+m["std"], alpha=0.15)
    ax.set_xlabel("Window length L"); ax.set_ylabel("MAE (price units)")
    ax.set_title("MAE vs. window length")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_mae_vs_L.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df_results = run_suite(seeds=(0,1,2,3,4), L_list=(5,10),
                           feature_sets=("C","CH","CHL","OHLC"),
                           epochs=25)
    df_results.to_csv("robust_results.csv", index=False)
    summarize_and_plot(df_results, save_prefix="robust")
