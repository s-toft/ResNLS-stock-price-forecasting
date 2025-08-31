# ResNLS-stock-price-forecasting
Original ResNLS repository: https://github.com/Yuanzhe-Jia/ResNLS?tab=readme-ov-file

This repository contains Python code to reproduce and extend the ResNLS 
model for daily stock‐price prediction. ResNLS combines a small residual
convolutional network with an LSTM to forecast the next day’s closing
price from a fixed‐length sliding window of historical prices. The
project includes the original close‐only model, an extended variant with
open/high/low/close (OHLC) inputs, leakage‐free versions, and a
robustness suite that explores different input sets and random seeds.

Key Scripts:
- stock_paper.py – Baseline ResNLS using the last five closing
prices (L=5) as input. Applies MinMax scaling to the entire data
series before splitting, reproducing the leaky setting of the original
paper.
- stock_paper_modified.py – Extended model that augments each
input window with open, high and low values (OHLC). Uses the same
leaky scaling as the baseline.
- new_stock_paper.py – Leakage‑free baseline. Fits the MinMax
scaler only on the training windows and applies it chronologically to
validation/test, leaving a five‑day overlap at the split boundary.
- new_stock_modified.py – Leakage‑free version of the extended
OHLC model.
- robustness_suite.py – Runs experiments across multiple random
seeds, different window lengths (L=5 and L=10), and input
ablations (close only C, close+high CH, close+high+low CHL, and
full OHLC). Compares the ResNLS models to an LSTM‐only variant and
a naive persistence baseline that predicts tomorrow’s close as
today’s. Outputs summary tables and plots.
- plotting_stock.py – Loads saved prediction/target arrays
(*.npy) and produces composite plots of predicted vs. actual
trajectories and histograms of per‑sample ΔMAE values with bootstrap
confidence intervals.

Usage:
This code requires Python 3 and standard scientific packages such as
NumPy, Pandas, scikit‑learn, Matplotlib and PyTorch. The baostock
package is used to fetch historical daily OHLC data for the
Shanghai Composite Index.

Notes:
The original ResNLS code scales the entire series before splitting,
which leaks information from the test period into the training
transformation. The new_ versions fix this by fitting scalers only on
the training set. In our experiments, the leakage‑free extended model
did not outperform the leakage‑free baseline; a naive persistence
predictor (predicting the next close as the current close) was
surprisingly competitive. Use the robustness suite to explore these
effects under different random seeds and input combinations.


