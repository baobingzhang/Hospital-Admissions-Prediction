#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# PyTorch & Pyro imports
# --------------------------
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal

# --------------------------
# Sklearn imports
# --------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def main():
    # ============================
    # Step 1: Read Raw Data
    # ============================
    data_file = 'data.csv'
    try:
        df_original = pd.read_csv(data_file)
    except Exception as e:
        print("Error reading data file:", e)
        sys.exit(1)

    print("Raw data preview:")
    print(df_original.head())
    print("\nData structure:")
    df_original.info()
    print("\nDescriptive statistics:")
    print(df_original.describe(include='all'))

    # ============================
    # Step 2: Initial Data Processing & Rounding
    # ============================
    df_cleaned = df_original.copy()

    # Convert Date to datetime
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_cleaned['Month'] = df_cleaned['Date'].dt.month

    # Round Temperature, Rainfall, and Humidity to two decimal places
    for col in ['Temperature', 'Rainfall', 'Humidity']:
        df_cleaned[col] = df_cleaned[col].round(2)

    print("\nPreview of processed data (Date, Month, Temperature, Rainfall, Humidity):")
    print(df_cleaned[['Date', 'Month', 'Temperature', 'Rainfall', 'Humidity']].head())

    # ============================
    # Step 3: Fill Missing Values (Using Monthly Average) & Round Up Admissions
    # ============================
    def fill_missing_by_month(df, col, round_up=False):
        """
        Fill missing values in column `col` using the monthly average.
        If round_up is True, round up (ceiling) the value regardless of whether it is missing.
        """
        monthly_avg = df.groupby('Month')[col].mean()

        def fill_func(row):
            if pd.isna(row[col]):
                val = monthly_avg.loc[row['Month']]
            else:
                val = row[col]
            return int(np.ceil(val)) if round_up else val

        df[col] = df.apply(fill_func, axis=1)
        return df

    # Fill missing values for Temperature, Rainfall, and Humidity (retain two decimals)
    for col in ['Temperature', 'Rainfall', 'Humidity']:
        df_cleaned = fill_missing_by_month(df_cleaned, col, round_up=False)
        df_cleaned[col] = df_cleaned[col].round(2)

    # Fill missing values for Male_Admissions and Female_Admissions (round up)
    for col in ['Male_Admissions', 'Female_Admissions']:
        df_cleaned = fill_missing_by_month(df_cleaned, col, round_up=True)

    print("\nMissing values after filling:")
    print(df_cleaned[['Temperature', 'Rainfall', 'Humidity', 'Male_Admissions', 'Female_Admissions']].isna().sum())

    # Compute Total_Admissions as the sum of Male_Admissions and Female_Admissions
    df_cleaned['Total_Admissions'] = df_cleaned['Male_Admissions'] + df_cleaned['Female_Admissions']

    # ============================
    # Step 4: Save Cleaned Data
    # ============================
    df_cleaned.to_csv('cleaned_data.csv', index=False)
    print("\nCleaned data saved to cleaned_data.csv")

    # ============================
    # Step 5: Feature Engineering
    # ============================
    df = pd.read_csv('cleaned_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract Month and Quarter
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter

    # One-hot encode County (if exists)
    if 'County' in df.columns:
        df = pd.get_dummies(df, columns=['County'], drop_first=True)

    # Convert Date to an ordinal number
    df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

    # Define feature columns for modeling
    feature_cols = ['Temperature', 'Rainfall', 'Humidity', 'Population',
                    'Water_Bodies', 'Month', 'Quarter', 'Date_Ordinal']
    if 'County_C2' in df.columns:
        feature_cols.append('County_C2')

    X = df[feature_cols].values
    y = df['Total_Admissions'].values

    # ============================
    # Step 6: Time Series Split
    # ============================
    unique_dates = df['Date'].drop_duplicates().sort_values()
    cutoff_index = int(len(unique_dates) * 0.8)
    cutoff_date = unique_dates.iloc[cutoff_index]

    train_idx = df['Date'] <= cutoff_date
    test_idx = df['Date'] > cutoff_date

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\nNumber of training samples:", X_train.shape[0])
    print("Number of test samples:", X_test.shape[0])
    print("Cutoff date:", cutoff_date)

    # ============================
    # Step 7: Data Normalization
    # ============================
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    import torch
    X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float)
    y_train_torch = torch.tensor(y_train, dtype=torch.float)
    X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float)
    y_test_torch = torch.tensor(y_test, dtype=torch.float)

    # ============================
    # Step 8: Define a Deeper Bayesian Neural Network with Learnable Noise
    # ============================
    class BayesianNN(PyroModule):
        def __init__(self, n_features, hidden_size1=50, hidden_size2=50):
            super().__init__()
            # First layer
            self.linear1 = PyroModule[nn.Linear](n_features, hidden_size1)
            self.linear1.weight = PyroSample(
                dist.Normal(0., 1.).expand([hidden_size1, n_features]).to_event(2)
            )
            self.linear1.bias = PyroSample(
                dist.Normal(0., 1.).expand([hidden_size1]).to_event(1)
            )

            # Second layer
            self.linear2 = PyroModule[nn.Linear](hidden_size1, hidden_size2)
            self.linear2.weight = PyroSample(
                dist.Normal(0., 1.).expand([hidden_size2, hidden_size1]).to_event(2)
            )
            self.linear2.bias = PyroSample(
                dist.Normal(0., 1.).expand([hidden_size2]).to_event(1)
            )

            # Output layer
            self.linear3 = PyroModule[nn.Linear](hidden_size2, 1)
            self.linear3.weight = PyroSample(
                dist.Normal(0., 1.).expand([1, hidden_size2]).to_event(2)
            )
            self.linear3.bias = PyroSample(
                dist.Normal(0., 1.).expand([1]).to_event(1)
            )

            # Learnable observation noise (log_sigma)
            self.log_sigma = PyroSample(dist.Normal(torch.tensor(0.), torch.tensor(0.1)))

        def forward(self, x, y=None):
            # Forward pass
            h1 = torch.relu(self.linear1(x))
            h2 = torch.relu(self.linear2(h1))
            mean = self.linear3(h2).squeeze(-1)  # [batch_size]

            # Compute learnable noise sigma = exp(log_sigma)
            sigma = torch.exp(self.log_sigma)

            with pyro.plate("data", x.shape[0]):
                obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
            return mean

    bnn = BayesianNN(n_features=X_train_torch.shape[1], hidden_size1=50, hidden_size2=50)

    # Use AutoDiagonalNormal as the guide
    guide = AutoDiagonalNormal(bnn)

    # Define optimizer and loss function
    optim = Adam({"lr": 0.005})
    svi = SVI(bnn, guide, optim, loss=Trace_ELBO())

    # ============================
    # Step 9: Train the BNN (SVI)
    # ============================
    pyro.clear_param_store()
    n_steps = 20000  # Adjust according to data size
    for step in range(n_steps):
        loss = svi.step(X_train_torch, y_train_torch)
        if step % 300 == 0:
            print(f"[Iteration {step}] loss={loss:.4f}")

    # ============================
    # Step 10: Prediction (Predictive)
    # ============================
    n_samples = 200
    predictive = Predictive(
        model=bnn,
        guide=guide,
        num_samples=n_samples,
        return_sites=["_RETURN"]
    )

    with torch.no_grad():
        samples = predictive(X_test_torch)
        # samples["_RETURN"] shape: [n_samples, n_test]
        preds = samples["_RETURN"].detach().cpu().numpy()

    y_pred_mean = preds.mean(axis=0)
    y_pred_std = preds.std(axis=0)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))

    # Calculate 95% prediction interval coverage
    lower_bound = y_pred_mean - 1.96 * y_pred_std
    upper_bound = y_pred_mean + 1.96 * y_pred_std
    coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))

    print(f"\nBayesian NN (Deeper+LearnableNoise) Test RMSE: {rmse:.2f}")
    print(f"95% Prediction Interval Coverage: {coverage*100:.2f}%")

    # ============================
    # Step 11: Visualization
    # ============================
    plt.figure(figsize=(10,6))
    plt.errorbar(
        range(len(y_test)),
        y_pred_mean,
        yerr=1.96*y_pred_std,
        fmt='o',
        label='predictions w/ 95% CI',
        alpha=0.7
    )
    plt.scatter(range(len(y_test)), y_test, color='red', label='real value')
    plt.xlabel("test sample index")
    plt.ylabel("total admissions")
    plt.title("Bayesian Neural Network (Deeper + Learnable Noise)")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    main()
