import math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

def main():
    # ============================
    # 1. Data Reading and Preprocessing
    # ============================
    df = pd.read_csv("data.csv")

    # Convert 'Date' to datetime and extract the month
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    # Round climate variables (Temperature, Rainfall, Humidity) to two decimal places
    for col in ['Temperature', 'Rainfall', 'Humidity']:
        df[col] = df[col].round(2)

    # Fill missing values in climate variables using the mean within each County and Month group
    for col in ['Temperature', 'Rainfall', 'Humidity']:
        df[col] = df.groupby(['County', 'Month'])[col].transform(lambda x: x.fillna(x.mean()))

    # For admission counts: fill missing values using the group mean within County and Month, then apply ceiling
    for col in ['Male_Admissions', 'Female_Admissions']:
        df[col] = df.groupby(['County', 'Month'])[col].transform(lambda x: np.ceil(x.fillna(x.mean())))
        # If there are still missing values after filling, set them to 0
        df[col] = df[col].fillna(0).astype(int)

    # Calculate Total_Admissions as the sum of Male_Admissions and Female_Admissions
    df['Total_Admissions'] = df['Male_Admissions'] + df['Female_Admissions']

    print("First 5 rows after preprocessing:")
    print(df.head())

    # ============================
    # 2. Prepare Data for Hierarchical Bayesian Model
    # ============================
    # Map County to an index: e.g., C1 -> 0, C2 -> 1, ...
    county_mapping = {c: i for i, c in enumerate(df['County'].unique())}
    df['County_idx'] = df['County'].map(county_mapping)

    # Features: Temperature, Rainfall, Humidity; Target: Total_Admissions
    X = df[['Temperature', 'Rainfall', 'Humidity']].values.astype(np.float32)
    y = df['Total_Admissions'].values.astype(np.float32)
    county_idx = df['County_idx'].values.astype(np.int64)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    county_tensor = torch.tensor(county_idx)

    num_obs = X_tensor.shape[0]
    num_predictors = X_tensor.shape[1]
    num_groups = len(county_mapping)

    # ============================
    # 3. Define Model Parameters (MAP Estimation)
    # ============================
    # Global intercept
    global_intercept = torch.randn(1, requires_grad=True)

    # Beta coefficients for predictors, length = 3 (Temperature, Rainfall, Humidity)
    beta = torch.randn(num_predictors, requires_grad=True)

    # Group intercepts for each county, shape = [num_groups]
    group_intercepts = torch.randn(num_groups, requires_grad=True)

    # Define trainable parameters for σ_α (between-group variance) and σ_y (observation noise)
    # Using softplus to map any real number to a positive value to avoid negative variances
    sigma_alpha_param = torch.randn(1, requires_grad=True)
    sigma_y_param = torch.randn(1, requires_grad=True)

    # ============================
    # 4. Define the Negative Log Posterior (MAP Objective Function)
    # ============================

    def negative_log_posterior():
        # Map sigma_alpha and sigma_y to positive scalars
        sigma_alpha = F.softplus(sigma_alpha_param).squeeze()
        sigma_y = F.softplus(sigma_y_param).squeeze()

        # ------------------------------
        # Likelihood Term
        # ------------------------------
        # Compute mu_i = global_intercept + group_intercepts[county] + β * X
        mu = global_intercept + group_intercepts[county_tensor] + (X_tensor * beta).sum(dim=1)

        # Likelihood: y_i ~ N(mu_i, sigma_y^2)
        # Negative log likelihood:
        # 0.5 * sum((y - mu)^2 / sigma_y^2) + 0.5 * N * log(2π * sigma_y^2)
        nlp_lik = 0.5 * torch.sum((y_tensor - mu)**2 / (sigma_y**2))
        # Convert constant to Tensor to avoid "must be Tensor, not float" error
        # Split into log(2π) + log(sigma_y^2)
        LOG2PI = torch.log(torch.tensor(2.0 * math.pi, dtype=torch.float32))
        nlp_lik += 0.5 * num_obs * (LOG2PI + torch.log(sigma_y**2))

        # ------------------------------
        # Prior Terms
        # ------------------------------

        # Prior 1: global_intercept ~ N(0, 10^2)
        #   => -log p(global_intercept) = 0.5 * (gi^2 / 100) + 0.5 * log(2π * 100)
        nlp_prior_gi = 0.5 * (global_intercept**2) / 100.0
        nlp_prior_gi += 0.5 * (LOG2PI + math.log(100.0))  # math.log(100.0) is a pure float, which is fine

        # Prior 2: beta ~ N(0, 10^2) for each dimension independently
        #   => 0.5 * sum(β_i^2 / 100) + 0.5 * d * log(2π * 100)
        nlp_prior_beta = 0.5 * torch.sum(beta**2) / 100.0
        nlp_prior_beta += 0.5 * num_predictors * (LOG2PI + math.log(100.0))

        # Prior 3: group_intercepts ~ N(0, sigma_alpha^2)
        #   => 0.5 * sum(α_i^2 / sigma_alpha^2) + 0.5 * K * log(2π * sigma_alpha^2)
        nlp_prior_group = 0.5 * torch.sum(group_intercepts**2) / (sigma_alpha**2)
        nlp_prior_group += 0.5 * num_groups * (LOG2PI + torch.log(sigma_alpha**2))

        # Prior 4: sigma_alpha ~ Half-Normal(scale=5)
        #   => p(σ_α) ∝ exp(-σ_α^2 / (2*5^2)), so negative log prior = σ_α^2 / (2 * 25)
        nlp_prior_sigma_alpha = (sigma_alpha**2) / (2.0 * 25.0)

        # Prior 5: sigma_y ~ Half-Normal(scale=5)
        nlp_prior_sigma_y = (sigma_y**2) / (2.0 * 25.0)

        # Total negative log posterior
        nlp = nlp_lik + nlp_prior_gi + nlp_prior_beta + nlp_prior_group + nlp_prior_sigma_alpha + nlp_prior_sigma_y
        return nlp

    # ============================
    # 5. Optimization (MAP Estimation)
    # ============================
    optimizer = optim.Adam(
        [global_intercept, beta, group_intercepts, sigma_alpha_param, sigma_y_param],
        lr=0.05
    )

    num_iters = 3000
    for it in range(num_iters):
        optimizer.zero_grad()
        loss = negative_log_posterior()
        loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print(f"Iteration {it}, Loss: {loss.item():.3f}")

    # ============================
    # 6. Output Results
    # ============================
    sigma_alpha_est = F.softplus(sigma_alpha_param).item()
    sigma_y_est = F.softplus(sigma_y_param).item()

    print("\n=== Model Estimation Results ===")
    print("Global intercept:", global_intercept.item())
    print("Beta coefficients (Temperature, Rainfall, Humidity):", beta.detach().numpy())
    print("Group intercepts (for each county):", group_intercepts.detach().numpy())
    print("Estimated sigma_alpha:", sigma_alpha_est)
    print("Estimated sigma_y:", sigma_y_est)

    # Simple prediction comparison
    mu_pred = (global_intercept + group_intercepts[county_tensor] + (X_tensor * beta).sum(dim=1)).detach().numpy()
    print("\n=== Some Predicted vs. Actual Total_Admissions ===")
    for i in range(min(10, num_obs)):
        print(f"Index {i} | Predicted: {mu_pred[i]:.2f}, Actual: {y_tensor[i].item()}")

if __name__ == "__main__":
    main()
