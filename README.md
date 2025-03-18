# Hospital-Admissions-Prediction
A Bayesian Neural Network for predicting hospital admissions using climate and demographic data

# Predicting Hospital Admissions Using a Bayesian Neural Network

This project implements a Bayesian Neural Network (BNN) for predicting hospital admissions using climate and demographic data. The model not only provides point predictions but also quantifies uncertainty through predicted confidence intervals.

## Project Description

The project follows these main steps:

1. **Data Reading and Preprocessing:**
   - Raw data is loaded and basic statistics are examined.
   - The `Date` column is converted to datetime format, and the month is extracted.
   - Temperature, Rainfall, and Humidity values are rounded to two decimal places.

2. **Missing Value Handling:**
   - Missing values in climate variables are filled using the monthly average.
   - For Male_Admissions and Female_Admissions, missing values are filled and then rounded up (ceiling) to ensure integer counts.
   - A new column, `Total_Admissions`, is computed as the sum of Male_Admissions and Female_Admissions.

3. **Feature Engineering:**
   - Additional features are extracted from the `Date` column, such as Month, Quarter, and an ordinal date.
   - If available, the `County` feature is one-hot encoded.
   - These features are used as inputs for the model.

4. **Data Splitting and Normalization:**
   - The data is split into training (first 80% of dates) and testing (remaining 20%) sets.
   - Features are normalized using StandardScaler.

5. **Model Building:**
   - A Bayesian Neural Network is built using Pyro, featuring two hidden layers (50 neurons each) and a learnable observation noise parameter.
   
6. **Training and Evaluation:**
   - The model is trained using Stochastic Variational Inference (SVI).
   - Evaluation metrics include RMSE, 95% prediction interval coverage, and prediction accuracy (relative error ≤ 10%).

7. **Visualization:**
   - An error bar plot is generated showing the predicted mean and 95% confidence intervals, compared to the true values.

## Data Availability

The dataset used in this project is not included in this repository. If you need the data, please contact me via email at [baobinzhang1992@gmail.com](mailto:your-email@example.com).

## Installation and Usage

### Requirements

- Python 3.x
- pandas, numpy, matplotlib, torch, pyro-ppl, scikit-learn

Install required packages using:
```bash
pip install -r requirements.txt


