import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(r"C:\Users\amit\OneDrive\Desktop\Suraj Official\Senior DataAnalytics Engineer Internship\data\raw_telco_churn.csv")

# Clean numeric data & missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

# Handle duplicates and uniform categories
df.drop_duplicates(inplace=True)
service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})

# Outlier handling (IQR)
for col in ['MonthlyCharges', 'TotalCharges']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.where(df[col] > Q3 + 1.5 * IQR, Q3 + 1.5 * IQR, df[col])
    df[col] = np.where(df[col] < Q1 - 1.5 * IQR, Q1 - 1.5 * IQR, df[col])

# Feature engineering
bins = [-1, 12, 24, 48, 60, 100]
labels = ['0-1 Year', '1-2 Years', '2-4 Years', '4-5 Years', '5+ Years']
df['TenureCohort'] = pd.cut(df['tenure'], bins=bins, labels=labels)

# Export cleaned data
df.to_csv("../data/cleaned_telco_churn.csv", index=False)
print("Processing complete. Cleaned file saved to data/cleaned_telco_churn.csv")