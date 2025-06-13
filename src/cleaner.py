# File: src/data_cleaning/data_cleaner.py

import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Load dataset with low_memory=False."""
    df = pd.read_csv(filepath, sep='|', low_memory=False)
    print("[OK] Data loaded successfully.")
    return df

def handle_missing_values(df):
    """Fill or drop missing values."""
    print("[INFO] Handling missing values...")

    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    print("[OK] Missing values handled.")
    return df

def fix_negative_values(df):
    """Fix negative TotalPremium and TotalClaims."""
    print("[INFO] Fixing negative premium and claim values...")

    if 'TotalPremium' in df.columns:
        df['TotalPremium'] = df['TotalPremium'].abs()
    if 'TotalClaims' in df.columns:
        df['TotalClaims'] = df['TotalClaims'].abs()

    print("[OK] Negative values fixed.")
    return df

def convert_to_categorical(df):
    """Convert selected columns to category type."""
    print("[INFO] Converting columns to categorical...")

    cols_to_category = ['Gender', 'MaritalStatus', 'Province', 'PostalCode',
                        'CoverType', 'VehicleType', 'make', 'Model']
    for col in cols_to_category:
        if col in df.columns:
            df[col] = df[col].astype('category')

    print("[OK] Columns converted to categorical.")
    return df

def calculate_loss_ratio(df):
    """Calculate Loss Ratio = Claims / Premium."""
    print("[INFO] Calculating loss ratio...")

    df['LossRatio'] = np.where(
        df['TotalPremium'] > 0,
        df['TotalClaims'] / df['TotalPremium'],
        np.nan
    )

    print("[OK] Loss ratio calculated.")
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to CSV."""
    print(f"[INFO] Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("[OK] Cleaned data saved.")

def main():
    # Define paths
    #project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join('C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/Insurance-Risk-Analytics', 'data', 'MachineLearningRating_v3.txt')
    output_path = os.path.join('C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/Insurance-Risk-Analytics', 'data', 'cleaned_insurance_data.csv')

    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    df = load_data(data_path)

    # Clean data
    df = handle_missing_values(df)
    df = fix_negative_values(df)
    df = convert_to_categorical(df)
    df = calculate_loss_ratio(df)

    # Save cleaned data
    save_cleaned_data(df, output_path)

if __name__ == '__main__':
    main()