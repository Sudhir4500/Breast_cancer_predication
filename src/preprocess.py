import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(file_path="data/data.csv"):
    """Load the breast cancer dataset from a CSV file."""
    df = pd.read_csv(file_path)
    # Drop irrelevant columns: 'id' and 'Unnamed: 32' (if present)
    columns_to_drop = ['id', 'Unnamed: 32'] if 'Unnamed: 32' in df.columns else ['id']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    # Convert diagnosis (M/B) to binary (1/0)
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
    else:
        raise ValueError("Expected 'diagnosis' column in dataset")
    # Ensure y is numeric
    y = y.astype(int)
    return X, y

def preprocess_data(X, y):
    """Preprocess data: handle missing values, scale features, balance classes."""
    # Ensure X is numeric
    non_numeric_cols = X.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"Non-numeric columns found: {non_numeric_cols}")
    
    # Handle missing values
    print("Missing values before imputation:\n", X.isnull().sum())
    if X.isnull().sum().sum() > 0:
        # Impute with mean for numeric columns
        X = X.fillna(X.mean(numeric_only=True))
        # Check if any NaN remain
        if X.isnull().sum().sum() > 0:
            raise ValueError("NaN values persist after imputation. Check dataset.")
    print("Missing values after imputation:\n", X.isnull().sum())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Verify no NaN in scaled data
    if np.isnan(X_scaled).any():
        raise ValueError("NaN values found in scaled data.")
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    
    return X_balanced, y_balanced, scaler

# Example usage
if __name__ == "__main__":
    X, y = load_data()
    X_proc, y_proc, scaler = preprocess_data(X, y)
    print(f"Processed data shape: {X_proc.shape}, Classes: {np.unique(y_proc)}")