import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from preprocess import load_data, preprocess_data
from models import train_logistic_regression, train_random_forest, train_cnn
from visualize import plot_confusion_matrix, plot_roc_curve

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def predict_new_patient(model, scaler, patient_data, feature_names):
    """Predict diagnosis for a new patient and show contributing features."""
    # Convert patient_data to DataFrame to preserve feature names
    patient_data_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_data_scaled = scaler.transform(patient_data_df)
    prob = model.predict_proba(patient_data_scaled)[0, 1] if hasattr(model, 'predict_proba') else model.predict(patient_data_scaled)[0]
    label = "Malignant" if prob > 0.5 else "Benign"
    print(f"Prediction: {label} (Probability of Malignant: {prob:.2%})")
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        print("Top 5 contributing features:\n", importance.head())

def main():
    try:
        # Load and preprocess data
        print("Loading data...")
        X, y = load_data()
        print("Preprocessing data...")
        X_proc, y_proc, scaler = preprocess_data(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_proc, y_proc, test_size=0.2, random_state=42
        )
        
        # Define models
        models = [
            ("Logistic Regression", train_logistic_regression, False),
            ("Random Forest", train_random_forest, False),
            ("CNN", train_cnn, True)
        ]
        
        # Train, evaluate, and save models
        ensure_dir('results/models')
        ensure_dir('results/plots')
        
        for name, train_func, needs_prob in models:
            print(f"\nTraining {name}...")
            # Cross-validation
            if name != "CNN":  # CNN requires custom handling
                model = train_func(X_train, y_train, X_test, y_test)[0]
                scores = cross_val_score(model, X_proc, y_proc, cv=5, scoring='f1')
                print(f"Cross-validation F1 scores: {scores.mean():.4f} ± {scores.std():.4f}")
            
            # Train and evaluate
            if needs_prob:
                model, metrics, y_pred, y_pred_prob = train_func(X_train, y_train, X_test, y_test)
            else:
                model, metrics, y_pred = train_func(X_train, y_train, X_test, y_test)
                y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Print metrics
            print(f"Results for {name}:")
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
            
            # Visualize
            plot_confusion_matrix(y_test, y_pred, name)
            plot_roc_curve(y_test, y_pred_prob, name)
            
            # Save model
            if name == "CNN":
                model.save(f'results/models/{name.lower().replace(" ", "_")}.keras')
            else:
                joblib.dump(model, f'results/models/{name.lower().replace(" ", "_")}.pkl')
            
            # Feature importance for Random Forest
            if name == "Random Forest":
                feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                print("Top 5 features:\n", feature_importance.head())
        
        # Save scaler
        joblib.dump(scaler, 'results/models/scaler.pkl')
        
        # Example prediction for a new patient
        print("\nClinician Decision Support Example (Random Forest):")
        rf_model = joblib.load('results/models/random_forest.pkl')
        predict_new_patient(rf_model, scaler, X_test[0], X.columns)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()