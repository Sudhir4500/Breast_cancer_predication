import pandas as pd
import numpy as np
import joblib

def predict_new_patient(model, scaler, patient_data, feature_names, true_label, model_name="Model"):
    """Predict diagnosis for a new patient, compare with true label, and show contributing features."""
    # Convert patient_data to DataFrame to preserve feature names
    patient_data_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_data_scaled = scaler.transform(patient_data_df)
    prob = model.predict_proba(patient_data_scaled)[0, 1] if hasattr(model, 'predict_proba') else model.predict(patient_data_scaled)[0]
    predicted_label = "M" if prob > 0.5 else "B"
    correct = predicted_label == true_label
    print(f"\n{model_name} Prediction:")
    print(f"Predicted Diagnosis: {predicted_label} (Probability of Malignant: {prob:.2%})")
    print(f"True Diagnosis: {true_label}")
    print(f"Prediction Correct: {'Yes' if correct else 'No'}")
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        print("Top 5 contributing features:\n", importance.head())

def main():
    try:
        # Define feature names (30 features, excluding id and diagnosis)
        feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
            'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
            'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
            'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
            'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
            'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        # Input data (your provided row, excluding id and diagnosis)
        patient_data = [
            13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259
        ]
        
        # True diagnosis from input
        true_label = 'B'
        
        # Load scaler and models
        scaler = joblib.load('results/models/scaler.pkl')
        rf_model = joblib.load('results/models/random_forest.pkl')
        lr_model = joblib.load('results/models/logistic_regression.pkl')
        
        # Predict with random forest
        predict_new_patient(rf_model, scaler, patient_data, feature_names, true_label, "Random Forest")
        
        # Predict with logistic regression
        predict_new_patient(lr_model, scaler, patient_data, feature_names, true_label, "Logistic Regression")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()