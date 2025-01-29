import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = 'heart-disease.csv'  # Path to the dataset
data = pd.read_csv(data_path)

# Data preprocessing
X = data.drop(columns=['target'])  # Assuming 'target' is the target column
y = data['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump((model, scaler), model_file)

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    loaded_model, loaded_scaler = pickle.load(model_file)

# Create the GUI application
def predict():
    try:
        user_input = {feature: float(entry_fields[feature].get()) for feature in feature_names}
        input_df = pd.DataFrame([user_input])
        input_scaled = loaded_scaler.transform(input_df)

        prediction = loaded_model.predict(input_scaled)
        prediction_proba = loaded_model.predict_proba(input_scaled)

        if prediction[0] == 1:
            result_text = "The model predicts that the patient is at HIGH RISK of heart disease."
        else:
            result_text = "The model predicts that the patient is at LOW RISK of heart disease."

        result_text += f"\n\nPrediction Probability:\nLow Risk: {prediction_proba[0][0]*100:.2f}%\nHigh Risk: {prediction_proba[0][1]*100:.2f}%"

        messagebox.showinfo("Prediction Result", result_text)
    except ValueError as e:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

# Initialize the GUI
root = tk.Tk()
root.title("Heart Disease Risk Prediction")

feature_names = X.columns
entry_fields = {}

# Create input fields
frame = tk.Frame(root)
frame.pack(pady=10)

for feature in feature_names:
    label = tk.Label(frame, text=f"{feature}:")
    label.grid(row=feature_names.get_loc(feature), column=0, padx=5, pady=5, sticky='w')
    entry = tk.Entry(frame)
    entry.grid(row=feature_names.get_loc(feature), column=1, padx=5, pady=5)
    entry.insert(0, f"{X[feature].mean():.2f}")
    entry_fields[feature] = entry

# Add predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)

# Run the GUI
root.mainloop()
