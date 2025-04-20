import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load dataset
data = pd.read_csv("Medpredict_Dataset_Cleaned.csv")

# Select relevant features and target variables
features = ['Age', 'Gender', 'BMI', 'Hemoglobin', 'CholCheck', 'Smoker', 'PhysActivity', 'Diabetes', 'HighBP', 'Fatty Liver']
targets = ['HeartDiseaseorAttack', 'Stroke', 'Diabetes', 'HighBP', 'Fatty Liver', 'Anemia']

X = data[features]
y = data[targets]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# GUI for prediction
def predict_disease():
    try:
        input_data = [
            float(age_entry.get()), int(gender_var.get()), float(bmi_entry.get()),
            float(hemoglobin_entry.get()), int(chol_check_var.get()), int(smoker_var.get()),
            int(phys_activity_var.get()), int(diabetes_var.get()), int(highbp_var.get()),
            int(fatty_liver_var.get())
        ]
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        diseases = targets
        result = "\n".join([f"{diseases[i]}: {'Yes' if prediction[0][i] else 'No'}" for i in range(len(diseases))])
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", f"Invalid Input: {e}")

# Create Tkinter GUI
root = tk.Tk()
root.title("MedPredict - Disease Prediction")
root.geometry("400x500")

tk.Label(root, text="Enter Patient Details", font=("Arial", 14)).pack()

tk.Label(root, text="Age:").pack()
age_entry = tk.Entry(root)
age_entry.pack()

tk.Label(root, text="Gender (1=Male, 0=Female):").pack()
gender_var = tk.StringVar(value="1")
tk.Entry(root, textvariable=gender_var).pack()

tk.Label(root, text="BMI:").pack()
bmi_entry = tk.Entry(root)
bmi_entry.pack()

tk.Label(root, text="Hemoglobin:").pack()
hemoglobin_entry = tk.Entry(root)
hemoglobin_entry.pack()

tk.Label(root, text="Cholesterol Check (1=Yes, 0=No):").pack()
chol_check_var = tk.StringVar(value="1")
tk.Entry(root, textvariable=chol_check_var).pack()

tk.Label(root, text="Smoker (1=Yes, 0=No):").pack()
smoker_var = tk.StringVar(value="0")
tk.Entry(root, textvariable=smoker_var).pack()

tk.Label(root, text="Physical Activity (1=Yes, 0=No):").pack()
phys_activity_var = tk.StringVar(value="1")
tk.Entry(root, textvariable=phys_activity_var).pack()

tk.Label(root, text="Diabetes (1=Yes, 0=No):").pack()
diabetes_var = tk.StringVar(value="0")
tk.Entry(root, textvariable=diabetes_var).pack()

tk.Label(root, text="High Blood Pressure (1=Yes, 0=No):").pack()
highbp_var = tk.StringVar(value="0")
tk.Entry(root, textvariable=highbp_var).pack()

tk.Label(root, text="Fatty Liver (1=Yes, 0=No):").pack()
fatty_liver_var = tk.StringVar(value="0")
tk.Entry(root, textvariable=fatty_liver_var).pack()

tk.Button(root, text="Predict Disease", command=predict_disease).pack()

root.mainloop()
