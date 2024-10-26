import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk
from tkinter import ttk

def predict():
    # Get user input from the form
    user_data = [
        float(entrys[label].get()) for label in labels
    ]

    # Standardize user input
    user_input_scaled = scaler.transform(np.array(user_data).reshape(1, -1))

    # Make prediction
    prediction = knn_classifier.predict(user_input_scaled)

    # Display prediction
    result_label.config(text=f"Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")

# Load your dataset (replace 'heart.csv' with your actual dataset)
# Ensure that your dataset includes features and labels, where labels represent the presence of heart disease (1 for yes, 0 for no)
dataset = pd.read_csv('heart.csv')

# Assume 'X' contains features and 'y' contains labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Create GUI form
root = tk.Tk()
root.title("Heart Disease Prediction Form")

# Create form labels and entry widgets
labels = ['Age:', 'Sex:', 'Chest Pain Type:', 'Resting Blood Pressure:', 'Serum Cholesterol:', 
          'Fasting Blood Sugar:', 'Resting ECG:', 'Max Heart Rate Achieved:', 'Exercise-Induced Angina:', 
          'oldpeak:']

entrys = {}

for i, label_text in enumerate(labels):
    ttk.Label(root, text=label_text).grid(row=i, column=0, sticky='e', pady=5)
    entrys[label_text] = ttk.Entry(root, width=10, textvariable=tk.StringVar())
    entrys[label_text].grid(row=i, column=1, pady=5)

# Create prediction button
ttk.Button(root, text="Predict", command=predict).grid(row=len(labels), column=0, columnspan=2, pady=10)

# Display prediction result
result_label = ttk.Label(root, text="")
result_label.grid(row=len(labels) + 1, column=0, columnspan=2, pady=10)

# Start the GUI main loop
root.mainloop()
