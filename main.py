# ==========================================
# Author: Dikshit Sharma
# Project: Student Pass/Fail Prediction
# Using Decision Tree Classifier
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("students.csv")

print("Dataset:\n")
print(df)

# ==========================================
# Features and Target
# ==========================================

X = df[["hours", "sleep"]]

y = df["result"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Create Decision Tree Model
# ==========================================

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

predictions = model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(predictions)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

# ==========================================
# Decision Tree Visualization
# ==========================================

plt.figure(figsize=(10,6))

plot_tree(
    model,
    feature_names=["hours", "sleep"],
    class_names=["Fail", "Pass"],
    filled=True
)

plt.title("Decision Tree")

plt.savefig("tree.png")

plt.show()
