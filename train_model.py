import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------- LOAD DATA ----------------
data = pd.read_csv("data.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# ---------------- SPLIT FIRST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# ---------------- AUGMENT TRAIN ONLY ----------------
def augment(X):
    noise = np.random.normal(0, 0.01, X.shape)
    return X + noise

X_train_aug = augment(X_train)

# Combine only training data
X_train = np.vstack((X_train, X_train_aug))
y_train = np.hstack((y_train, y_train))

print("Training size after augmentation:", X_train.shape)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------- SAVE ----------------
joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")