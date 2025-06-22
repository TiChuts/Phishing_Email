import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error
)

df = pd.read_csv("Dataset/phishingEmail.csv").dropna()

safe = df[df["Email Type"] == "Safe Email"]
phish = df[df["Email Type"] == "Phishing Email"]
safe_sample = safe.sample(phish.shape[0], random_state=42)
data = pd.concat([phish, safe_sample], ignore_index=True)

X = data["Email Text"].astype(str).values
y = data["Email Type"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

svm_pipeline = Pipeline([
    ("svm", SVC(C=100, gamma="auto", probability=True))
])

svm_pipeline.fit(X_train, y_train)

preds = svm_pipeline.predict(X_test)
probs = svm_pipeline.predict_proba(X_test)

acc = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
loss = log_loss(y_test, probs)
error_rate = 1 - acc
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\nEvaluation Results:")
print(f"  Accuracy   : {acc:.4f}")
print(f"  Precision  : {precision:.4f}")
print(f"  Recall     : {recall:.4f}")
print(f"  F1 Score   : {f1:.4f}")
print(f"  Log Loss   : {loss:.4f}")
print(f"  Error Rate : {error_rate:.4f}")
print(f"  RMSE       : {rmse:.4f}")

cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot()
plt.title("Confusion Matrix")
plt.show()
