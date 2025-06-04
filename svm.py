import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score,
    log_loss, mean_squared_error
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Dataset/phishingEmail.csv")
df = df.dropna()

print("Dimension of the raw data:", df.shape)

Safe_Email = df[df["Email Type"] == "Safe Email"].sample(df[df["Email Type"] == "Phishing Email"].shape[0], random_state=0)
Phishing_Email = df[df["Email Type"] == "Phishing Email"]
Data = pd.concat([Safe_Email, Phishing_Email], ignore_index=True)

X = Data["Email Text"].astype(str).values
y = Data["Email Type"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train_enc, y_test_enc = train_test_split(X, y_encoded, test_size=0.3, random_state=0, stratify=y_encoded)

svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", SVC(C=100, gamma="auto", probability=True))
])

svm_pipeline.fit(X_train, y_train_enc)

y_pred_enc = svm_pipeline.predict(X_test)

y_proba = svm_pipeline.predict_proba(X_test)

accuracy = accuracy_score(y_test_enc, y_pred_enc)
precision = precision_score(y_test_enc, y_pred_enc)
recall = recall_score(y_test_enc, y_pred_enc)
f1 = f1_score(y_test_enc, y_pred_enc)
logloss = log_loss(y_test_enc, y_proba)
error_rate = 1 - accuracy
rmse = np.sqrt(mean_squared_error(y_test_enc, y_pred_enc))

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"RMSE: {rmse:.4f}")

cm = confusion_matrix(y_test_enc, y_pred_enc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SVM with TF-IDF")
plt.show()
