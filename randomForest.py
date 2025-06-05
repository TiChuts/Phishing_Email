import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from Dataset.algorithms import load_and_preprocess

df, X_train, X_test, y_train_raw, y_test_raw = load_and_preprocess()

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
loss = log_loss(y_test, probs)
rmse = np.sqrt(mean_squared_error(y_test, preds))
error_rate = 1 - acc

print(f"Accuracy     : {acc * 100:.2f} %")
print(f"Precision    : {precision * 100:.2f} %")
print(f"Recall       : {recall * 100:.2f} %")
print(f"F1 Score     : {f1 * 100:.2f} %")
print(f"Log Loss     : {loss:.4f}")
print(f"Error Rate   : {error_rate * 100:.2f} %")
print(f"RMSE         : {rmse:.4f}")

cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot()
plt.title("Confusion Matrix")
plt.show()
