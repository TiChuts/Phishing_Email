import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.preprocessing import LabelEncoder
from Dataset.algorithms import load_and_preprocess

df, x_train, x_test, y_train, y_test = load_and_preprocess()

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

model = DecisionTreeClassifier()
model.fit(x_train, y_train_enc)

preds = model.predict(x_test)
probs = model.predict_proba(x_test)

acc = accuracy_score(y_test_enc, preds)
precision = precision_score(y_test_enc, preds, pos_label=le.transform(["Phishing Email"])[0])
recall = recall_score(y_test_enc, preds, pos_label=le.transform(["Phishing Email"])[0])
f1 = f1_score(y_test_enc, preds, pos_label=le.transform(["Phishing Email"])[0])
loss = log_loss(y_test_enc, probs)
error_rate = 1 - acc
rmse = np.sqrt(mean_squared_error(y_test_enc, preds))

print(f"Accuracy     : {acc * 100:.2f} %")
print(f"Precision    : {precision * 100:.2f} %")
print(f"Recall       : {recall * 100:.2f} %")
print(f"F1 Score     : {f1 * 100:.2f} %")
print(f"Log Loss     : {loss:.4f}")
print(f"Error Rate   : {error_rate * 100:.2f} %")
print(f"RMSE         : {rmse:.4f}")

cm = confusion_matrix(y_test_enc, preds)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot()
plt.title("Confusion Matrix")
plt.show()
