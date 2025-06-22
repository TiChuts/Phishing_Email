import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
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

lg = LogisticRegression(max_iter=1000)
lg.fit(x_train, y_train_enc)

pred_lg = lg.predict(x_test)
pred_probs = lg.predict_proba(x_test)

accuracy = accuracy_score(y_test_enc, pred_lg)
precision = precision_score(y_test_enc, pred_lg, pos_label=le.transform(['Phishing Email'])[0])
recall = recall_score(y_test_enc, pred_lg, pos_label=le.transform(['Phishing Email'])[0])
f1 = f1_score(y_test_enc, pred_lg, pos_label=le.transform(['Phishing Email'])[0])
logloss = log_loss(y_test_enc, pred_probs)

true_probs = np.array([pred_probs[i, label] for i, label in enumerate(y_test_enc)])
rmse = np.sqrt(np.mean((true_probs - 1.0) ** 2))
error_rate = 1 - accuracy

print(f"Accuracy     : {accuracy * 100:.2f} %")
print(f"Precision    : {precision * 100:.2f} %")
print(f"Recall       : {recall * 100:.2f} %")
print(f"F1 Score     : {f1 * 100:.2f} %")
print(f"Log Loss     : {logloss:.4f}")
print(f"Error Rate   : {error_rate * 100:.2f} %")
print(f"RMSE         : {rmse:.4f}")

cm = confusion_matrix(y_test_enc, pred_lg)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot()
plt.title("Confusion Matrix")
plt.show()
