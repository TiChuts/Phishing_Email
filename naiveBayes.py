import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    mean_squared_error,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.naive_bayes import MultinomialNB
from Dataset.algorithms import load_and_preprocess
from sklearn.preprocessing import LabelEncoder

df, x_train, x_test, y_train, y_test = load_and_preprocess()

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

nb = MultinomialNB()
nb.fit(x_train, y_train_encoded)

pred_nav = nb.predict(x_test)
pred_probs = nb.predict_proba(x_test)

accuracy = accuracy_score(y_test_encoded, pred_nav)
precision = precision_score(y_test_encoded, pred_nav, pos_label=le.transform(['Phishing Email'])[0])
recall = recall_score(y_test_encoded, pred_nav, pos_label=le.transform(['Phishing Email'])[0])
f1 = f1_score(y_test_encoded, pred_nav, pos_label=le.transform(['Phishing Email'])[0])
logloss = log_loss(y_test_encoded, pred_probs)

true_probs = np.array([pred_probs[i, label] for i, label in enumerate(y_test_encoded)])
rmse = np.sqrt(np.mean((true_probs - 1.0) ** 2))

error_rate = 1 - accuracy

print(f"Accuracy     : {accuracy * 100:.2f} %")
print(f"Precision    : {precision * 100:.2f} %")
print(f"Recall       : {recall * 100:.2f} %")
print(f"F1 Score     : {f1 * 100:.2f} %")
print(f"Log Loss     : {logloss:.4f}")
print(f"Error Rate   : {error_rate * 100:.2f} %")
print(f"RMSE         : {rmse:.4f}")

clf_nav = confusion_matrix(y_test_encoded, pred_nav)
ConfusionMatrixDisplay(clf_nav, display_labels=le.classes_).plot()
plt.show()
