import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from Dataset.algorithms import load_and_preprocess

df, x_train, x_test, y_train, y_test = load_and_preprocess()

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(x_train, y_train_encoded)

pred_xgb = xgb.predict(x_test)
pred_proba = xgb.predict_proba(x_test)

accuracy = accuracy_score(y_test_encoded, pred_xgb)
precision = precision_score(y_test_encoded, pred_xgb)
recall = recall_score(y_test_encoded, pred_xgb)
f1 = f1_score(y_test_encoded, pred_xgb)
logloss = log_loss(y_test_encoded, pred_proba)
error_rate = 1 - accuracy
rmse = np.sqrt(mean_squared_error(y_test_encoded, pred_xgb))

print(f"Accuracy     : {accuracy * 100:.2f} %")
print(f"Precision    : {precision * 100:.2f} %")
print(f"Recall       : {recall * 100:.2f} %")
print(f"F1 Score     : {f1 * 100:.2f} %")
print(f"Log Loss     : {logloss:.4f}")
print(f"Error Rate   : {error_rate * 100:.2f} %")
print(f"RMSE         : {rmse:.4f}")

cm = confusion_matrix(y_test_encoded, pred_xgb)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot()
plt.title("Confusion Matrix")
plt.show()
