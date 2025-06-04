import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, log_loss, mean_squared_error, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from Dataset.algorithms import load_and_preprocess
import numpy as np

df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(x_train, y_train)

pred_xgb = xgb.predict(x_test)
pred_proba = xgb.predict_proba(x_test)

accuracy = accuracy_score(y_test, pred_xgb)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred_xgb, average='weighted')
logloss = log_loss(y_test, pred_proba)
error_rate = 1 - accuracy
rmse = np.sqrt(mean_squared_error(y_test, pred_xgb))

print(f"Accuracy from XGB: {accuracy * 100:.2f} %")
print(f"Precision from XGB: {precision * 100:.2f} %")
print(f"Recall from XGB: {recall * 100:.2f} %")
print(f"F1 Score from XGB: {f1 * 100:.2f} %")
print(f"Log Loss from XGB: {logloss:.4f}")
print(f"Error Rate from XGB: {error_rate:.4f}")
print(f"RMSE from XGB: {rmse:.4f}")
print("Classification Report : \n", classification_report(y_test, pred_xgb))

clf_xgb = confusion_matrix(y_test, pred_xgb)
ConfusionMatrixDisplay(confusion_matrix=clf_xgb, display_labels=['Phishing Email', 'Safe Email']).plot()
plt.show()
