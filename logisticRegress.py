import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, root_mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay
)
from Dataset.algorithms import load_and_preprocess 

df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

lg = LogisticRegression()
lg.fit(x_train, y_train)

pred_lg = lg.predict(x_test)
pred_probs = lg.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, pred_lg)
precision = precision_score(y_test, pred_lg)
recall = recall_score(y_test, pred_lg)
f1 = f1_score(y_test, pred_lg)
logloss = log_loss(y_test, pred_probs)
rmse = root_mean_squared_error(y_test, pred_lg)
error_rate = 1 - accuracy

print(f"Accuracy     : {accuracy * 100:.2f} %")
print(f"Precision    : {precision * 100:.2f} %")
print(f"Recall       : {recall * 100:.2f} %")
print(f"F1 Score     : {f1 * 100:.2f} %")
print(f"Log Loss     : {logloss:.4f}")
print(f"Error Rate   : {error_rate * 100:.2f} %")
print(f"RMSE         : {rmse:.4f}")

cm = confusion_matrix(y_test, pred_lg)
ConfusionMatrixDisplay(cm, display_labels=['Phishing Email', 'Safe Email']).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
