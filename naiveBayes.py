import matplotlib.pyplot as plt

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
import numpy as np

# Load processed data
df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

# Naive Bayes
nb = MultinomialNB()
nb.fit(x_train, y_train)

# Prediction
pred_nav = nb.predict(x_test)
pred_probs = nb.predict_proba(x_test)

# Metrics
accuracy = accuracy_score(y_test, pred_nav)
precision = precision_score(y_test, pred_nav)
recall = recall_score(y_test, pred_nav)
f1 = f1_score(y_test, pred_nav)
logloss = log_loss(y_test, pred_probs)
rmse = mean_squared_error(y_test, pred_nav, squared=False)
error_rate = 1 - accuracy

# Print metrics
print(f"Accuracy     : {accuracy * 100:.2f} %")
print(f"Precision    : {precision * 100:.2f} %")
print(f"Recall       : {recall * 100:.2f} %")
print(f"F1 Score     : {f1 * 100:.2f} %")
print(f"Log Loss     : {logloss:.4f}")
print(f"Error Rate   : {error_rate * 100:.2f} %")
print(f"RMSE         : {rmse:.4f}")

# Confusion Matrix
clf_nav = confusion_matrix(y_test, pred_nav)
ConfusionMatrixDisplay(clf_nav, display_labels=['Phishing Email', 'Safe Email']).plot()
plt.show()
