import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    zero_one_loss,
    mean_squared_error
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from Dataset.algorithms import load_and_preprocess_kfold  

# Load and clean data using your custom preprocessing function
df, X, y = load_and_preprocess_kfold()  

# K-Fold setup
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accs = []
losses = []
rmses = []

print("Starting K-Fold Cross Validation...\n")  

# K-Fold training
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/{k}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # SVM with TfidfVectorizer
    SVM = Pipeline([("tfidf", TfidfVectorizer()), ("SVM", SVC(C=100, gamma="auto"))])
    SVM.fit(X_train, y_train)

    # Make predictions
    s_ypred = SVM.predict(X_val)

    # Metrics calculation
    acc = accuracy_score(y_val, s_ypred)
    loss = zero_one_loss(y_val, s_ypred)
    f1 = f1_score(y_val, s_ypred)
    rmse = np.sqrt(mean_squared_error(y_val, s_ypred))

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Loss: {loss:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")

    accs.append(acc)
    losses.append(loss)
    rmses.append(rmse)

    if fold == k - 1:
        # Confusion matrix on the last fold
        cm = confusion_matrix(y_val, s_ypred)
        ConfusionMatrixDisplay(cm, display_labels=['Phishing Email', 'Safe Email']).plot()
        plt.title(f"Confusion Matrix (Fold {fold + 1})")
        plt.show()

# Final summary
print("Cross-Validation Summary:")
print(f"Average Accuracy: {np.mean(accs):.4f}")
print(f"Average Loss: {np.mean(losses):.4f}")
print(f"Average RMSE: {np.mean(rmses):.4f}")
