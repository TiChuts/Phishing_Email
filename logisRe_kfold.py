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
from sklearn.model_selection import StratifiedKFold
from Dataset.algorithms import load_and_preprocess_kfold 

df, X, y = load_and_preprocess_kfold()

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accs, precisions, recalls, f1s, log_losses, error_rates, rmses = [], [], [], [], [], [], []

print("Starting K-Fold Cross Validation...\n")  

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{k}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    loss = log_loss(y_val, probs)
    error_rate = 1 - acc
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    accs.append(acc)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    log_losses.append(loss)
    error_rates.append(error_rate)
    rmses.append(rmse)

    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  Log Loss   : {loss:.4f}")
    print(f"  Error Rate : {error_rate:.4f}")
    print(f"  RMSE       : {rmse:.4f}\n")

    cm = confusion_matrix(y_val, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Phishing", "Safe"]).plot()
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.show()
    print("-" * 50)

print("Cross-Validation Summary:")
print(f"Average Accuracy   : {np.mean(accs):.4f}")
print(f"Average Precision  : {np.mean(precisions):.4f}")
print(f"Average Recall     : {np.mean(recalls):.4f}")
print(f"Average F1 Score   : {np.mean(f1s):.4f}")
print(f"Average Log Loss   : {np.mean(log_losses):.4f}")
print(f"Average Error Rate : {np.mean(error_rates):.4f}")
print(f"Average RMSE       : {np.mean(rmses):.4f}")
