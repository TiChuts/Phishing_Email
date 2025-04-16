import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    mean_squared_error
)
from Dataset.algorithms import load_and_preprocess_kfold

df, X, y = load_and_preprocess_kfold()

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accs, f1s, losses, rmses = [], [], [], []

print("Starting 10-Fold Cross Validation...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    loss = log_loss(y_val, probs)
    loss_rate = 1 - acc

    true_probs = np.array([probs[i, label] for i, label in enumerate(y_val)])
    rmse = np.sqrt(mean_squared_error(np.ones_like(true_probs), true_probs))

    accs.append(acc)
    f1s.append(f1)
    losses.append(loss)
    rmses.append(rmse)

    print(f"Fold {fold}")
    print(f"Accuracy     : {acc*100:.2f}%")
    print(f"F1 Score     : {f1*100:.2f}%")
    print(f"Log Loss     : {loss:.4f}")
    print(f"Loss Rate    : {loss_rate*100:.2f}%")
    print(f"RMSE         : {rmse:.4f}")
    
    cm = confusion_matrix(y_val, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Phishing", "Safe"]).plot()
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.show()
    print("-" * 50)

print("\nCross-Validation Summary:")
print(f"Average Accuracy : {np.mean(accs)*100:.2f}%")
print(f"Average F1 Score : {np.mean(f1s)*100:.2f}%")
print(f"Average Log Loss : {np.mean(losses):.4f}")
print(f"Average RMSE     : {np.mean(rmses):.4f}")
print(f"Average Loss Rate: {(1 - np.mean(accs))*100:.2f}%")
