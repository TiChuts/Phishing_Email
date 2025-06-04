import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, mean_squared_error, f1_score,
    recall_score, precision_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from Dataset.algorithms import load_and_preprocess_kfold

df, X, y = load_and_preprocess_kfold()
X = X.toarray()

k = 5
epochs = 5
batch_size = 32
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accs, losses, rmses, f1s, recalls, precisions = [], [], [], [], [], []

print("Starting K-Fold Cross Validation...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

    print(f"\nFold {fold + 1}/{k}")

    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    y_pred = model.predict(x_val).flatten()
    y_pred_binary = (y_pred >= 0.5).astype(int)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    f1 = f1_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    precision = precision_score(y_val, y_pred_binary)
    error_rate = 1 - acc

    accs.append(acc)
    losses.append(loss)
    rmses.append(rmse)
    f1s.append(f1)
    recalls.append(recall)
    precisions.append(precision)

    print(f"Fold {fold + 1} Accuracy   : {acc:.4f}")
    print(f"Fold {fold + 1} Precision  : {precision:.4f}")
    print(f"Fold {fold + 1} Recall     : {recall:.4f}")
    print(f"Fold {fold + 1} F1 Score   : {f1:.4f}")
    print(f"Fold {fold + 1} Log Loss   : {loss:.4f}")
    print(f"Fold {fold + 1} Error Rate : {error_rate:.4f}")
    print(f"Fold {fold + 1} RMSE       : {rmse:.4f}")

    cm = confusion_matrix(y_val, y_pred_binary)
    ConfusionMatrixDisplay(cm, display_labels=["Phishing", "Safe"]).plot()
    plt.title(f"Fold {fold + 1} Confusion Matrix")
    plt.show()

print("\nCross-Validation Summary:")
print(f"Average Accuracy   : {np.mean(accs)*100:.2f}%")
print(f"Average Precision  : {np.mean(precisions)*100:.2f}%")
print(f"Average Recall     : {np.mean(recalls)*100:.2f}%")
print(f"Average F1 Score   : {np.mean(f1s)*100:.2f}%")
print(f"Average Log Loss   : {(1 - np.mean(loss)):.2f}%")
print(f"Average Error Rate : {(1 - np.mean(accs))*100:.2f}%")
print(f"Average RMSE       : {np.mean(rmses):.4f}")

