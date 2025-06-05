import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, confusion_matrix,
    ConfusionMatrixDisplay, mean_squared_error
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import pandas as pd

df = pd.read_csv("Dataset/phishingEmail.csv").dropna()
phishing = df[df['Email Type'] == 'Phishing Email']
safe = df[df['Email Type'] == 'Safe Email'].sample(len(phishing), random_state=0)
df = pd.concat([phishing, safe]).sample(frac=1, random_state=0).reset_index(drop=True)

texts = df['Email Text'].astype(str).values
labels = (df['Email Type'] == 'Phishing Email').astype("float32").values

max_len = 150
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, padding='post', maxlen=max_len)
y = np.array(labels)

k = 5
epochs = 5
batch_size = 16
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accs, f1s, recalls, precisions, log_losses, error_rates, rmses = [], [], [], [], [], [], []

print("Starting CNN K-Fold Cross Validation...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}/{k}")

    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=1)

    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    preds = (model.predict(x_val).flatten() > 0.5).astype(int)

    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    logloss = log_loss(y_val, preds)
    error_rate = 1 - accuracy
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    accs.append(accuracy)
    f1s.append(f1)
    recalls.append(recall)
    precisions.append(precision)
    log_losses.append(logloss)
    error_rates.append(error_rate)
    rmses.append(rmse)

    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  Log Loss   : {logloss:.4f}")
    print(f"  Error Rate : {error_rate:.4f}")
    print(f"  RMSE       : {rmse:.4f}")

    cm = confusion_matrix(y_val, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Phishing", "Safe"]).plot()
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.show()

print("\nCross-Validation Summary:")
print(f"Average Accuracy   : {np.mean(accs)*100:.2f}%")
print(f"Average Precision  : {np.mean(precisions)*100:.2f}%")
print(f"Average Recall     : {np.mean(recalls)*100:.2f}%")
print(f"Average F1 Score   : {np.mean(f1s)*100:.2f}%")
print(f"Average Log Loss   : {np.mean(log_losses):.4f}")
print(f"Average Error Rate : {np.mean(error_rates)*100:.2f}%")
print(f"Average RMSE       : {np.mean(rmses):.4f}")
