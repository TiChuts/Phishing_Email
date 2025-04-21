import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from Dataset.algorithms import load_and_preprocess_kfold  

# Load processed data
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

    # Tokenization and padding
    max_len = 150
    tk = Tokenizer()
    tk.fit_on_texts(X_train)
    X_train_seq = tk.texts_to_sequences(X_train)
    X_val_seq = tk.texts_to_sequences(X_val)
    
    X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_val_padded = tf.keras.preprocessing.sequence.pad_sequences(X_val_seq, maxlen=max_len, padding='post')

    # Define the model
    model = Sequential([
        Embedding(input_dim=len(tk.word_index)+1, output_dim=50, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_padded, y_train, epochs=5, batch_size=16, validation_data=(X_val_padded, y_val), verbose=0)

    # Make predictions
    y_pred_probs = model.predict(X_val_padded)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    loss = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Loss (MSE): {loss:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")

    accs.append(acc)
    losses.append(loss)
    rmses.append(rmse)

    if fold == k - 1:
        # Confusion matrix on the last fold
        cm = confusion_matrix(y_val, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=['Phishing Email', 'Safe Email']).plot()
        plt.title(f"Confusion Matrix (Fold {fold + 1})")
        plt.show()

# Final summary
print("Cross-Validation Summary:")
print(f"Average Accuracy: {np.mean(accs):.4f}")
print(f"Average Loss (MSE): {np.mean(losses):.4f}")
print(f"Average RMSE: {np.mean(rmses):.4f}")
