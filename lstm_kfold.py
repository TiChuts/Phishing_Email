import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from Dataset.algorithms import load_and_preprocess_kfold  

# Load preprocessed features and labels using your custom preprocessing function
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

    # Tokenizer and text preprocessing for each fold
    max_len = 150
    tk = Tokenizer()
    tk.fit_on_texts(df['Email Text'])
    sequences = tk.texts_to_sequences(df['Email Text'])
    vector = pad_sequences(sequences, padding='post', maxlen=max_len)
    
    X_train_seq = pad_sequences(tk.texts_to_sequences(X_train), padding='post', maxlen=max_len)
    X_val_seq = pad_sequences(tk.texts_to_sequences(X_val), padding='post', maxlen=max_len)

    # LSTM Model
    model = Sequential()
    model.add(Embedding(input_dim=len(tk.word_index) + 1, output_dim=50, input_length=max_len))
    model.add(LSTM(units=100))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fit model
    model.fit(X_train_seq, y_train, epochs=5, batch_size=16, verbose=0, validation_data=(X_val_seq, y_val))

    # Prediction
    pred_mlp = (model.predict(X_val_seq) > 0.5).astype(int).flatten()

    # Metrics calculation
    acc = accuracy_score(y_val, pred_mlp)
    loss = mean_squared_error(y_val, pred_mlp)
    f1 = f1_score(y_val, pred_mlp)
    rmse = np.sqrt(mean_squared_error(y_val, pred_mlp))

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Loss (MSE): {loss:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")

    accs.append(acc)
    losses.append(loss)
    rmses.append(rmse)

    if fold == k - 1:
        # Confusion matrix on the last fold
        cm = confusion_matrix(y_val, pred_mlp)
        ConfusionMatrixDisplay(cm, display_labels=['Phishing Email', 'Safe Email']).plot()
        plt.title(f"Confusion Matrix (Fold {fold + 1})")
        plt.show()

# Final summary
print("Cross-Validation Summary:")
print(f"Average Accuracy: {np.mean(accs):.4f}")
print(f"Average Loss (MSE): {np.mean(losses):.4f}")
print(f"Average RMSE: {np.mean(rmses):.4f}")
