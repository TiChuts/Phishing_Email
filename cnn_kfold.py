import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from Dataset.algorithms import load_and_preprocess_kfold

# Load preprocessed TF-IDF features and labels
df, X, y = load_and_preprocess_kfold()
X = X.toarray()

# K-Fold setup
k = 5
epochs = 5
batch_size = 32
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accs = []
losses = []
rmses = []

# K-Fold training
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{k}")

    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Define model
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

    # Evaluate model
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    y_pred = model.predict(x_val).flatten()
    y_pred_binary = (y_pred >= 0.5).astype(int)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    print(f"Fold {fold + 1} Loss: {loss:.4f}")
    print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

    accs.append(acc)
    losses.append(loss)
    rmses.append(rmse)

# Final Results
print("\nCross-Validation Summary:")
print(f"Average Accuracy : {np.mean(accs)*100:.2f}%")
print(f"Average RMSE     : {np.mean(rmses):.4f}")
print(f"Average Loss Rate: {(1 - np.mean(accs))*100:.2f}%")
