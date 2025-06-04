import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
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

max_len = 150
tk = Tokenizer()
tk.fit_on_texts(df['Email Text'])
sequences = tk.texts_to_sequences(df['Email Text'])
vector = pad_sequences(sequences, padding='post', maxlen=max_len)

x = np.array(vector)
y = np.array(df['Email Type'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("CNN is processing...")
model = Sequential([
    Embedding(input_dim=len(tk.word_index) + 1, output_dim=50, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 5
batch_size = 16
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

loss, accuracy = model.evaluate(x_test, y_test)

pred_probs = model.predict(x_test)
pred_labels = (pred_probs > 0.5).astype("int32").flatten()

precision = precision_score(y_test, pred_labels)
recall = recall_score(y_test, pred_labels)
f1 = f1_score(y_test, pred_labels)
logloss = log_loss(y_test, pred_probs)
rmse = root_mean_squared_error(y_test, pred_labels)
error_rate = 1 - accuracy

print(f"Accuracy     : {accuracy * 100:.2f} %")
print(f"Precision    : {precision * 100:.2f} %")
print(f"Recall       : {recall * 100:.2f} %")
print(f"F1 Score     : {f1 * 100:.2f} %")
print(f"Log Loss     : {logloss:.4f}")
print(f"Error Rate   : {error_rate * 100:.2f} %")
print(f"RMSE         : {rmse:.4f}")

cm = confusion_matrix(y_test, pred_labels)
ConfusionMatrixDisplay(cm, display_labels=['Phishing Email', 'Safe Email']).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
