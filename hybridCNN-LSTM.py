import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
from Dataset.algorithms import load_and_preprocess 

# Load processed data
df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()
      
max_len = 150
tk = Tokenizer()

model = Sequential([
    Embedding(input_dim=len(tk.word_index)+1,output_dim=50,input_length=max_len),
    
    # CNN 
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    # LSTM 
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(x_test, y_test)
)

y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype(int) 

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

