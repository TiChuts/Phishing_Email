import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Embedding,GRU,LSTM,Bidirectional,SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,SimpleRNN
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from Dataset.algorithms import load_and_preprocess 

# Load processed data
df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

#Neural Network
max_len = 150
tk = Tokenizer()

# Process the text
tk.fit_on_texts(df['Email Text'])
sequences = tk.texts_to_sequences(df['Email Text'])
vector = pad_sequences(sequences,padding='post', maxlen=max_len)
len(vector)
x = np.array(vector)
y = np.array(df["Email Type"])

#Split the dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(vector,df['Email Type'], test_size=0.2, random_state =0)

# LSTM
print("LSTM is processing...")
model = Sequential() 
model.add(Embedding(input_dim=len(tk.word_index)+1,output_dim=50,input_length=max_len))
model.add(LSTM(units=100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
model.summary()
historical  = model.fit(x_train,y_train, epochs=5, batch_size=16, validation_data=(x_test,y_test))