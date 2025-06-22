import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding,SimpleRNN
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

from Dataset.algorithms import load_and_preprocess 

import matplotlib.pyplot as plt

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

# Simple RNN
model_smp = Sequential() # Sequential() API
model_smp.add(Embedding(input_dim=len(tk.word_index)+1,output_dim=50,input_length=150))
model_smp.add(SimpleRNN(units=100))
model_smp.add(Dropout(0.45))
model_smp .add(Dense(1, activation='sigmoid'))
model_smp.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
model_smp.summary()

historical_smp  = model_smp.fit(x_train,y_train, epochs=5, batch_size=16, validation_data=(x_test,y_test))
pd.DataFrame(historical_smp.history)
pd.DataFrame(historical_smp.history)[['accuracy', 'val_accuracy']].plot()
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')

pd.DataFrame(historical_smp.history)[['loss', 'val_loss']].plot()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
y_pred_prob_smp = model_smp.predict(x_test)
y_pred_smp = (y_pred_prob_smp > 0.5).astype(int)
cnf_smp = confusion_matrix(y_test,y_pred_smp)
ax_smp = ConfusionMatrixDisplay(confusion_matrix=cnf_smp,display_labels=['phishing','normal']).plot()
plt.title("Confusion Matrix")
plt.show()