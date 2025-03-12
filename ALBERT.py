import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
from Dataset.algorithms import load_and_preprocess

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

# ALBERT

# Load preprocessed dataset
df, x_train, x_test, y_train, y_test = load_and_preprocess()

# Load ALBERT tokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# Convert NumPy arrays back to lists before indexing
train_texts = df["Email Text"].iloc[list(range(len(x_train)))].tolist()
test_texts = df["Email Text"].iloc[list(range(len(x_test)))].tolist()

# Tokenize email text
train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512, return_tensors="tf")
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="tf")

# Extract input tensors
train_inputs = train_encodings["input_ids"]
train_masks = train_encodings["attention_mask"]
test_inputs = test_encodings["input_ids"]
test_masks = test_encodings["attention_mask"]

model = TFAlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


model.fit(
    [train_inputs, train_masks], y_train,
    validation_data=([test_inputs, test_masks], y_test),
    epochs=3, batch_size=8
)


loss, accuracy = model.evaluate([test_inputs, test_masks], y_test)
print(f"Test Accuracy: {accuracy:.2f}")

def predict_email(email_text):
    email_text = email_text.lower().strip()
    
    tokens = tokenizer([email_text], padding=True, truncation=True, max_length=512, return_tensors="tf")
    
    logits = model.predict([tokens["input_ids"], tokens["attention_mask"]]).logits
    pred_label = tf.argmax(logits, axis=1).numpy()[0]

    return "Phishing" if pred_label == 1 else "Legitimate"
