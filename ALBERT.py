import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
from Dataset.algorithms import load_and_preprocess

df, x_train, x_test, y_train, y_test = load_and_preprocess()

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

train_texts = df["Email Text"].iloc[list(range(len(x_train)))].tolist()
test_texts = df["Email Text"].iloc[list(range(len(x_test)))].tolist()

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=64, return_tensors="tf")
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=64, return_tensors="tf")

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
    epochs=5, batch_size=8
)


loss, accuracy = model.evaluate([test_inputs, test_masks], y_test)
print(f"Test Accuracy: {accuracy:.2f}")