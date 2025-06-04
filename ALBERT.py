import tensorflow as tf
from transformers import TFAlbertForSequenceClassification, AlbertTokenizer
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, root_mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay
)
from Dataset.algorithms import load_and_preprocess
import numpy as np
import matplotlib.pyplot as plt

set_global_policy("mixed_float16")

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

df, x_train, x_test, y_train, y_test = load_and_preprocess()

model_name = "albert-base-v2"
tokenizer = AlbertTokenizer.from_pretrained(model_name)

train_texts = df["Email Text"].iloc[list(range(len(x_train)))].tolist()
test_texts = df["Email Text"].iloc[list(range(len(x_test)))].tolist()

train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=64, return_tensors="tf")
test_encodings = tokenizer(test_texts, padding="max_length", truncation=True, max_length=64, return_tensors="tf")

train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"]},
    y_train
)).batch(16).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"]},
    y_test
)).batch(16).prefetch(tf.data.AUTOTUNE)

model = TFAlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)

for group in model.albert.encoder.albert_layer_groups[:2]:
    for layer in group.albert_layers:
        layer.trainable = False

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5
)

pred_logits = model.predict(test_dataset).logits
pred_probs = tf.nn.softmax(pred_logits, axis=1).numpy()
pred_labels = np.argmax(pred_probs, axis=1)

accuracy = accuracy_score(y_test, pred_labels)
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
