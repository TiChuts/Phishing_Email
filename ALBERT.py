import tensorflow as tf
from transformers import TFAlbertForSequenceClassification, AlbertTokenizer
from tensorflow.keras.mixed_precision import set_global_policy
from Dataset.algorithms import load_and_preprocess

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

# Load preprocessed dataset
df, x_train, x_test, y_train, y_test = load_and_preprocess()

# Albert
model_name = "albert-base-v2"
tokenizer = AlbertTokenizer.from_pretrained(model_name)

# Tokenization with reduced max_length for speed optimization
train_texts = df["Email Text"].iloc[list(range(len(x_train)))].tolist()
test_texts = df["Email Text"].iloc[list(range(len(x_test)))].tolist()

train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=48, return_tensors="tf")
test_encodings = tokenizer(test_texts, padding="max_length", truncation=True, max_length=48, return_tensors="tf")

# Extract input tensors
train_inputs = train_encodings["input_ids"]
train_masks = train_encodings["attention_mask"]
test_inputs = test_encodings["input_ids"]
test_masks = test_encodings["attention_mask"]

# Create TensorFlow dataset with efficient data pipeline
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": train_inputs, "attention_mask": train_masks}, y_train
)).batch(16).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": test_inputs, "attention_mask": test_masks}, y_test
)).batch(16).prefetch(tf.data.AUTOTUNE)

# Load ALBERT model for binary classification
model = TFAlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Enable mixed precision for faster computation
set_global_policy("mixed_float16")

for group in model.albert.encoder.albert_layer_groups[:2]:  
    for layer in group.albert_layers:
        layer.trainable = False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=3, 
)
