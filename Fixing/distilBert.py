import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)

df = pd.read_csv("Dataset/phishingEmail.csv")
df.dropna(inplace=True)
df["label"] = df["Email Type"].apply(lambda x: 1 if x == "Phishing Email" else 0)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Email Text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=64)
train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()
trainer.evaluate()

predictions = trainer.predict(test_dataset)

preds = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()

true_labels = test_labels.numpy()  

acc = accuracy_score(true_labels, preds)
print(f"Test Accuracy: {acc:.4f}")