import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Load and clean data
df = pd.read_csv("Dataset/phishingEmail.csv")
df = df.dropna()  # Drop missing values

print("Dimension of the raw data:", df.shape)
email_type_counts = df['Email Type'].value_counts()

# Create the bar chart with custom colors
unique_email_types = email_type_counts.index.tolist()
color_map = {'Phishing Email': 'red', 'Safe Email': 'green'}
colors = [color_map.get(email_type, 'gray') for email_type in unique_email_types]

plt.figure(figsize=(8, 6))
plt.bar(unique_email_types, email_type_counts, color=colors)
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types with Custom Colors')
plt.xticks(rotation=45)
plt.tight_layout()

# Balance the dataset
Safe_Email = df[df["Email Type"] == "Safe Email"]
Phishing_Email = df[df["Email Type"] == "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])  # Balance dataset
Data = pd.concat([Safe_Email, Phishing_Email], ignore_index=True)

# Prepare features and labels
X = Data["Email Text"].values
y = Data["Email Type"].values

# Ensure X is fully flattened and string-converted
X = [str(x) for x in X]  # Convert all entries to string explicitly

# Split data into training and testing sets
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# SVM with TfidfVectorizer
SVM = Pipeline([("tfidf", TfidfVectorizer()), ("SVM", SVC(C=100, gamma="auto"))])
SVM.fit(X_train, y_train)

# Make predictions
s_ypred = SVM.predict(x_test)

# Evaluate performance
print("Accuracy Score Report: \n", accuracy_score(y_test, s_ypred))
print("Classification Report:\n", classification_report(y_test, s_ypred))
print("Confusion Matrix:\n", confusion_matrix(y_test, s_ypred))
