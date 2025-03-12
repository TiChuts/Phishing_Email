import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_csv("Dataset/phishingEmail.csv")
df.isna().sum()
df = df.dropna()

print("Dimension of the row data:",df.shape)   
email_type_counts = df['Email Type'].value_counts()


# Create the bar charts
unique_email_types = email_type_counts.index.tolist()
 
color_map = {
    'Phishing Email': 'red',
    'Safe Email': 'green',}
colors = [color_map.get(email_type, 'gray') for email_type in unique_email_types]

# Create the bar chart with custom colors
plt.figure(figsize=(8, 6))
plt.bar(unique_email_types, email_type_counts, color=colors)
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types with Custom Colors')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

Safe_Email = df[df["Email Type"]== "Safe Email"]
Phishing_Email = df[df["Email Type"]== "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])
Safe_Email.shape,Phishing_Email.shape

Data= pd.concat([Safe_Email, Phishing_Email], ignore_index = True)

X = Data["Email Text"].values
y = Data["Email Type"].values

# SVM
X_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
SVM = Pipeline([("tfidf", TfidfVectorizer()),("SVM", SVC(C = 100, gamma = "auto"))])
SVM.fit(X_train,y_train)
s_ypred = SVM.predict(x_test)

print("Accuracy Score Report : \n",accuracy_score(y_test,s_ypred ))