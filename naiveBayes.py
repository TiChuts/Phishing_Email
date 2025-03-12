from sklearn.metrics import accuracy_score,f1_score,classification_report,ConfusionMatrixDisplay,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from Dataset.algorithms import load_and_preprocess 

import matplotlib.pyplot as plt

# Load processed data
df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

# Naive Bayes
nb = MultinomialNB()
nb.fit(x_train,y_train)
pred_nav = nb.predict(x_test)

# Checking the performance

print(f"Accuracy from Naive Bayes: {accuracy_score(y_test,pred_nav)*100:.2f} %")
print(f"F1 Score from Naive Bayes: {f1_score(y_test,pred_nav)*100:.2f} %")
print("Classification Report :\n\n",classification_report(y_test,pred_nav))

# Confusion matrix
clf_nav = confusion_matrix(y_test,pred_nav)
cx_ = ConfusionMatrixDisplay(clf_nav,display_labels=['phishing_mail','safe_mail']).plot()
plt.show()