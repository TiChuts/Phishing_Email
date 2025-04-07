import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from Dataset.algorithms import load_and_preprocess 

# Load processed data
df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()


# Logistic Regression
lg = LogisticRegression()
lg.fit(x_train,y_train)

# Prediction
pred_lg = lg.predict(x_test)
# Performance
print("")
print(f"Accuracy from logistic regression:{accuracy_score(y_test,pred_lg)*100:.2f} %")
print(f"F1 score from logistic regression: {f1_score(y_test,pred_lg)*100:.2f} %")
print("Classification report : \n",classification_report(y_test,pred_lg))

# Confusion Matrix
clf_lg = confusion_matrix(y_test,pred_lg)
cx_ = ConfusionMatrixDisplay(clf_lg,display_labels=['Phishing Email','Safe Email']).plot()
plt.title("Confusion matrix")
plt.show()