import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,f1_score,classification_report,ConfusionMatrixDisplay,confusion_matrix
from Dataset.algorithms import load_and_preprocess 
from sklearn.ensemble import RandomForestClassifier

# Load processed data
df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

# Random Forest
rnf = RandomForestClassifier() 
rnf.fit(x_train,y_train)

# Prediction
pred_rnf = rnf.predict(x_test)

# Performance
print(f"Accuracy from rnadom forest:{accuracy_score(y_test,pred_rnf)*100:.2f} %")
print(f"F1 score from random forest: {f1_score(y_test,pred_rnf)*100:.2f} %")
print("Classification report : \n",classification_report(y_test,pred_rnf))

# Confusion matrix
clf_rnf = confusion_matrix(y_test,pred_rnf)
cx_ = ConfusionMatrixDisplay(clf_rnf,display_labels=['Phishing Email','Safe Email']).plot()
plt.title("Confusion matrix")
plt.show()
