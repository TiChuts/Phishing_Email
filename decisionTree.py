import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,f1_score,classification_report,ConfusionMatrixDisplay,confusion_matrix
from Dataset.algorithms import load_and_preprocess 
from sklearn.tree import DecisionTreeClassifier

# Load processed data
df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

# Decision Tree
dtr = DecisionTreeClassifier() 
dtr.fit(x_train,y_train)

# Prediction
pred_dtr = dtr.predict(x_test)

# Performance
print(f"Accuracy from Decision Tree:{accuracy_score(y_test,pred_dtr)*100:.2f} %")
print(f"F1 score from Decision Tree: {f1_score(y_test,pred_dtr)*100:.2f} %")
print("Classification report : \n",classification_report(y_test,pred_dtr))

# Confusion matrix
clf_dtr = confusion_matrix(y_test,pred_dtr)
cx_ = ConfusionMatrixDisplay(clf_dtr,display_labels=['Phishing Email','Safe Email']).plot()
plt.title("Confusion matrix")
plt.show()