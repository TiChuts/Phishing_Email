import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from Dataset.algorithms import load_and_preprocess 

# Load processed data
df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

# MLP Classifier
mlp = MLPClassifier() 
mlp.fit(x_train,y_train)

# Prediction
pred_mlp = mlp.predict(x_test)

# Performance
print(f"Accuracy from MLP:{accuracy_score(y_test,pred_mlp)*100:.2f} %")
print(f"F1 Score from MLP: {f1_score(y_test,pred_mlp)*100:.2f} %")
print("Classification Report : \n",classification_report(y_test,pred_mlp))

# Confusion Matrix
clf_mlp = confusion_matrix(y_test,pred_mlp)
cx_ = ConfusionMatrixDisplay(clf_mlp,display_labels=['Phishing Email','Safe Email']).plot()
plt.title("Confusion Matrix")
plt.show()
