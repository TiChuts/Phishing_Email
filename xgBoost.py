from sklearn.metrics import accuracy_score,f1_score,classification_report,ConfusionMatrixDisplay,confusion_matrix
from xgboost import XGBClassifier
from Dataset.algorithms import load_and_preprocess 

import matplotlib.pyplot as plt

# Load processed data
df, x_train, x_test, y_train, y_test = load_and_preprocess()

def main():
    print("Main script running after preprocessing...")

if __name__ == "__main__":
    main()

#xgBoost
xgb = XGBClassifier()
xgb.fit(x_train,y_train)

#prediction
pred_xgb = xgb.predict(x_test)

#performance
print(f"Accuracy from XGB:{accuracy_score(y_test,pred_xgb)*100:.2f} %")
print(f"F1 Score from XGB: {f1_score(y_test,pred_xgb)*100:.2f} %")
print("Classification Report : \n",classification_report(y_test,pred_xgb))

#confusion matrix
clf_xgb = confusion_matrix(y_test,pred_xgb)
cx_ = ConfusionMatrixDisplay(clf_xgb,display_labels=['phishing_mail','safe_mail']).plot()
plt.show()