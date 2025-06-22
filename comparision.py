import plotly.express as px
# Comparison 
models = ['Naive Bayes', 'Logistic Regression', 'XGBoost', 'Decision Tree', 'Random Forest', 'MLP Classifier', 'CNN','LSTM','SVM','Hybrid']
accuracies = [97.55, 97.95, 97.32, 93.04, 97.95, 98.4, 97.45, 98.31, 49.9, 61.49]

# Create the bar chart
fig = px.bar(x=models, y=accuracies, labels={'x': 'Models', 'y': 'Accuracy'},
             title="Performance of the models", text=accuracies)

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', marker_color='blue')

# Show the plot
fig.show()