import plotly.express as px
# Comparison 
models = ['Naive Bayes', 'Logistic Regression', 'XGBoost', 'Decision Tree', 'Random Forest', 'MLP Classifier', 'CNN','LSTM','SVM']
accuracies = [97.55, 97.95, 97.32, 93.1, 97.58, 98.43, 97.73, 97.46, 49.9]

# Create the bar chart
fig = px.bar(x=models, y=accuracies, labels={'x': 'Models', 'y': 'Accuracy'},
             title="Performance of the models", text=accuracies)

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', marker_color='blue')

# Show the plot
fig.show()