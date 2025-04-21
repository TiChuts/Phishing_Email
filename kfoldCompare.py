import plotly.express as px
# Comparison 
models = ['Naive Bayes', 'Logistic Regression', 'XGBoost', 'Decision Tree', 'Random Forest', 'MLP Classifier', 'CNN']
accuracies = [97.1, 97.45, 96.84, 91.98, 97.23, 98.08, 98.08]

# Create the bar chart
fig = px.bar(x=models, y=accuracies, labels={'x': 'Models', 'y': 'Accuracy'},
             title="Performance of the models", text=accuracies)

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', marker_color='blue')

# Show the plot
fig.show()