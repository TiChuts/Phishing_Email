import plotly.express as px
# Comparison 
models = ['Albert - 5', 'Tiny Bert', 'XGBoost', 'Decision Tree', 'Random Forest', 'MLP Classifier', 'CNN','LSTM']
accuracies = [56.34, 55.67, 97.32, 92.82, 97.52, 98.43, 94.4, 96.98]

# Create the bar chart
fig = px.bar(x=models, y=accuracies, labels={'x': 'Models', 'y': 'Accuracy'},
             title="Performance of the models", text=accuracies)

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', marker_color='blue')

# Show the plot
fig.show()