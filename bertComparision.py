import plotly.express as px
# Comparison 
models = ['Albert', 'Tiny Bert', 'Electra']
accuracies = [73.97, 61.49, 61.49]

# Create the bar chart
fig = px.bar(x=models, y=accuracies, labels={'x': 'Models', 'y': 'Accuracy'},
             title="Performance of the models", text=accuracies)

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', marker_color='blue')

# Show the plot
fig.show()