import matplotlib.pyplot as plt

# Define the metrics and their values
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [99.68, 99.15, 99.12, 99.33]

# Create a range for the x-axis
x = range(99, 101)

# Create a line graph
plt.plot(metrics, values, marker='o', linestyle='-', color='b')

# Set y-axis limits
plt.ylim(99, 100)

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Percentage (%)')
plt.title('Performance Metrics')

# Show the graph
plt.grid()
plt.show()


