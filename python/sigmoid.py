import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values
x_values = np.linspace(-7, 7, 100)

# Compute corresponding y values using the sigmoid function
y_values = sigmoid(x_values)

# Plot the sigmoid function
plt.plot(x_values, y_values, label='Sigmoid Function')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.legend()
plt.grid(True)
plt.show()
