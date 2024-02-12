import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.fmax(0,x)
    
def Linear(x):
    return x

# Generate x values
x_values = np.linspace(-7, 7, 100)

# Compute corresponding y values using the sigmoid function
y_values = sigmoid(x_values)

# Plot the sigmoid function
plt.plot(x_values, y_values, label='Sigmoid Function')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5) 
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.legend()
plt.grid(True)
plt.show()

y_relu = ReLU(x_values)

plt.plot(x_values, y_relu, label='ReLU function')
plt.title('Rectified Linear Unit Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5)
#plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.legend()
plt.grid(True)
plt.show()

y_lin = Linear(x_values)

plt.plot(x_values, y_lin, label='Linear function')
plt.title('Linear Function')
plt.xlabel('x')
plt.ylabel('x')
plt.axhline(0, color='black', linewidth=.5)
plt.axvline(0, color='black', linewidth=.5)
#plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.legend()
plt.grid(True)
plt.show()

