
import numpy as np
import tensorflow as tf

w = tf.Variable(0,dtype=tf.float32)

optimizer = tf.keras.optimizers.Adam(0.1)  # learning reate 0.1

def train_step():
  with tf.GradientTape() as tape:
    cost = w**2 -10*w +25
  trainable_variables = [w]
  grads = tape.gradient(cost, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))

print(w)
train_step()
print(w)

for i in range(1000):
  train_step()
  
print(w)

# if our neural network depends on x on top of the W (network parameters)

print(" new ")
w = tf.Variable(0,dtype=tf.float32)  
x = np.array([1.0, -10.0, 25.0], dtype=np.float32) # coefficient of the cost fucntion
optimizer = tf.keras.optimizers.Adam(0.1)  # learning reate 0.1

def training(x,w, optimizer):
  def cost_fun():
    return x[0] * w **2 + x[1] * w + x[2]

  for i in range(1000):
    optimizer.minimize(cost_fun, [w])
    
  return w
  
w = training(x, w, optimizer) 

print(w)
