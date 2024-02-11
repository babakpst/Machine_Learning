#!/usr/bin/env python
# coding: utf-8

# ## Steepest descent
# **Babak Poursartip**

# In[1]:


import sys
sys.executable


# Importing the required libraries

# In[2]:


import numpy as np
import numpy.linalg as la # linear algebra

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
# if using a Jupyter notebook, include:
#get_ipython().run_line_magic('matplotlib', 'inline')


# Defining the function and its gradient

# In[3]:


# x, the indep. variable of the func., is an array of size n
def f(x):
  return 0.5*x[0]**2 + 2.5*x[1]**2
  #return np.sin(0.5*x[0]**2-0.5*x[1]**2+3)*np.cos(2*x[0] + 1 -np.exp(x[1]))

def df(x):
  # put the derivative with respect to each component in the array
  d1 = x[0]*np.cos(2*x[0] - np.exp(x[1]) +1)*np.cos(0.5*x[0]**2-0.25*x[1]**2 +3)-2*np.sin( 2*x[0]-np.exp(x[1]+1))*np.sin( 0.5*x[0]**2 -0.25*x[1]**2 +3 )
  d2 = np.exp(x[1]) * np.sin( 2*x[0]-np.exp(x[1])+1) * np.sin( 0.5*x[0]**2 -0.25*x[1]**2 +3)  - 0.5*x[1]* np.cos(2*x[0] - np.exp(x[1]) +1)*np.cos( 0.5*x[0]**2 -0.25*x[1]**2 +3)
  #return np.array([d1 , d2])
  return np.array([x[0], 5*x[1]])


# Let's plot the function here:

# In[4]:


def plot_func(guesses):
  # 3d plot of the function
  Title      = "function"
  fig = plt.figure()
  ax = fig.gca(projection="3d")
  xmesh, ymesh = np.mgrid[-1:3:50j,-1:3:50j]

  fmesh = f(np.array([xmesh, ymesh]))
  surf =ax.plot_surface(xmesh, ymesh, fmesh, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
  plt.title(Title, fontsize = 16)
  plt.xlabel("x_1",     fontsize=12)
  plt.ylabel("x_2",     fontsize=12)
  #plt.zlabel("f(x)",     fontsize=12)
  fig.colorbar(surf, shrink=0.5, aspect=5)


  # contour plot of the function
  fig,ax = plt.subplots()  
  plt.axis('equal')
  plt.contour(xmesh, ymesh, fmesh,20)
  plt.colorbar();
  plt.xlabel("x_1",     fontsize=12)
  plt.ylabel("x_2",     fontsize=12)
  it_array = np.array(guesses)
  plt.plot(it_array.T[0], it_array.T[1], "x-")
  plt.show()

# Steepest descent

# In[5]:


def steepest_descent(tol, max_iters):
  print("{}".format(" steepest descent starts ..."))
  # initial guess
  #x1 = np.zeros(2)
  x1 = [10.0,20.5]
  x0 = x1
  guesses = [x1]
  iters = 0 #iteration counter
  step_size = 0.0001
  
  while ((step_size > tol) and (iters < max_iters)):
    s0 = -df(x0)
    s1 = -df(x1)  
    if iters != 0:
      rate = abs( np.dot( ((x1-x0).transpose()),(s1-s0)))/( pow((la.norm(s1-s0)),2) )
    else:
      rate = 0.00001

    x0 = x1 #Store current x value in prev_x    
    x1 = x1 - rate * df(x0) #Grad descent
    step_size = abs(la.norm(x1 - x0)) #Change in x
    iters = iters+1 #iteration count
    guesses.append(x1)

    print("{:} {:5d} {:} {:} {:} {:10.3e} {:} {:6.3f}".format("Iteration:",iters," \n sol:",x1, "step size:", step_size, "rate: ", rate))
    
  print("The local minimum occurs at", x0)
  return guesses


# The main function

# In[6]:



def main():
    print(" Steepest descent code starts ...")
 
    tol = 0.000001 #This tells us when to stop the algorithm
    max_iters = 10000 # maximum number of iterations
    guesses = steepest_descent(tol, max_iters)

    plot_func(guesses)
    
    print(" Steepest descent terminates successfully!")

if __name__ =="__main__":
    main()


# In[ ]:




