#!/usr/bin/env python
# coding: utf-8

# In[16]:



# Babak Poursartip
# April 17, 2022
# Broadcasting:
# General rule: 
# A(mxn) +,-,*,/ B(1xn)  --> B will be converted to  B(mxn)
# A(mxn) +,-,*,/ B(mx1)  --> B will be converted to  B(mxn)
# A(mx1) +,-,*,/ scalar  --> scalar will be converted to  B(mx1)
# A(1xn) +,-,*,/ scalar  --> scalar will be converted to  B(1xn)


import numpy as np

def main():
  print(" Broadcasting:")

  A = np.array([
      [10,12,11,5.5],
      [4.2,4.6,8.6,7.5],
      [12.6,45,62.2,5.5],
  ])

  print(A)

  b = A.sum(axis=0) # size: 1x4
  print(" sum A: \n{}".format(b))
  
  # Broacasting: python auto expands b (1x4) to the size of A (3x4).
  B = 100*A/A.sum(axis=0)
  print(" percents: \n{}".format(B))

  # Broadcasting: converting a single number to a vector
  c = np.array([
      [1],
      [2],
      [3],
      [4]
  ])
  print(" vector before adding: \n {}".format(c))
  d = c + 100
  print(" vector after adding: \n {}".format(d))
  
  # Broadcasting: converting a matrix (1xn) to (mxn) 
  E = np.array([
      [1,2,3],
      [5,6,7]
  ])
  print(" E: \n{}".format(E))
  
  f = np.array([10, 20, 40])
  print(" f: \n{}".format(f))

  G = E + f
  print(" G: \n{}".format(G))

  h = np.array([[10], [20]])
  print(" h: \n{}".format(h))
  
  I = E + h
  print(" I: \n{}".format(I))


if __name__=="__main__":
    main()
    


# In[ ]:




