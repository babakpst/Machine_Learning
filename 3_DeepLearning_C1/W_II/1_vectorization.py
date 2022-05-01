#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Babak Poursartip
# 04/16/2022
# Vecotorization


# In[1]:


import numpy as np
import time

def main():
    
    a = np.array([1,2,3,4],dtype=np.int)
    print(" array: {}".format(a))
    
    size = 1000000
    v1 = np.random.rand(size)
    v2 = np.random.rand(size)
    tic = time.time()
    tmp = np.dot(v1, v2)
    toc = time.time()
    
    print(" vectorized version of size {} lasts {} ms. value {}".format(size, 1000*(toc-tic), tmp))
    
    tic = time.time()
    tmp = 0
    for i in range(size):
        tmp += v1[i]*v2[i]
    toc = time.time()
    print(" non-vectorized version of size {} lasts {} ms. value {}".format(size, 1000*(toc-tic), tmp))
    
    # matrix-vector mulitplication
    size = 10
    A = np.random.rand(size,size)
    v1 = np.random.rand(size)
    u = np.dot(A, v1)
    print(" matrix-vector multiplication: {} ".format(u))    
    
    u_exp = np.exp(u)
    print(" exponent: {} ".format(u_exp))
    u_log = np.log(u)
    print(" log: {} ".format(u_log))
    u_abs = np.abs(u)
    print(" abs: {} ".format(u_abs))
    u_pow2 = u**2
    print(" pow2: {} ".format(u_pow2))
    
    v2 = np.zeros(size)
    v3 = np.zeros((size,1)) # This is a vector
    print(v2)
    print(v3)
    

if __name__=="__main__":
    main()


# In[ ]:




