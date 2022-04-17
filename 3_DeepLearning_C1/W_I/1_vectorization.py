#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Babak Poursartip
# 04/16/2022
# Vecotorization


# In[4]:


import numpy as np
import time

def main():
    
    a = np.array([1,2,3,4])
    print(" array: {}".format(a))
    
    size = 1000000
    v1 = np.random.rand(size)
    v2 = np.random.rand(size)
    tic = time.time()
    c = np.dot(v1, v2)
    toc = time.time()
    
    print(" vectorized version of size {} lasts {} ms.".format(size, 1000*(toc-tic)))
    
    tic = time.time()
    tmp = 0
    for i in range(size):
        tmp += v1[i]*v2[i]
    toc = time.time()
    print(" non-vectorized version of size {} lasts {} ms.".format(size, 1000*(toc-tic)))
    
    


if __name__=="__main__":
    main()


# In[ ]:





# In[ ]:




