
import numpy as np




def printMe(a):
  sizes = a.shape

  print(" sizes: {}".format(sizes))

  for i in range(sizes[0]):
    print(" i: ".format(i))
    for j in range(sizes[1]):
      for k in range(sizes[2]):
        print("{:5.2f} ".format(a[i][j][k]),  end=" ")
      print()




#a = np.random.randn(5,4,6)
#a = np.random.randn(5,4,3)
a = np.ones((5,4,3))
printMe(a)

#a = np.pad(a,(2,2), mode='constant', constant_values=(0,0))
a = np.pad(a, ((0,0), (2,3), (0,0)), mode='constant', constant_values=(0,0))
printMe(a)







