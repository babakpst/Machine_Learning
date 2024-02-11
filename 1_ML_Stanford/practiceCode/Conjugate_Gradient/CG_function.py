import numpy as np
import numpy.linalg as la # linear algebra

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

def ddf(x):
  return np.array([[1, 0], [0,5]])

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


def conjgrad(tol, max_iters):

    x = [10.0,20.5]
    guesses = [x]
    iters = 0 #iteration counter

    r = -df(x)
    p = r
    rsold = np.dot(np.transpose(r), r)
    
    for i in range(max_iters):
        first_der = df(x)
        second_der = ddf(x)
        #print("p",p)
        #print("df", first_der)
        #print((np.dot(np.transpose(first_der),p) ))
        #print(( (np.matmul((np.matmul(np.transpose(p),second_der)),p)) ))
        alpha = - (np.dot(np.transpose(first_der),p) )/ ( (np.matmul((np.matmul(np.transpose(p),second_der)),p)) )
        #print("alpha",alpha)
        x = x + np.dot(alpha, p)
        #print("x",x)
        r = -df(x)
        rsnew = np.dot(np.transpose(r), r)
        #print(" error: ",i,np.sqrt(rsnew) )
        guesses.append(x)
        if (np.sqrt(rsnew) < 1e-8) and (i>1):
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
        print(" sol: ", x)
    print(" final sol: ", x)
    return guesses


def main():
    print(" Conjugate gradient code starts ...")
 
    tol = 0.000001 #This tells us when to stop the algorithm
    max_iters = 10 # maximum number of iterations
    guesses = conjgrad(tol, max_iters)

    plot_func(guesses)
    
    print(" Conjugate gradient terminates successfully!")

if __name__=="__main__":
  main()
