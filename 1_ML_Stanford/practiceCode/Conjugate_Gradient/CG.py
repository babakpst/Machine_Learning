import numpy as np

def conjgrad(A, b, x):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix 
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """  
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)
    
    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        #print(" error: ",i,np.sqrt(rsnew) )
        if np.sqrt(rsnew) < 1e-8:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
        #print(" sol: ", x)
    return x


def main():

  A = np.array([[5, -2, 0], [-2, 5, 1], [0, 1, 5]])
  b = np.array([20, 10, -10])
  #print(b)
  #print(b.size)
  x = np.zeros(b.size)
  x=conjgrad(A, b, x)

  print(" The answer is:", x)


if __name__=="__main__":
  main()
