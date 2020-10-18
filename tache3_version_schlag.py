
# Autograd & Numpy
import autograd
import autograd.numpy as np

# Pandas
import pandas as pd

#math
import math as math


# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10] # [width, height] (inches). 

# Jupyter & IPython
from IPython.display import display


def grad(f):
    g = autograd.grad
    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f


def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f


def f1(x1,x2):
    return 3.0*x1*x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2 -0.8

def display_contour(f, x, y, levels):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig, ax = plt.subplots()
    contour_set = plt.contour(
        X, Y, Z, colors="grey", linestyles="dashed", 
        levels=levels 
    )
    ax.clabel(contour_set)
    plt.grid(True)
    plt.xlabel("$x_1$") 
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal")

def f2(x1,x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2





#question 6 et tache 3
# 
def g(theta, x0, y0,f, delta):
    c=f(x0,y0)
    gra=grad(f)(x0,y0)
    vecteur_unitaire= gra/(np.linalg.norm(gra))

    

    matrice_rotation_theta = np.array( [ [np.cos(theta) , np.sin(-theta)] , [np.sin(theta) , np.cos(theta) ] ] )
    
    argument = np.transpose( np.array( [x0, y0] ) ) + delta*(matrice_rotation_theta @ vecteur_unitaire)

    L=np.transpose(argument)

    return f(L[0],L[1]) - c




def newtonréel(g,a0,x0, y0,f, delta ,n,eps):
    c=f(x0,y0)

    for i in range(n):
        h1=g(a0, x0, y0,f,delta)
        h2=autograd.grad(g,0)(a0,x0,y0,f,delta)



        a0=a0-(h1/h2)
      

        if abs(g(a0,x0,y0,f,delta))<eps:

            gra=grad(f)(x0,y0)

            vecteur_unitaire= gra/(np.linalg.norm(gra))

            matrice_rotation_theta= np.array([ [np.cos(a0),np.sin(-a0)] , [np.sin(a0) , np.cos(a0) ] ] )

                
        
            solution = np.transpose( np.array( [x0, y0] ) ) + delta*(matrice_rotation_theta @ vecteur_unitaire)





            return solution

        
    else:
         raise ValueError(f"no convergence in {n} steps.") 


def level_curve(f, x0, y0, eps, delta=0.01, N=100,n=100):
    
    c=f(x0,y0)
    theta0 = - (math.pi)/2

    points= np.zeros(shape=(2,N))


















    for i in range(N):
        points[0][i] = newtonréel(g, theta0, x0, y0,f,delta,n,eps)[0]
        points[1][i] = newtonréel(g, theta0, x0, y0,f,delta,n,eps)[1]

        x0=points[0][i]
        y0=points[1][i]


    return points





def interface_graphique(points):
    plt.scatter(points[0],points[1])
    plt.show()


interface_graphique(level_curve(f2,0.0,0.0,0.001))







#question 7 tache 4

#necessairement si la courbe de niveau fait un noeud, alors en ce point le gradient est nul.   

