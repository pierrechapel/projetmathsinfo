
# Autograd & Numpy
from IPython.display import display
import autograd
import autograd.numpy as np

# Pandas
import pandas as pd

# math
import math as math


# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]  # [width, height] (inches).

# Jupyter & IPython


def grad(f):
    g = autograd.grad

    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f


eps = 10**-12
N = 100


def J(f):
    j = autograd.jacobian

    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f


def f1_Newton(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.array([3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2 - 0.8, x1-x2])


def f1(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2


def Newton(F, x0, y0, eps=0.01, N=100):
    yn = y0+eps

    for i in range(N):

        a = np.array([x0, yn]) - np.linalg.inv(J(F)(x0, yn))@F(x0, yn)
        xn, yn = a[0], a[1]
        if np.sqrt((xn - x0)**2 + (yn - y0)**2) <= eps:
            return np.array([xn, yn])
        x0, y0 = xn, yn
    else:
        raise ValueError(f"no convergence in {N} steps.")


# question 6 et tache 3
''''
def display_contour(f, tableau, x, y, levels):
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
    plt.scatter(tableau[0], tableau[1])
    plt.gca().set_aspect("equal")
    plt.show()

def level_curve(f, x0, y0, eps, delta=0.1, N=100,n=100):
    global pas
    pas = delta
    points= np.zeros(shape=(2,N))
    points[0][0], points[1][0]= x0, y0
    for i in range(1,N) :
        global x
        global y
        x, y = points[0,i-1], points[1,i-1]
        gra = grad(f)(x,y)
        direction = (matrice_rotation@gra)*delta/np.linalg.norm(gra)
        x1, y1 =  x + direction[0] , y + direction[1]
        # La boucle est-elle bouclée ?
        if np.sqrt((x1-points[0][0])**2 + (y1-points[1][0])**2) <delta/2 :
            return points[:,0:i]
        points[0][i], points[1][i] = Newton(fonction_avec_contrainte, x1, y1, eps, n)[0],Newton(fonction_avec_contrainte, x1, y1, eps, n)[1]
    return points
'''


def f2(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2


def fonction_avec_contrainte(x1, y1, c=1.0):
    return np.array([f2(x1, y1) - c, ((x1-x)**2+(y1-y)**2) - pas**2])


matrice_rotation = np.array([[0, 1], [-1, 0]])
'''
coordonnées = level_curve(f2, 0.0,0.0,0.01,0.1,1000,100)
plt.show(display_contour(f2, coordonnées, x=np.linspace(-1.0, 3.0, 100), 
    y=np.linspace(-2.0, 2.0, 100), 
    levels=[2**i for i in range(-3, 8)]))'''
# question 7 tache 4

# necessairement si la courbe de niveau fait un noeud, alors en ce point le gradient est nul.

# tâche 4:


def intersection_segment(a0, b0, a1, b1, a2, b2, a3, b3):
    a = (b1-b0)/(a1-a0)
    b = b1 - a*a1
    c = (b3-b2)/(a3-a2)
    d = b3 - c*a3

    if ((b2 >= a*a2 + b and b3 <= a*a3 + b) or (b2 <= a*a2 + b and b3 >= a*a3 + b)) and ((b0 >= c*a0 + d and b1 <= c*a1 + d) or (b0 <= c*a0 + d and b1 >= c*a1 + d)):
        return False
    else:
        return True

# on définit alors le nouveau programme level_curve :


def level_curve4(f, x0, y0, eps, delta=0.1, N=100, n=100):
    global pas
    pas = delta
    points = np.zeros(shape=(2, N))
    points[0][0], points[1][0] = x0, y0
    for i in range(1, N):
        global x
        global y
        x, y = points[0, i-1], points[1, i-1]
        gra = grad(f)(x, y)
        direction = (matrice_rotation@gra)*delta/np.linalg.norm(gra)
        x1, y1 = x + direction[0], y + direction[1]
        # La boucle est-elle bouclée ?
        points[0][i], points[1][i] = Newton(fonction_avec_contrainte, x1, y1, eps, n)[
            0], Newton(fonction_avec_contrainte, x1, y1, eps, n)[1]
        if intersection_segment(x0, y0, points[0, 1], points[1, 1], points[0][i-1], points[1][i-1], points[0][i], points[1][i]) == False and (i != 1 and i != 2):
            return points[:, 0:i]
    return points


def display_contour(f, tableau, x, y, levels):
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
    plt.scatter(tableau[0], tableau[1])
    plt.gca().set_aspect("equal")
    plt.show()


def f2(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2


def fonction_avec_contrainte(x1, y1, c=1.0):
    return np.array([f2(x1, y1) - c, ((x1-x)**2+(y1-y)**2) - pas**2])


matrice_rotation = np.array([[0, 1], [-1, 0]])

coordonnées = level_curve4(f2, 0.0, 0.0, 0.01, 0.1, 1000, 100)
'''plt.show(display_contour(f2, coordonnées, x=np.linspace(-1.0, 3.0, 100),
                         y=np.linspace(-2.0, 2.0, 100),
                         levels=[2**i for i in range(-3, 8)]))
'''
# tache n°6 :


def gamma(t, P1, P2, u1, u2):
    matrice_base = np.array([[u1[0], u2[0]], [u1[1], u2[1]]])
    t = np.array(t)
    # P2-P1
    P3 = np.array([P2[0] - P1[0], P2[1]-P1[1]])
    if np.linalg.det(matrice_base) != 0:
        coeff = (np.linalg.inv(matrice_base)@P3)[0] * 2
        coeff2 = (np.linalg.inv(matrice_base)@P3)[1] * 2
        a = P1[0]
        b = coeff * u1[0]
        c = P2[0]-P1[0] - coeff*u1[0]
        d = P1[1]
        e = coeff * u1[1]
        f = P2[1] - P1[1] - coeff*u1[1]
        x = a + b*t + c*t**2
        y = d + e*t + f*t**2
        return np.array([x, y])
    else:
        # dans le meilleur des cas on ne peut interpoler que par la droite qui passe par les deux points et sinon l'énoncé dit de le faire quand même :
        x = P1[0] + (P2[0]-P1[0])*t
        y = P1[1] + (P2[1] - P1[1])*t
        return np.array([x, y])


# tache 7 : # on implémente une nouvelle fonction levelcurve qui interpole désormais la courbe de niveau
# le nouveau paramètre écart est le nombre de points que l'on choisit d'avoir dans l'implémentation
def level_curve7(f, x0, y0, eps, oversampling, delta=0.1, N=100, n=100):
    if oversampling == 1:
        return level_curve4(f, x0, y0, eps, delta=0.1, N=100, n=100)
    else:
        global pas
        pas = delta
        points = np.zeros(shape=(2, (N*oversampling)))
        points[0][0], points[1][0] = x0, y0
        for i in range(1, N):
            global x
            global y
            x, y = points[0, (i-1)*oversampling], points[1, (i-1)*oversampling]
            gra = grad(f)(x, y)
            direction = (matrice_rotation@gra)*delta/np.linalg.norm(gra)
            x1, y1 = x + direction[0], y + direction[1]
            # La boucle est-elle bouclée ?
            points[0][i*oversampling], points[1][i*oversampling] = Newton(fonction_avec_contrainte, x1, y1, eps, n)[
                0], Newton(fonction_avec_contrainte, x1, y1, eps, n)[1]
            tangente1 = matrice_rotation@gra
            tangente2 = matrice_rotation@grad(f)(
                points[0][i*oversampling], points[1][i*oversampling])
            interpolation = gamma(np.linspace(0, 1, oversampling+1), [x, y], [
                                  points[0][i*oversampling], points[1][i*oversampling]], tangente1, tangente2)
            for j in range(1, len(interpolation)-1):
                points[0][(i-1)*oversampling+j] = interpolation[0][j]
                points[1][(i-1) * oversampling + j] = interpolation[1][j]
                if intersection_segment(x0, y0, points[0, 1], points[1, 1], points[0][(i-1)*oversampling+j-1], points[1][(i-1)*oversampling+j-1],
                                        points[0][(i-1)*oversampling+j], points[1][(i-1)*oversampling+j]) == False and ((i-1)*oversampling+j != 1 and (i-1)*oversampling+j != 2):
                    print(i, j)
                    return points[:, 0:(i-1)*oversampling+j]

        return points


coordonnées7 = level_curve7(
    f2, 0.0, 0.0, 0.01, 1000, 0.2, 100, 100)

display_contour(f2, coordonnées7, x=np.linspace(-1.0, 3.0, 100),
                y=np.linspace(-2.0, 2.0, 100),
                levels=[2**i for i in range(-3, 8)])
