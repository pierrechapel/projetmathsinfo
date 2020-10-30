def gamma(t, P1, P2, u1, u2):
    matrice_base = np.array([[u1[0], u2[0]], [u1[1], u2[1]]])
    t = np.array(t)
    # P2-P1
    ax = plt.axes
    P3 = np.array([P2[0] - P1[0], P2[1]-P1[1]])
    
    quantitetest1 = u2[1]*(P2[0]-P1[0])-u2[0]*(P2[1]-P1[1])
    quantitetest2 = -u1[1]*(P2[0]-P1[0])+u1[0]*(P2[1]-P1[1])
    determinant = np.linalg.det(matrice_base)

    if ( ( determinant<0 and quantitetest1<0 and quantitetest2<0) or (determinant>0 and quantitetest1>0 and quantitetest2>0) ):
        coeff = (np.linalg.inv(matrice_base)@P3)[0] * 2
        coeff2 = (np.linalg.inv(matrice_base)@P3)[1] * 2
        a = P1[0]
        b = coeff * u1[0]
        c = P2[0]-P1[0] - coeff*u1[0]
        d = P1[1]
        e = coeff * u1[1]
        f = P2[1] - P1[1] - coeff*u1[1]
        x = a + b*t + c*t**2
        y = d + e*t + f*t**20
        plt.scatter(x,y)
        return np.array([x, y])
    else:
        # dans le meilleur des cas on ne peut interpoler que par la 
        #droite qui passe par les deux points et sinon l'énoncé dit de le faire quand même :
        x = P1[0] + (P2[0]-P1[0])*t
        y = P1[1] + (P2[1] - P1[1])*t
        plt.scatter(x,y)
        return np.array([x, y])
    ax.arrow(P1[0],P1[1],u1[0],u1[1], color ='red', head_width = 0.1 )
    ax.arrow(P2[0],P2[1],u2[0],u2[1], color ='blue', head_width = 0.1 ) 