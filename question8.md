en posant les équations du système, on obtient:
$$
\left\{
    \begin{array}{ll}
        a=x_1\\
        a+b+c=x_2\\
        d=y_1\\
        d+e+f= y_2\\
    \end{array}
\right.


$$

ainsi que 
$$

\begin{pmatrix}
b\\
e 
\end{pmatrix}
= 
\lambda. u_1

$$
où $\lambda \in \R$
et 
$$

\begin{pmatrix}
b+2c\\
e+2f
\end{pmatrix}
= 
\mu. u_2


$$

où $\mu \in \R$ .

En manipulant les équations, on obtient 
$P_2=P_1+ \lambda/2 . u_1+\mu/2 . u_2$

Cas 1 : $(u_1,u_2)$ est libre : 
on peut donc poser avec $B=(u_1,u_2)$ 
$$

P= \begin{pmatrix}
u_{11} & u_{21}\\
u_{12} & u_{22}
\end{pmatrix}

$$

la matrice inversible de changement de base.
On obtient donc: 
$$

\begin{pmatrix}
\lambda\\
\mu
\end{pmatrix}
=
2P^{-1}(P_2-P_1)
$$

On connait donc $\lambda$ et $\mu$, donc on obtient les solutions:
$$
\left\{
    \begin{array}{ll}
        a=x_1\\
        b=\lambda u_{11}\\

        c=x_2-x_1-\lambda u_{11}\\
        d=y_1\\
        e=\lambda u_{12}\\
        f=y_2-y_1-\lambda u_{12}\\
    \end{array}
\right.
$$


Cas 2 : si $(u_1,u_2)$ est liée:

on pose $u_1=\alpha . u_2, \alpha \in \R$, ce qui donne en choisissant l'existence de $\lambda$ et $\mu$ tels que 
$$

\begin{pmatrix}
b\\
e
\end{pmatrix}
=\lambda . u_1 
$$
et 
$$
\begin{pmatrix}
b+2c\\
e+2f

\end{pmatrix}
= \mu . u_1

$$

en combinant les équations pour éliminer $\lambda$ et $\mu$, on obtient finalement, si $u_{11} \not ={0}$ :
$$
y_2= \frac{u_{12}}{u_{11}} . (x_2 - x_1)+y_1
$$
 c'est à dire que $(P_1P_2)$ est la direction des vecteurs $u_1$ et $u_2$, donc une interpolation possible vérifiant les conditions est le segment $[P_1P_2]$.

Si  $u_{11}=0$ , $x_1=x_2$ et puisque $u_{12} \not={0}$ alors $u_{21}=0$ et de même, la direction $(P_1P_2)$ est bien la direction des vecteurs.

Ainsi dans le cas où $B$ est liée, l'interpolation existe si et seulement si $(P_1P_2)$ dirige les deux vecteurs $u_1$ et $u_2$ et elle prend la forme:
$$
\forall t\in [0,1]:
\gamma (t)=
\begin{pmatrix}
x_1+(x_2-x_1)t\\
y_1+(y_2-y_1)t

\end{pmatrix}

$$