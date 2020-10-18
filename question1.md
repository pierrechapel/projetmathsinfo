montrons que  $E_{c} = \{(x,y) \in \R | f(x,y)=c \}$ est un fermé et borné.

 $E_{c}$ est fermé car il est l'image réciproque de $\{c\}$ par $f$ continue.

 Supposons qu'il n'est pas borné: on peut donc construire une suite $(X_{k})_{k\in \N}$ telle que

 $\forall k \in \N,  f(X_{k})=c$ et $\parallel X_{k} \parallel > 2^{k}$ 

 Ainsi en passant à la limite sur $k$ on obtient puisque $E_{c}$ est fermé $\lim\limits_{k \to \infin} f(X_{k}) = c$, or l'hypothèse donne  $\lim\limits_{k \to \infin} f(X_{k}) = +\infin$, ce qui est absurde.

 Ainsi $E_{c}$ est bien borné. 
