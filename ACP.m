function[U,D,V]=ACP(X)

%Centralisation des variables
Xc = X-1*mean(X);
%Normalisation des variables
Xcn = Xc*diag(1./std(Xc));
%Calcul de la matrice de covariance/correlation
p = size(X,1); %nombre d'individus 
Rx = (1/p)*(Xcn')*Xcn;
%Calcul des vep et vap
[U,D] = eig(Rx);
%Données transformées
V = Xcn*U;
%Tableau final
% Xp=V(:,1:2);