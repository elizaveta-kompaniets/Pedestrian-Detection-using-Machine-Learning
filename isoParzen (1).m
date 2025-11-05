function Z=isoParzen(appr,sigParz,N)

% appr: donnee d'apprentissage
% sigParz: std du noyau de Parzen
% N: nombre d'isocontours


% d�termination du range de calcul

xmin=min(appr(:,1)-4*sigParz);
xmax=max(appr(:,1)+4*sigParz);
ymin=min(appr(:,2)-4*sigParz);
ymax=max(appr(:,2)+4*sigParz);

% d�termination de la grille de calcul

x=(xmin:(xmax-xmin)/100:xmax);
y=(ymin:(ymax-ymin)/100:ymax);

[X,Y]=meshgrid(x,y);

XYvec=zeros(2,size(X,1)*size(X,2));
XYvec(1,:)=reshape(X,1,size(X,1)*size(X,2));
XYvec(2,:)=reshape(Y,1,size(X,1)*size(X,2));

% calcul de la densit� et affichage

Z=gaussParzen(XYvec',appr,sigParz);
Z=reshape(Z,size(X,1),size(X,2));
contour(X,Y,Z,N);  

% visu de la densit� en 3D
% figure,
% surf(X,Y,Z);