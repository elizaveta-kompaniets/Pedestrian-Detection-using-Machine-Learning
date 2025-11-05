function[hog]=hogfeatures(Im_g,filt,T,N)

%% Filtrer l'image (passe-haut: gradient)
% h1 = [-1 0 1]; %noyau du filtrage
% h2=h1';
G_ligne = imfilter(Im_g,filt,'conv'); %image filtrée par lignes
G_colonne = imfilter(Im_g,filt','conv'); %image filtrée par colonnes
%imshow(G_ligne);

%% Calcul de la norme et de l'orientation du gradient

%Norme
Norme_grad=sqrt(double(G_ligne.^2+G_colonne.^2)); %double pour la double precision (?)
max_grad=max(max(Norme_grad)); %max de la norme du gradient (calculé seulement pour l'affichage)
%imshow(Norme_grad/max_grad); %pour affichage on normalise les normes du grad en divisant par le max

%Orientation
Orient_grad=atan(double(G_colonne./G_ligne));
Orient_grad(Orient_grad<0)=Orient_grad(Orient_grad<0)+pi; %pour eviter les valeurs négatives
%imshow(Orient_grad);

%Affichage
% figure, imshowpair(Norme_grad/max_grad,Orient_grad,'montage');
% title("Norme et orientation du gradient")

%% Calcul de HOG

[H,W]=size(Im_g); %taille de l'image
% T=8; %taille de la cellule HOG
% N=9; %nombre de colonnes dans l'histogramme

%Regroupement des cellules en sous-blocs
hog=[]; %initialisation
for i=1:T:H
    for j=1:T:W
        cellule_norm=Norme_grad(i:i+T-1,j:j+T-1); %bloc 8x8
        cellule_norm=cellule_norm(:); %transformation en vecteur-colonne
        cellule_orr=Orient_grad(i:i+T-1,j:j+T-1);
        cellule_orr=cellule_orr(:);
        for k=1:N
            intervalle=[(k-1)*(pi/N),k*(pi/N)]; %limites de chaque colonne
            ind=find(cellule_orr>=intervalle(1) & cellule_orr<intervalle(2)); %indices des orientations correspondant à chaque bin
            hogcell(k)=sum(cellule_norm(ind)); %valeur de l'histogramme pour chaque colonne dans l'histogramme
        end
        hog=[hog,hogcell];
    end
end