%%% HOG sur une seule image

%% Chargement de l'image
Im=imread("O:\Documents\2A\Bloc 10\Bureau d'études\dataset\Train\pos\B10_crop001001c.png");
%imshow(Im);

% Conversion de l'image en niveau de gris
Im_g=rgb2gray(Im);
%imshow(Im_g);

% Affichage
% figure, imshowpair(Im,Im_g,'montage');
% title("Image originale et image en noir et blanc") 

%% Filtrer l'image (passe-haut: gradient)
h1 = [-1 0 1]; %noyau du filtrage
G_ligne = double(imfilter(Im_g,h1,'conv')); %image filtrée par lignes , %double pour la double precision
G_colonne = double(imfilter(Im_g,h1','conv')); %image filtrée par colonnes
%imshow(G_ligne);

%Affichage
% figure, imshowpair(G_ligne,G_colonne,'montage');
% title("Image filtrée par ligne; Image filtrée par colonne ")

%% Calcul de la norme et de l'orientation du gradient

%Norme
Norme_grad=sqrt(G_ligne.^2+G_colonne.^2); 
max_grad=max(max(Norme_grad)); %max de la norme du gradient 
%imshow(Norme_grad/max_grad); %pour affichage on normalise les normes du grad en divisant par le max

%Orientation
Orient_grad=atan((G_colonne./G_ligne));
Orient_grad(Orient_grad<0)=Orient_grad(Orient_grad<0)+pi; %pour eviter les valeurs négatives
%imshow(Orient_grad);

% Affichage
figure, imshowpair(Norme_grad/max_grad,Orient_grad,'montage');
title("Norme et orientation du gradient")

%% Calcul de HOG

[H,W]=size(Im_g); %taille de l'image
T=8; %taille de la cellule HOG
N=9; %nombre de colonnes dans l'histogramme

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

%% Visualisation des HOG
% param pour la visu:
param.ImageSize=[H W];
param.WindowSize=[H W];
param.CellSize=[8 8];
param.BlockSize=[1 1];
param.BlockOverlap=[0 0];
param.NumBins=9;
param.UseSignedOrientation=0;
%visualisation
visu=vision.internal.hog.Visualization(hog, param);
figure; imshow(uint8(Im_g)); 
hold on;
plot(visu)
title('HOGs manual')
pause(0.1)
