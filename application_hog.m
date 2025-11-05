close all
% clear all
% clc

%% Application de l'HOG sur tous les images

filt=[-1 0 1]; %noyau de filtrage
T=8; %taille de la cellule HOG
N=9; %nombre de colonnes dans l'histogramme
M=1152; %nombre de caractéristiques

%positives
filenames=dir(['dataset',filesep,'Train',filesep,'pos',filesep,'*.png']);
Np=length(filenames);
train_matrix_pos=zeros(Np,M);
for ifile=1:Np
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    train_matrix_pos(ifile,:)=hogfeat';
end

%negatives
filenames=dir(['dataset',filesep,'Train',filesep,'neg',filesep,'*.png']);
Nn=length(filenames);
train_matrix_neg=zeros(Nn,M);
for ifile=1:Nn
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    train_matrix_neg(ifile,:)=hogfeat';
end


%% Application de l'ACP

train_matrix=[train_matrix_pos; train_matrix_neg];
labels_connues=[ones(Np,1);zeros(Nn,1)]; %labels pour les images: 1 avec piétons, 0 sans piétons

%ACP
%Centralisation des variables
Xc = train_matrix-1*mean(train_matrix);
%Normalisation des variables
Xcn = Xc*diag(1./std(Xc));
%Calcul de la matrice de covariance/correlation
p = size(train_matrix,1); %nombre d'individus 
Rx = (1/p)*(Xcn')*Xcn;
%Calcul des vep et vap
[U,D] = eig(Rx);
%Données transformées
V = Xcn*U;


%Etude des vap (dimension de l'espace/nombre de composantes principales à garder)
% plot(diag(D))
% title('Valeurs propres de la matrice de covariance');
% xlabel('Valeurs propres')
%on prend la dimension 30
Xp=V(:,1:30);

%Affichage
figure
%scatter(V(1:800,2),V(1:800,1),"red")
scatter3(Xp(1:800,1),Xp(1:800,2),Xp(1:800,3),'o',"red")
hold on
%scatter(V(800:1600,2),V(800:1600,1),"blue")
scatter3(Xp(800:1600,1),Xp(800:1600,2),Xp(800:1600,3),'o',"blue")
title('Projection des données sur le plan des 3 composantes principales')


%% Classification des images de test - Méthode de Parzen

%HOG sur les positives
filenames=dir(['dataset',filesep,'Test',filesep,'pos',filesep,'*.png']);
Np_test=length(filenames);
train_matrix_pos=zeros(Np_test,M);
for ifile=1:Np_test
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    test_matrix_pos(ifile,:)=hogfeat';
end

%HOG sur les negatives
filenames=dir(['dataset',filesep,'Test',filesep,'neg',filesep,'*.png']);
Nn_test=length(filenames);
test_matrix_neg=zeros(Nn,M);
for ifile=1:Nn_test
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    test_matrix_neg(ifile,:)=hogfeat';
end

test_matrix=[test_matrix_pos; test_matrix_neg]; %matrice de tous les images de test
labels=[ones(Np_test,1),zeros(Nn_test,1)]; %labels des images de test (pour la phase de validation)
N_test=Nn_test+Np_test; %nombre d'images de test

%ACP sur les images de test
Xc_test = test_matrix-1*mean(train_matrix);
Xcn_test = Xc_test*diag(1./std(Xc));
V_test = Xcn_test*U;
Xp_test=V_test(:,1:30);


% Calcul de densit� de vraissemblance par la methode de Parzen
sig=0.4; %sigma optimale pour dimension 30: 0.4
Xp0=Xp(labels_connues==0,:); %matrice d'apprenissage des images sans piétons
Xp1=Xp(labels_connues==1,:); %matrice d'apprenissage des images avec piétons
z_neg=gaussParzen(Xp_test,Xp0,sig); %densité de vraisemblance des images à la classe "sans piétons"
z_pos=gaussParzen(Xp_test,Xp1,sig); %densité de vraisemblance des images à la classe "avec piétons"

% Phase de validation 
maxi=[z_neg,z_pos];
[m,label]=max(maxi');
label=label-1;
labels_estimees=label';
for i=1:N_test
    test(i)=labels(i)-labels_estimees(i);
    err = sum(abs(test))/N_test;
end

