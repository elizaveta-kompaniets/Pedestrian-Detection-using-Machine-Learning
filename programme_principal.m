close all
clear all
clc

%% Application de l'HOG sur les images d'apprentissage

filt=[-1 0 1]; %noyau de filtrage
T=8; %taille de la cellule HOG
N=9; %nombre de colonnes dans l'histogramme
M=1152; %nombre de caractristiques

%Debut Timer HOG
HOG_start=tic;

%positives
filenames=dir(['dataset',filesep,'Train',filesep,'pos',filesep,'*.png']);
Np=length(filenames); %nombre d'images d'apprentisage avec pitones
train_matrix_pos=zeros(Np,M);
for ifile=1:Np
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    train_matrix_pos(ifile,:)=hogfeat';
end


%negatives
filenames=dir(['dataset',filesep,'Train',filesep,'neg',filesep,'*.png']);
Nn=length(filenames); %nombre d'images d'apprentisage sans pitones
train_matrix_neg=zeros(Nn,M);
for ifile=1:Nn
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    train_matrix_neg(ifile,:)=hogfeat';
end

train_matrix=[train_matrix_pos; train_matrix_neg];
labels_connues=[ones(Np,1);zeros(Nn,1)]; %labels pour les images: 1 avec pitons, 0 sans pitons

%Fin Timer HOG
HOG_end=toc(HOG_start);

%% Application de l'ACP sur les images d'apprentissage

%Debut Timer ACP
ACP_start=tic;

%Centralisation des variables
Xc = train_matrix-1*mean(train_matrix);
%Normalisation des variables
Xcn = Xc*diag(1./std(Xc));
%Calcul de la matrice de covariance/correlation
p = size(train_matrix,1); %nombre d'individus 
Rx = (1/p)*(Xcn')*Xcn;
%Calcul des vep et vap
[U,D] = eig(Rx);
%Donnes transformes
V = Xcn*U;

%%Etude des vap (dimension de l'espace/nombre de composantes principales  garder)
% plot(diag(D))
% title('Valeurs propres de la matrice de covariance');
% xlabel('Valeurs propres')
dim=30; % On prend la dimension 30 (critre de coude + variance explique)
Xp=V(:,1:dim);

%Fin Timer ACP
ACP_end=toc(ACP_start);

% %Affichage
% figure
% scatter3(Xp(1:800,1),Xp(1:800,2),Xp(1:800,3),'o',"red")
% hold on
% scatter3(Xp(800:1600,1),Xp(800:1600,2),Xp(800:1600,3),'o',"blue")
% 
% xlabel('CP 1')
% ylabel('CP 2')
% zlabel('CP 3')
% title('Projection des donnes d apprentissage sur le plan des 3 composantes principales')



%% Application de l'HOG et de l'ACP sur les images de test

%Debut Timer HOG
HOG_test_start=tic;

%HOG sur les positives
filenames=dir(['dataset',filesep,'Test',filesep,'pos',filesep,'*.png']);
Np_test=length(filenames);
test_matrix_pos=zeros(Np_test,M);
for ifile=1:Np_test
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    test_matrix_pos(ifile,:)=hogfeat';
end

%HOG sur les negatives
filenames=dir(['dataset',filesep,'Test',filesep,'neg',filesep,'*.png']);
Nn_test=length(filenames);
test_matrix_neg=zeros(Nn_test,M);
for ifile=1:Nn_test
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    test_matrix_neg(ifile,:)=hogfeat';
end

test_matrix=[test_matrix_pos; test_matrix_neg]; %matrice de tous les images de test
labels=[ones(Np_test,1);zeros(Nn_test,1)]; %labels des images de test (pour la phase de validation)
N_test=Nn_test+Np_test; %nombre d'images de test

%Fin Timer HOG
HOG_test_end=toc(HOG_test_start);

%Debut Timer ACP
ACP_test_start=tic;

%ACP sur les images de test
Xc_test = test_matrix-1*mean(train_matrix);
Xcn_test = Xc_test*diag(1./std(Xc));
V_test = Xcn_test*U;
Xp_test=V_test(:,1:dim);

%Fin Timer ACP
ACP_test_end=toc(ACP_test_start);

%% Classification des images de test - Mthode de Parzen

% Debut Timer Parzen
parzen_start=tic;

% Phase d'apprentissage et de classification - Calcul de densit de vraissemblance par la methode de Parzen
sig=0.375; %sigma optimale pour dimension 30: 0.4; pour dim 100: 4
Xp0=Xp(labels_connues==0,:); %matrice d'apprenissage des images sans pitons
Xp1=Xp(labels_connues==1,:); %matrice d'apprenissage des images avec pitons
z_neg=gaussParzen(Xp_test,Xp0,sig); %densit de vraisemblance des images  la classe "sans pitons"
z_pos=gaussParzen(Xp_test,Xp1,sig); %densit de vraisemblance des images  la classe "avec pitons"

% Phase de validation de la classification
maxi=[z_neg,z_pos];
[m,label]=max(maxi');
label=label-1;
labels_predites=label';

% Fin Timer Parzen
parzen_end=toc(parzen_start);

% Calcul d'erreur
for i=1:N_test
    test(i)=labels(i)-labels_predites(i);
    err = sum(abs(test))/N_test*100;
end

%% Classification des images de test - Mthode de Parzen (recherche de sigma optimale)

% % Phase d'apprentissage et de classification - Calcul de densit de vraissemblance par la methode de Parzen
% sig=[0.025:0.025:1]; %sigma optimale pour dimension 30: 0.4; pour dim 100: 4
% sig=sig';
% Xp0=Xp(labels_connues==0,:); %matrice d'apprenissage des images sans pitons
% Xp1=Xp(labels_connues==1,:); %matrice d'apprenissage des images avec pitons
% 
% err = zeros(1,length(sig)); % Initialisation du vecteur d'erreurs
% 
% for k = 1:length(sig)
%     z_neg=gaussParzen(Xp_test,Xp0,sig(k)); %densit de vraisemblance des images  la classe "sans pitons"
%     z_pos=gaussParzen(Xp_test,Xp1,sig(k)); %densit de vraisemblance des images  la classe "avec pitons"
% 
%     % Phase de validation de la classification
%     maxi=[z_neg,z_pos];
%     [m,label]=max(maxi');
%     label=label-1;
%     labels_predites=label';
% 
%     test = zeros(1, N_test);
%     for i=1:N_test
%         test(i)=labels(i)-labels_predites(i);
%     end
%     err(k) = sum(abs(test))/N_test*100;
% end
% 
% figure
% plot(sig,err,'r-.');
% xlabel('Ecart-type du noyau de Parzen')
% ylabel('Pourcentage d erreur')
% title('Evolution du pourcentage d erreur de la classification des images de test')
% 
% [err_minimale,i]=min(err); %on rcupre l'indice du minimum du vecteur d'erreur
% sig_optimal=sig(i); %on determine le sigma optimale qui donne le pourcenatge d'erreur minimal

%% SVM Gaussien

% Debut Timer SVM
svm_start=tic;

%Phase d'apprentissage
SVMModel = fitcsvm(Xp,labels_connues,"KernelFunction","rbf","OptimizeHyperparameter","auto","KernelScale","auto");

%Phase de classification
[labels_predites,score] = predict(SVMModel,Xp_test);

%  Fin Timer SVM
svm_end=toc(svm_start);

% Phase de validation
for i=1:N_test
    test(i)=labels(i)-labels_predites(i);
    err = sum(abs(test))/N_test*100;
end

%% Rseau de neurones feedforward (les caractristiques HOG ou les donnes ACP)

% Phase d'apprentissage

%Dbut Timer Rseau Apprentissage
Reseau_App_start=tic;

%Bases d'apprentissage et de test
%HOG
base_apprentissage=train_matrix; 
base_test=test_matrix;
nb_caract=M;
%ACP
% base_apprentissage=Xp; 
% base_test=Xp_test;
% nb_caract=dim;

% Cration du rseau
% net1 = feedforwardnet(3); %rseau 1
% net2 = feedforwardnet(6); %rseau 2
% net3 = feedforwardnet([3 6]); %rseau 3
net4 = feedforwardnet([3 6 9]); %rseau 4
% Paramtres 
net.divideParam.trainRatio=0.8;  
net.divideParam.valRatio=0.2;

% Entrainement du rseau
% net1 = train(net1,base_apprentissage',labels_connues'); %entrainement du rseau 1 sur les caractristiques HOG
% net2 = train(net2,train_matrix',labels_connues'); %entrainement du rseau 2 sur les caractristiques HOG
% net3 = train(net3,base_apprentissage',labels_connues'); %entrainement du rseau 3 sur les caractristiques HOG
net4 = train(net4,Xp',labels_connues'); %entrainement du rseau 4 sur les caractristiques HOG

%Fin Timer Rseau Apprentissage
Reseau_App_end=toc(Reseau_App_start);

% %Performance
% labels_predites1 = net1(base_apprentissage');
% labels_predites2 = net2(base_apprentissage');
% labels_predites3 = net3(base_apprentissage');
% labels_predites4 = net4(base_apprentissage');
% 
% for i=1:(Np+Nn)
%     if labels_predites1(i)<0.5
%         labels_predites1(i)=0;
%     else
%         labels_predites1(i)=1;
%     end
% 
%     if labels_predites2(i)<0.5
%         labels_predites2(i)=0;
%     else
%         labels_predites2(i)=1;
%     end
% 
%     if labels_predites3(i)<0.5
%         labels_predites3(i)=0;
%     else
%         labels_predites3(i)=1;
%     end
% 
%     if labels_predites4(i)<0.5
%         labels_predites4(i)=0;
%     else
%         labels_predites4(i)=1;
%     end
% end
% 
% perf1 = perform(net1,labels_connues,labels_predites1');
% perf2 = perform(net2,labels_connues,labels_predites2');
% perf3 = perform(net3,labels_connues,labels_predites3');
% perf4 = perform(net4,labels_connues,labels_predites4');
% 
% %Visualisation des performances
% x=linspace(1,4,4);
% perf=[perf1,perf2,perf3,perf4];
% figure
% plot(x,perf,'ro')
% xlabel('Numro du rseau')
% ylabel('Performance du rseau')
% title('Performances des rseaux de neurones')

% Phase de test

%Dbut Timer Rseau Classification
Reseau_Class_start=tic;

% labels_predites=net2(test_matrix'); %meilleur rseau pour le hog
labels_predites=net4(Xp_test'); %meilleur rseau pour l'acp

%Fin Timer Rseau Classification
Reseau_Class_end=toc(Reseau_Class_start);


for i=1:N_test
    if labels_predites(i)<0.5
        labels_predites(i)=0;
    else
        labels_predites(i)=1;
    end
end

% Phase de validation
for i=1:N_test
    test(i)=labels(i)-labels_predites(i);
    err = sum(abs(test))/N_test*100;
end


%% Classification des images de test - Rseau de neurones convolutif sur les images en couleur et en N&B
close all
clear all
clc

% Dbut Timer - Rseau convolutif 
Reseau_conv_start=tic;

%Chargement des bases de donnes des images
imds = imageDatastore("O:\Documents\2A\Bloc 10\Bureau d'études\dataset\Train","IncludeSubfolders",true,'LabelSource','foldernames');
% imds_test = imageDatastore("O:\Documents\2A\Bloc 10\Bureau d'�tudes\dataset\Test","IncludeSubfolders",true,'LabelSource','foldernames');
imds_test = imageDatastore("O:\Documents\2A\Bloc 10\Bureau d'études\ImNonNormalise","IncludeSubfolders",true,'LabelSource','foldernames');
nb_images=1600;
nb_images_test=50;

%Taille d'une image
image = readimage(imds,1); 
taille = size(image);

%Structures des rseaux
layers1 = [
 imageInputLayer([128 64 3]) %
 convolution2dLayer(3,8,'Padding','same') 
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %carr de 2x2, Stride 
 fullyConnectedLayer(2) %2 = nb de classes
 softmaxLayer
 classificationLayer];

layers2 = [
 imageInputLayer([128 64 3]) %
 convolution2dLayer(3,8,'Padding','same') 
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %carr de 2x2, Stride 

 %Couches ajoutes
 convolution2dLayer(3,16,'Padding','same') %couche ajoute
 batchNormalizationLayer %couche ajoute
 reluLayer %couche ajoute
 maxPooling2dLayer(2,'Stride',2) %couche ajoute

 fullyConnectedLayer(2) %2 = nb de classes
 softmaxLayer
 classificationLayer];

layers3 = [
 imageInputLayer([128 64 3]) %
 convolution2dLayer(3,8,'Padding','same') 
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %carr de 2x2, Stride 

 %Couches ajoutes
 convolution2dLayer(3,16,'Padding','same') %couche ajoute
 batchNormalizationLayer %couche ajoute
 reluLayer %couche ajoute
 maxPooling2dLayer(2,'Stride',2) %couche ajoute
 convolution2dLayer(3,32,'Padding','same') %couche ajoute
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %couche ajoute
 %Fin couches ajoutes

 fullyConnectedLayer(2) %2 = nb de classes
 softmaxLayer
 classificationLayer];

layers4 = [
 imageInputLayer([128 64 3]) %
 convolution2dLayer(3,8,'Padding','same') 
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %carr de 2x2, Stride 

 %Couches ajoutes
 convolution2dLayer(3,16,'Padding','same') %couche ajoute
 batchNormalizationLayer %couche ajoute
 reluLayer %couche ajoute
 maxPooling2dLayer(2,'Stride',2) %couche ajoute
 convolution2dLayer(3,32,'Padding','same') %couche ajoute
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %couche ajoute
 convolution2dLayer(3,64,'Padding','same') %couche ajoute
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %couche ajoute
 %Fin couches ajoutes

 fullyConnectedLayer(2) %2 = nb de classes
 softmaxLayer
 classificationLayer];

layers5 = [
 imageInputLayer([128 64 3]) %
 convolution2dLayer(3,8,'Padding','same') 
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %carr de 2x2, Stride 

 %Couches ajoutes
 convolution2dLayer(3,16,'Padding','same') %couche ajoute
 batchNormalizationLayer %couche ajoute
 reluLayer %couche ajoute
 maxPooling2dLayer(2,'Stride',2) %couche ajoute
 convolution2dLayer(3,32,'Padding','same') %couche ajoute
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %couche ajoute
 convolution2dLayer(3,64,'Padding','same') %couche ajoute
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %couche ajoute
 convolution2dLayer(3,64,'Padding','same') %couche ajoute
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %couche ajoute
 %Fin couches ajoutes

 fullyConnectedLayer(2) %2 = nb de classes
 softmaxLayer
 classificationLayer];

%Prparation des donnes d'apprentissage
[trainData,valData]=splitEachLabel(imds,0.8,0.2);

%Paramtres du rseau
options = trainingOptions('sgdm','InitialLearnRate',0.01, 'MaxEpochs',6,'Shuffle','every-epoch','ValidationData',valData,'ValidationFrequency',10,'Verbose',false,'Plots','training-progress');

%Apprentissage des rseaux
% net1 = trainNetwork(trainData,layers1,options);
% net2 = trainNetwork(trainData,layers2,options);
% net3 = trainNetwork(trainData,layers3,options);
% net4 = trainNetwork(trainData,layers4,options);
net5 = trainNetwork(trainData,layers5,options);

% % Calcul de prcision pour de tous les rseaux et le choix du rseau le
% % plus performant
% 
% labels_predites1 = classify(net1,valData);
% labels_predites2 = classify(net2,valData);
% labels_predites3 = classify(net3,valData);
% labels_predites4 = classify(net4,valData);
% labels_predites5 = classify(net5,valData);
% 
% accuracy1=0;
% accuracy2=0;
% accuracy3=0;
% accuracy4=0;
% accuracy5=0;
% for i=1:length(labels_predites1)
%     if labels_predites1(i)==valData.Labels(i)
%         accuracy1=accuracy1+1;
%     end
%     if labels_predites2(i)==valData.Labels(i)
%         accuracy2=accuracy2+1;
%     end
%     if labels_predites3(i)==valData.Labels(i)
%         accuracy3=accuracy3+1;
%     end
%     if labels_predites4(i)==valData.Labels(i)
%         accuracy4=accuracy4+1;
%     end
%     if labels_predites5(i)==valData.Labels(i)
%         accuracy5=accuracy5+1;
%     end
% end
% accuracy1=accuracy1/length(labels_predites1)*100;
% accuracy2=accuracy2/length(labels_predites1)*100;
% accuracy3=accuracy3/length(labels_predites1)*100;
% accuracy4=accuracy4/length(labels_predites1)*100;
% accuracy5=accuracy5/length(labels_predites1)*100;
% 
% %Visualisation des prcision de chaque rseau
% x=linspace(1,5,5);
% perf=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5];
% figure
% plot(x,perf,'ro')
% xlabel('Numro du rseau')
% ylabel('Prcision globale du rseau (en %)')
% title('Performances des rseaux de neurones')

% Classification
labels_predites = classify(net5,imds_test);

% Fin - Rseau convolutif 
Reseau_conv_end=toc(Reseau_conv_start);

% Phase de validation
accuracy=0;
for i=1:length(labels_predites)
    if labels_predites(i)==imds_test.Labels(i)
        accuracy=accuracy+1;
    end
end
accuracy=accuracy/length(labels_predites)*100; %taux de prcision globale

err=100-accuracy; % taux d'erreur


%% Temps de calcul

% temps_calcul_parzen = HOG_end+HOG_test_end+ACP_end+ACP_test_end+parzen_end;
temps_calcul_SVM = HOG_end+HOG_test_end+ACP_end+ACP_test_end+svm_end;
% temps_calcul_sur_HOG=HOG_end+HOG_test_end+Reseau_App_end+Reseau_Class_end;
% temps_calcul_sur_ACP=HOG_end+HOG_test_end+ACP_end+ACP_test_end+Reseau_App_end+Reseau_Class_end;
% temps_calcul_sur_images=Reseau_conv_end;

%% Critres de performance

TN=0; %vrais ngatifs
FN=0; %faux ngatifs
FP=0; %faux positifs
TP=0; %vrais positifs

% % Calcul pour tous les modeles sauf reseau convolutif
% for i=1:N_test
%     if test(i)==-1
%         FP=FP+1;
%         ind_FP=i;
%     end
%     if test(i)==1
%         FN=FN+1;
%         ind_FN=i;
%     end
%     if test(i)==0
%         if labels(i)==0
%             TN=TN+1;
%         else
%             TP=TP+1;
%         end
%     end
% end

% Calcul pour le reseau convolutif
for i=1:400
    if labels_predites(i)=="pos" && imds_test.Labels(i)=="neg"
        FP=FP+1;
    end
    if labels_predites(i)=="neg" && imds_test.Labels(i)=="pos"
        FN=FN+1;
    end
    if labels_predites(i)=="neg" && imds_test.Labels(i)=="neg"
        TN=TN+1;
    end
    if labels_predites(i)=="pos" && imds_test.Labels(i)=="pos"
        TP=TP+1;
    end
end

%Matrice de confusion
conf=[TN,FN;FP,TP]; 

% Critres
sensitivity=TP/(TP+FN); %Sensibilit
precision = TP/(TP+FP); %Prcision
f_mesure=2*TP/(2*TP+FP+FN); %F-mesure: moyenne harmonique de la prcision et de la sensibilit
specificity=TN/(FP+TN); %Spcificit
accuracy=(TN+TP)/(FP+FN+TP+TN); %Prcision globale

%% Application de l'HOG et de l'ACP sur les images non normalisées

%Debut Timer HOG
HOG_test_start=tic;

%HOG sur les positives
filenames=dir(['ImNonNormalise',filesep,'pos',filesep,'*.png']);
Np_test=length(filenames);
test_matrix_pos=zeros(Np_test,M);
for ifile=1:Np_test
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    test_matrix_pos(ifile,:)=hogfeat';
end

%HOG sur les negatives
filenames=dir(['ImNonNormalise',filesep,'neg',filesep,'*.png']);
Nn_test=length(filenames);
test_matrix_neg=zeros(Nn_test,M);
for ifile=1:Nn_test
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    test_matrix_neg(ifile,:)=hogfeat';
end

test_matrix=[test_matrix_pos; test_matrix_neg]; %matrice de tous les images de test
labels=[ones(Np_test,1);zeros(Nn_test,1)]; %labels des images de test (pour la phase de validation)
N_test=Nn_test+Np_test; %nombre d'images de test

%Fin Timer HOG
HOG_test_end=toc(HOG_test_start);

%Debut Timer ACP
ACP_test_start=tic;

%ACP sur les images de test
Xc_test = test_matrix-1*mean(train_matrix);
Xcn_test = Xc_test*diag(1./std(Xc));
V_test = Xcn_test*U;
Xp_test=V_test(:,1:dim);

%Fin Timer ACP
ACP_test_end=toc(ACP_test_start);