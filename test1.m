%% Application sur tous les images

filt=[-1 0 1];
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
Np=length(filenames);
train_matrix_neg=zeros(Np,M);
for ifile=1:Np
    I=double(imread([filenames(ifile).folder,filesep,filenames(ifile).name]))/255;
    im=rgb2gray(I);
    hogfeat=hogfeatures(im,filt,T,N);
    train_matrix_neg(ifile,:)=hogfeat';
end
