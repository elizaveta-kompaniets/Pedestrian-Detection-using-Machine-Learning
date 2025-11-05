function[z]=gaussParzen(X,Apprk,sig)

%X: ensemble de données
%Apprk: ensemble de données d'apprentissage
%sig: ecart-type du noyau de Parzen

d = size(Apprk,2);
sigma = (sig^2)*eye(d); %variance
N=size(X,1);
Nk = size(Apprk,1);
z = zeros(N,1);
for i = 1:Nk
    z = z+mvnpdf(X,Apprk(i,:),sigma); %Apprk(i,:) = moyenne
end
z=z/Nk; %normalisation