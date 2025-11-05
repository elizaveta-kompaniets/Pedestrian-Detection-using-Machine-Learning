# Pedestrian Detection using Machine Learning
 
 🧠 Project Description
 
This academic project, conducted at the École Nationale Supérieure d'Électricité et de Mécanique (Université de Lorraine), focuses on developing a pedestrian detection method from still images, with potential integration into an embedded vision system for autonomous vehicles. The project belongs to the academic module B8-10 — Artificial Intelligence, Data Analysis and Learning Methods at ENSEM.

The work was carried out in MATLAB using the Dalal & Triggs (2005) pedestrian dataset, which contains thousands of positive (with pedestrians) and negative (without pedestrians) images. The normalized subset of 64×128-pixel positive samples, each centered on a pedestrian, enables effective learning across diverse silhouettes. The unnormalized subset was also used to assess model generalization on real, unconstrained images.

Feature extraction was performed using the Histogram of Oriented Gradients (HOG) method, followed by Principal Component Analysis (PCA) to reduce dimensionality and enhance computational efficiency.

Multiple classification models were then trained and compared for the pedestrian/non-pedestrian task, including:
- Probabilistic modeling (KDE method)
- Support Vector Machines (linear and nonlinear)
- Feedforward and Convolutional Neural Networks
  
Final evaluations demonstrated that Gaussian-kernel SVM and CNN models achieved the best accuracy (up to 99%) on the dataset.
