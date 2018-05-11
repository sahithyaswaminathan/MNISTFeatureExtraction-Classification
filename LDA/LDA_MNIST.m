%% Description
% 1. Dimensionality of the feature is reduced to 2, 3 and 9 for which train and test data accuracy is computed
% 2. Data is projected on 2D and 3D for which data visualization is incorporated
% 3. Test the maximum dimensionality for which LDA can be projected (Number of Classes - 1)
%======================================================================

clc;
clear all;

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

images_test = loadMNISTImages('t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');

N_train = 60000;
N_test = 10000;

%% Grouping the class images

[B, im_ind] = sort(labels, 'ascend');

for i = 1:N_train
    ind = im_ind(i,1);
    images_train(:,i) = images(:,ind);
end

%% Counting the number of images in each class

edges = unique(B);
counts = histc(B(:), edges);

%% Mean of each class
 
%mu1 = mean(images_train(:,1:counts(1)));

a = 0;
for i = 1:10
    start(i) = a+1;
    finish(i) = a + counts(i);  % Finding the start and end of a class
    a = finish(i);
end

for i = 1:10
    mu(:,i) = mean(images_train(:,start(i):finish(i))')';
end

%% Scatter matrix

mu1 = mean(images_train')';

Sw = zeros (784,784);
Sb = zeros (784,784);

for i = 1:10
    Pi = counts(i)/N_train;
    X = images_train(:,start(i):finish(i)) - mu(:,i)*ones(1,counts(i));
    Si = (1/counts(i)) * X *X';
    
    %Within scatter matrix
    Sw = Sw + Si * Pi;
    
    %Between scatter matrix
    Sb = Sb + Pi * (mu(:,i)- mu1) * (mu(:,i)-mu1)';
end
    
%% Singular value decomposition

St = pinv(Sw)* Sb;

[U, V, D] = svd(St);

W = U(:,1:9); %[2,3,9]

feature_vector = W' * images;

feature_test = W' * images_test;

%% Calculating the nearest neighbour 

dis = pdist2(feature_test',feature_vector','Euclidean');

[K, Index] = sort(dis,2); %Index of the specific sorted element%

k = 10;

Knn = K_NN(labels,Index, k);

A1(:,1) = Knn;

test_count = 0;

for i = 1:N_test
    if A1(i,1) ~= labels_test(i,1)
        test_count = test_count + 1;
    end
end

test_accuracy = (N_test - test_count)/N_test;

%% Plotting 2D and 3D visualization

%Reducing dimensionality to 2-D

W1 = U(:,1:2);
Y = W1' * images;

for i = 1:10
    num = (labels == i-1);
    f1 = Y(1,num);
    f2 = Y(2,num);
    
    subplot(2,5,i);
    scatter(f1',f2','m');
    title(['Image distribution', num2str(i)]);
end

%Reducing the dimensionality to 3 - 3D visualization

W2 = U(:,1:3);
Y1 = W2' * images;

for i = 1:10
    num = (labels == i-1);
    a1 = Y1(1,num);
    a2 = Y1(2,num);
    a3 = Y1(3,num);
    subplot(2,5,i);
    scatter3(a1',a2',a3','m');
    title(['Image distribution(3D)', num2str(i)]);
end

%% LDA fail condition
%Check
diag_vec = diag(V);
disp('Displaying the largest 10 eigenvalues of matrix W');
for value = 1:10
    disp(['Eigenvalue ', num2str(value), ': ', num2str(diag_vec(value))]);
end


