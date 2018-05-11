clc;
clear all;

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

images_test = loadMNISTImages('t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');

N_train = 60000;
N_test = 10000;

%% Calculating Mean for all the images

mu = mean(images')';

%% Calculating Covariance for all the images


X = images - mu*ones(1,60000); % Subracting the mean to translate the coordinate system to the location of the mean

X_t = transpose(X);

S = X * X_t * (1/N_train);

%% Calaculating Eigen values 

[vector,lambda] = svd(S); %The lambda values are stored in the increasing order. Hence for maximum variance, last n-dimensioned values must be taken



%% Selecting first 40/80/200/154 eigen vectors/eigen faces

eigen_face = 40; % [40, 80, 200, 154] 

[r,c] = size(images);

W = vector(:,1:eigen_face);

feature_vector = W' * images; %Feature Vector of the train images

%% Plot of Eigen Vectors

for i = 1:784
subplot(28,28,i)
image = reshape(vector(:,i),[28,28]);
imagesc(image);colormap(gray);axis off;
end

%% Calculating eigen faces for test data

X_test = images_test - mu*ones(1,10000);

feature_test = W' * images_test; %Feature Vector of the test images

%% Calculating the nearest neighbour 

%KNN approach
dis = pdist2(feature_test',feature_vector','Euclidean');

[K, Index] = sort(dis,2); %Index of the specific sorted element%

k = 10;

Knn = K_NN(labels,Index, k);
A(:,1) = Knn;
  
test_count = 0;

for i = 1:N_test
    if A(i,1) ~= labels_test(i,1)
        test_count = test_count + 1;
    end
end

test_accuracy = (N_test - test_count)/N_test;

Accuracy = test_accuracy;


%%Total Energy conservation Calculation

ratio = 0;
diag_lambda = diag(lambda);

ratio = sum(diag_lambda(1:eigen_face)) / sum(diag(lambda));


%% Error calculation

error = (1-Accuracy) *100;

%% Plotting 2D and 3D visualization

%Reducing dimensionality to 2-D

W1 = vector(:,1:2);
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

W2 = vector(:,1:3);
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
