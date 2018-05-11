W1 = vector(:,1:inc);

feature_vector = W1' * images;
feature_test = W1' * images_test;

%% Calculating the nearest neighbour 

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

error = (1 - test_accuracy)*100;

r = 0;
diag_lambda = diag(lambda);

ratio = sum(diag_lambda(1:inc)) / sum(diag(lambda));
