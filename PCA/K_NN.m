function [Knn] = K_NN (ytrain, Index, k)

Ind = Index(:,1:k);
 a = ytrain(Ind);
 Knn = mode(a,2);
end