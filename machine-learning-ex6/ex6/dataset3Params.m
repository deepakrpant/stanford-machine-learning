function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%generate various possible combinations of C and sigma
a = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
b = combvec(a,a); %all possible combos of the above variables
combos = size(b,2); %all possible combos
C = b(1,:); sigma = b(2,:);

%loop through and evaluate different values of C and sigma
for i = 1:combos
    model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(i)));
    predictions = svmPredict(model, Xval);
    error(i) = mean(double(predictions ~= yval));
end

%find the vlaues of C and sigma with least errors
[minerror,indx] = min(error);
C = C(indx) 
sigma = sigma(indx)






% =========================================================================

end
