function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


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

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
mean_list = 0;
mean_list2 = zeros(8 ,2);
mean_val = 0;
#pairing and counting the least
for i =1:8
  C = C_list(i);
  for j = 1:8
    sigma = sigma_list(j) ;  
   
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    pred = svmPredict(model, Xval);
    meanVal = mean(double(pred ~= yval));
    
    mean_list(j) = meanVal;
  endfor
 [val, ival] = min (mean_list);
  mean_list2(i,:) = [i , ival];
  mean_val(i) = [val];
endfor
[minimumValue, index] = min(mean_val);

sigma_min_index = mean_list2(index,2);
c_min_index = index;

C = C_list(c_min_index);
sigma = sigma_list(sigma_min_index);

% =========================================================================

end
