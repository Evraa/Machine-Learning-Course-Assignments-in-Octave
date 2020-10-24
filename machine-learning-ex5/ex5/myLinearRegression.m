function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
#Cost function for linear regression regularized:
  #theta transpose X
  #subtract y
  #power them
  #sum all
  #add reg.

hOfTheta = X * theta;
minusY = hOfTheta - y;
powered = minusY .* minusY;
summed = sum(powered);
avg = 1./(2.*m);
J = summed;

thetaPowered = theta .* theta;
thetaSummed = sum(thetaPowered) - thetaPowered(1);
regTerm = thetaSummed .* lambda;

J = J + regTerm;
J = J .* avg;

#Gradient Descent:

multiply = minusY' * X;
multiply = multiply ./m;
avg = lambda ./m;

thetaAvg = theta .* avg;
grad = multiply + thetaAvg';
#Dont forget, we dont regularize theta(1)
grad(1) = grad(1) - thetaAvg(1);
% =========================================================================


end
