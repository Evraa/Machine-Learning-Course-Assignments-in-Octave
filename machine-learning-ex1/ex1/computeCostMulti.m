function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

n = size(X,2);
avg = 1.0 / (2*m);
sum = 0;

for i = 1:m
  xes = 0;
  for k = 1:n
    xes = xes + ( X(i,k) * theta(k) );
  endfor
  
  
  varience = xes - (y(i));
  
  varience = varience * varience;
  sum = sum + varience;
endfor
J = avg * sum;



% =========================================================================

end
