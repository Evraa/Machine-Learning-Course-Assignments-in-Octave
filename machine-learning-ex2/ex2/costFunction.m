function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
#compute the cost
    #hoftheta..(theta * X) --> sigmoid
    #multiply and evaluate the sum
    #get J(theta)
    
n = size(theta,1);

for i = 1:m
  thetaMulX = 0;
  for k = 1:n
    mul = X(i,k) .* theta(k);
    thetaMulX = thetaMulX + mul;
  endfor
  hOfTheta = sigmoid(thetaMulX);
  
  val1 = log(hOfTheta);
  val2 = log(1-hOfTheta);
  
  val1 = val1 .* (-y(i));
  val2 = val2 .* -(1 - y(i));
  
  J = J + (val1 + val2);
endfor
J = J / m;


#Compute the gradient for each theta

for i = 1:n
  for k = 1:m
    matricesMult = X(k,:) * theta;
    hOfTheta = sigmoid(matricesMult);
    val1 = hOfTheta - y(k);
    val2 = val1 .* X(k,i);
    grad(i) = grad(i) + val2;
  endfor
  grad(i) = grad(i) ./ m;
endfor






% =============================================================

end
