function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
n = size(z ,1);

m = size(z,2);

for i = 1:n
  for j =1:m
    negZ = -z(i,j);
    val1 = exp(negZ);
    val1 = val1 + 1;
    val2 = 1.0/val1;
    g(i,j) = val2;  
  endfor
endfor



% =============================================================

end
