function [J grad] = nnCostFunction(nn_params,input_layer_size, hidden_layer_size,num_labels,X, y, lambda)
  
  
  
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%




% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

            
% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];
n = size(X, 2);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X_org = X;
Theta1_org = Theta1;
Theta2_org = Theta2;
#Reconstructing the y vector
y_matrix = eye(num_labels)(y,:);
yT = y_matrix;
y = 0;
y = yT;
y;


X_org = X;
y_org = y;
#It's not implemented based on any number of layers
a1 = X * Theta1';
a1 = sigmoid (a1);
a1 = [ones(m,1) a1];

a2 = a1 * Theta2';
a2 = sigmoid (a2);
a2T = a2';
a2 = 0;
a2 = a2T;

hOfTheta =(a2);
#hOfTheta is K x m
#and y is m x K
val1 = log(hOfTheta);
val1T = val1';
val1 = 0;
val1 = val1T;

val2 = log(1 - hOfTheta);
val2T = val2';
val2 = 0;
val2 = val2T;

cost = 0; 

for i = 1:m
   cost(i) = sum( (y(i,:) * val1(i,:)') + ( (1 - y(i,:)) * val2(i,:)') );
endfor
J = sum(cost);
J = J * (-1/m);

#Regulazied term
#Power up all the theta terms
Theta1_power = (Theta1) .**2;
#Sum them by rows at first
sum1 = sum(Theta1_power);
#Zero/Ground the first column, that's responsible for the bias term
sum1(:,1) = 0;
#Sum by the column that time
sum11 = sum(sum1);

Theta2_power = (Theta2) .**2;
sum2 = sum(Theta2_power);
sum2(:,1) = 0;
sum22 = sum(sum2);

#Add them and finish your equation
reg = sum11 + sum22;
reg = reg .* (lambda ./ (2.* m));
#Regulaized cost function.
J = J + reg;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
#Backpropagation algorithm

hiddenTheta = Theta1_org; #4 x 3
outputTheta = Theta2_org;

z =  X_org * hiddenTheta'; #16 x 4
hiddenLayer = [ones(m,1) sigmoid(z)]; # 16 x 5

probs = sigmoid(hiddenLayer * outputTheta');

sigma3 = probs - y_org; #m x K
#sigmoidGradient(hiddenLayer);#m x 5
#sigma3 * outputTheta; #m x 5

#size(sigma3); #m x K
#size (outputTheta); #K x 5

#This is the missing solution, scalar multplication with the sigmoid gradient
sigma2 =( sigma3 * outputTheta ).* [ones(size(z,1),1) sigmoidGradient(z)];
sigma2 = sigma2(:, 2:end);

hiddenDelta = sigma2' * X_org ;# 4 x 3
outputDelta = sigma3' * hiddenLayer; #4 x 5 Tmam

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad = Theta1_grad + (1/m) * hiddenDelta;
Theta2_grad = Theta2_grad + (1/m) * outputDelta;

% REGULARIZATION OF THE GRADIENT

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1_org(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2_org(:,2:end));
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
