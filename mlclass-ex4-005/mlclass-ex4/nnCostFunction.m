function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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

a2 = sigmoid([ones(m, 1) X] * Theta1');
a3 = sigmoid([ones(m, 1) a2] * Theta2');

%h_ThetaX = sigmoid([ones(m, 1) sigmoid([ones(m, 1) X] ...
%        * Theta1')] * Theta2');
h_ThetaX = a3;
         
%% convert y to y_vec of binary variables         
%# initialize y_vec
y_vec = zeros(m, num_labels);
%# create a linear index from {row,y}
idx = sub2ind(size(y_vec), 1:m, y');
%# set the proper elements of y_vec to 1
y_vec(idx) = 1;

J = -1 / m * sum((sum(log (h_ThetaX) .* y_vec) + ...
     sum((1 - y_vec) .* log (1 - h_ThetaX)))) + ... 
     lambda / 2 / m * ... 
     (sum(sum(Theta1(:, 2:size(Theta1, 2)) .^ 2)) + ... 
     sum(sum(Theta2(:, 2:size(Theta2, 2)) .^ 2)));


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

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3); 

delta_3 = a3 - y_vec; 
delta_2 = (delta_3 * Theta2);
delta_2 = delta_2(:, 2:size(delta_2, 2)) .* sigmoidGradient(z2);

Theta2_grad = (delta_3' * a2) / m;
Theta1_grad = (delta_2' * a1) / m; 

Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:, 2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
