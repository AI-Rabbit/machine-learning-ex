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
Theta1_grad = zeros(size(Theta1));  % (25, 401)
Theta2_grad = zeros(size(Theta2));  % (10, 26)
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% feed forward
a1 = [ones(m,1) X];

z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];

z3 = a2 * Theta2';
h = sigmoid(z3);

% expand y
y_raw =y;
y = zeros(m, num_labels);  % max(y) stands for K
for i = 1:m
    y(i, y_raw(i,1)) = 1;
end

% cost function
inner1 = - y .* log(h) - (1-y) .* log(1 - h);

reg_t1 = (lambda / (2 * m)) * sum((Theta1(:,2:end).^2), 'all');
reg_t2 = (lambda / (2 * m)) * sum((Theta2(:,2:end).^2), 'all');

J = 1 / m * sum(inner1, 'all') + reg_t1 + reg_t2;



% -------------------------------------------------------------
for i = 1:m
    a1i = a1(i,:)'; % (401, 1)

    z2i = [1 z2(i,:)]'; % (26, 1)
    a2i = a2(i,:)'; % (26, 1)

    hi = h(i,:)'; % (10, 1)
    yi = y(i,:)'; % (10, 1)
    
    d3i = hi-yi; % (10, 1)
    d2i = (Theta2'* d3i) .* sigmoidGradient(z2i); % (26, 1)
    Theta2_grad = Theta2_grad + d3i * a2i'; %10, 26
    Theta1_grad = Theta1_grad + d2i(2:end,1) * a1i'; %25, 401
end

Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;


% Regularized neural networks
reg_term_d1 = (lambda / m) * Theta1;
reg_term_d1(:,1) = 0;
Theta1_grad = Theta1_grad + reg_term_d1;

reg_term_d2 = (lambda / m) * Theta2;
reg_term_d2(:,1) = 0;
Theta2_grad = Theta2_grad + reg_term_d2;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
