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

% Add ones to the X data matrix
X = [ones(m, 1) X];

%calculate the hidden layer parameters
a1 = X;
z2  = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];

%Calculate final layer paramters
z3 = a2*Theta2';
a3 =  sigmoid(z3);
htheta = a3;

%reshape y into a 5000X10 matrix
K = size(htheta,2);

ynew = zeros(m,K);

for im = 1:m
    ynew(im,y(im,1)) = 1;
end

y = ynew;
 
% calculate the cost
for im = 1:m
    sum1(im,1) = sum(-y(im,:).*log(htheta(im,:))-(1-y(im,:)).*log(1-(htheta(im,:))));
end

J = 1/m*sum(sum1);

%Add the regularization term
n1 = size(Theta1,2); %number of features
n2 = size(Theta2,2); 

Theta1n = Theta1(:,2:n1);
Theta1unr = [Theta1n(:)];
sum2 = Theta1unr'*Theta1unr;

Theta2n = Theta2(:,2:n2); %remive the first entry
Theta2unr = [Theta2n(:)];
sum3 = Theta2unr'*Theta2unr;


reglr = (lambda/(2*m))*(sum2+sum3);
J = J+reglr;


%back prop
n = size(X,2); untsp1 = size(Theta2,2);
Delta1 = zeros(untsp1-1,n); %for 2 layers
Delta2 = zeros(K,untsp1); %for 2 layers
for im = 1:m
%     
%     a1 = X(im,:);
% z2  = a1*Theta1';
% a2 = sigmoid(z2);
% a2 = [ones(m,1) a2];
% 
% %Calculate final layer paramters
% z3 = a2*Theta2';
% a3 =  sigmoid(z3);
% htheta = a3;

    
    del3 = a3(im,:)-y(im,:);
    del3 = del3';
    
%     del2 = a2(im,:).*(1-a2(im,:));
%     del2 = del2';
    
    del2 = Theta2n'*del3.*sigmoidGradient(z2(im,:))';
    
    Delta1 = Delta1+del2*a1(im,:); 
    Delta2 = Delta2+del3*a2(im,:); 
    
end
Theta1_grad = Delta1/m; Theta2_grad = Delta2/m;

%regularize
Theta1_grad(:,2:n) = Theta1_grad(:,2:n)+lambda/m*Theta1(:,2:n);
Theta2_grad(:,2:untsp1) = Theta2_grad(:,2:untsp1)+lambda/m*Theta2(:,2:untsp1);







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
