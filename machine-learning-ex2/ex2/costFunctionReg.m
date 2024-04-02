function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
n = length(theta)-1;

h = sigmoid(X*theta);
temp1 = (-y.*log(h)-(1-y).*log(1-h));
thetaSel = theta(2:n+1,1);
temp2 = thetaSel.^2;
J = 1/m*sum(temp1)+lambda/(2*m)*sum(temp2);

for j = 1:n+1
    if j == 1
        grad(j,1) = 1/m*(X(:,j)'*(h-y));
    else
         grad(j,1) = (1/m*(X(:,j)'*(h-y)))+lambda/(m)*theta(j);
    end
end





% =============================================================

end
