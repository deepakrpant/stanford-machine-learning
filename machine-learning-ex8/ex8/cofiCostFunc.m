function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%cost function
sumn = (X*Theta'-Y).^2;
temp = sumn.*R;
J = (1/2)*sum(temp(:));

%gradient vectorized
for i = 1:num_movies
    Xtemp = X(i,:)'; Ytemp = Y(i,:)'; Rtemp = R(i,:)';
    temp1 = (Theta*Xtemp).*Rtemp;
    temp1 = temp1-Ytemp;
    temp2 = temp1'; 
    X_grad(i,:) = temp2*Theta;
end

for j = 1:num_users
    Thetatemp = Theta(j,:)'; Ytemp = Y(:,j); Rtemp = R(:,j);
    temp1 = (X*Thetatemp).*Rtemp;
    temp1 = temp1-Ytemp;
    temp2 = temp1'; 
    Theta_grad(j,:) = temp2*X;
end

%gradient unvectorized is as follows. Answers matchs
% for i = 1:num_movies
%     sum1 = 0;
%     for j = 1:num_users
%         if R(i,j) == 1
%             sum1 = sum1+(Theta(j,:)*X(i,:)'-Y(i,j))*Theta(j,:)';
%         end
%     end
%     X_grad(i,:) = sum1';
% end
% 
% for j = 1:num_users
%     sum1 = 0;
%     for i = 1:num_movies
%         if R(i,j) == 1
%             sum1 = sum1+(Theta(j,:)*X(i,:)'-Y(i,j))*X(i,:)';
%         end
%     end
%     Theta_grad(j,:) = sum1';
% end



%regularize the cost function
sum1 = (lambda/2)*sum(sum(Theta.*Theta));
sum2 = (lambda/2)*sum(sum(X.*X));
J = J+sum1+sum2;

%regularize the gradient
for i = 1:num_movies
 X_grad(i,:) = X_grad(i,:)+ lambda*X(i,:); 
end

for j = 1:num_users
 Theta_grad(j,:) = Theta_grad(j,:)+ lambda*Theta(j,:); 
end









% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
