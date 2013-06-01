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

n = length(theta);

for i = 1:m
	h_theta_of_x_of_i = sigmoid(X(i, :) * theta);
	J = J + ((-y(i) * log(h_theta_of_x_of_i)) - ((1 - y(i)) * log(1 - h_theta_of_x_of_i)));
endfor;
J = J / m;

J = J + ((lambda * sum(theta(2:end) .^ 2))/(2*m));

for j = 1:n
	for i = 1:m
		h_theta_of_x_of_i = sigmoid(X(i, :) * theta);
		grad(j) = grad(j) + ((h_theta_of_x_of_i - y(i)) * X(i, j));
	endfor;
	if(j==1)
		grad(j) = grad(j) / m;
	else
		grad(j) = (grad(j) + (lambda*theta(j)))/m;
	endif;
endfor;

% =============================================================

end
