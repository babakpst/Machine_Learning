function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


%theta2 = theta;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %theta
    %fprintf("y  \n"),y(1,:)
    %fprintf("x \n"),X(1,:)
    %fprintf("res \n"),(y(1)-X(1,:)*theta)
    
    
    %k = 1:m;
    %t1 = sum( (theta(1) + theta(2) .* X(k,2)) - y(k)); % Un-Vectorized
    %t2 = sum(((theta(1) + theta(2) .* X(k,2)) - y(k)) .* X(k,2)); % Un-Vectorized
    %theta(1) = theta(1) - (alpha/m) * (t1);
    %theta(2) = theta(2) - (alpha/m) * (t2);

    %X'*((X*theta) - y)
    %t1
    %t2    
    theta = theta - (alpha/m) * ( X'*((X*theta) - y)   );
    %theta
    %theta2
    %pause;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
