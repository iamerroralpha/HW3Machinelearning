function [theta,error,th0,th1] = gradientDescentRegularization(X, y, theta, alpha, num_iters, lambda)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
               %J_history = zeros(num_iters, 1);
error =zeros(num_iters, 1);
th0 = error;
th1 = error;
err=0;
theta(1) = 0;
theta(2) = 0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %fprintf('Cost: %f\n', computeCost(X, y, theta));
    temp1 = 0; temp2 = 0;
    for i = 1:m
        h = theta(1) + theta(2) * X(i,2); %the prediction for that particular time
        err=y(i)-(theta(1) + theta(2) * X(i,2));    %errors added by me
        error(iter)+=(err*err/m);                   %adding the errors and calculating RMSE
        temp1 +=h - y(i);
        temp2 +=(h - y(i)) * X(i,2);
    end
    
    temp1 = temp1 * alpha / m;
    temp2 = temp2 * alpha / m;
    
    theta(1) = theta(1) - temp1;
    th0(iter)=theta(1);
    theta(2) = theta(2)* (1 - alpha*lambda/m) - temp2;
    th1(iter)=theta(2);
    error(iter) = error(iter)^(1/2);
    % ============================================================

    % Save the cost J in every iteration    
    % J_history(iter) = computeCost(X, y, theta);

end

end
