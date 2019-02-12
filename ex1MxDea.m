%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
clear ; close all; clc

%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m 
%fprintf('Running warmUpExercise ... \n');
%fprintf('5x5 Identity Matrix: \n');
%warmUpExercise()

%fprintf('Program paused. Press enter to continue.\n');
%pause;


%% ======================= Part 2: Plotting =======================
%fprintf('Plotting Data ...\n')
data = load('deviation.txt');
EW = data(:, 1); %X is the population in 1000s
X = log10(EW);

y = data(:, 2); % y is the revenue

m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m


%plotData(X, y);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% =================== Part 3: Gradient descent ===================
%fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), X]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% compute and display initial cost
%computeCost(X, y, theta)

% run gradient descent
[theta0,error0, th00, th10] = gradientDescent(X, y, theta, alpha, iterations);
[theta,error, th0, th1] = gradientDescentRegularization(X, y, theta, alpha, iterations, 0); %lambda = 0
[theta1,error1, th01, th11] = gradientDescentRegularization(X, y, theta, alpha, iterations, 0.01); %lambda = 0.01
[theta2,error2, th02, th12] = gradientDescentRegularization(X, y, theta, alpha, iterations, 0.25); %lambda = 0.1
[theta3,error3, th03, th13] = gradientDescentRegularization(X, y, theta, alpha, iterations, 0.5); %lambda = 0.2
[theta4,error4, th04, th14] = gradientDescentRegularization(X, y, theta, alpha, iterations, 1); %lambda = 0.5
[theta5,error5, th05, th15] = gradientDescentRegularization(X, y, theta, alpha, iterations, 2); %lambda = 1


% print theta to screen
fprintf('Theta found by gradient descent without regularization: ');
fprintf('%f %f \n', theta0(1), theta0(2));
fprintf('Theta found by gradient descent with regularization: and lambda = 0 ');
fprintf('%f %f \n', theta(1), theta(2));
fprintf('Theta found by gradient descent with regularization: and lambda = 0.01 ');
fprintf('%f %f \n', theta1(1), theta1(2));
fprintf('Theta found by gradient descent with regularization: and lambda = 0.1 ');
fprintf('%f %f \n', theta2(1), theta2(2));
fprintf('Theta found by gradient descent with regularization: and lambda = 0.2 ');
fprintf('%f %f \n', theta3(1), theta3(2));
fprintf('Theta found by gradient descent with regularization: and lambda = 0.5');
fprintf('%f %f \n', theta4(1), theta4(2));
fprintf('Theta found by gradient descent with regularization: and lambda = 1 ');
fprintf('%f %f \n', theta5(1), theta5(2));
%ooooooooooooooooooooooooooooooooooooooooooooooooooooooo Da plotzzzz

% Plot the linear fit
% keep previous plot visible
plot(X(:,2), y, 'ok', "markersize", 3)
hold on;
plot(X(:,2), X*theta, '-',"linewidth", 2)

legend('Training data', 'Linear regression')
xlabel('log10(n)');
ylabel('dist');
hold off; % don't overlay any more plots on this figure

figure;
plot(error,"linewidth", 2);
ylabel('value of RMSE');
xlabel('Iteration');
hold off;

figure;
plot(th00,"linewidth", 2);
ylabel('value of Theta_0');
xlabel('Iteration');
hold off;

figure;
plot(th10,"linewidth", 2);
ylabel('value of Theta_1');
xlabel('Iteration');
hold off;



% oo--oo--oo--oo--oo--oo--oo--oo--oo--oo--oo--oo--oo--oo--oo--oo--ooendplotz


%{
""" commenting everything below this line 


% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20), 'LineWidth', 2)
legend('Curve for the cost function');
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);


"""

%}
