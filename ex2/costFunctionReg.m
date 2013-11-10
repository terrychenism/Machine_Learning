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

for i=1:m
     % X(i,:)  取 矩阵的第i行， 然后 theta 是一个只有一列的矩阵
  % 之前写 theta' * X(i,:)'  对两个矩阵都先transpose (反置）后在相乘
  % 直接写X(i,:) ×theta 应该也是一样的结果，不用两个都反置了。
     %J = J + ( -y(i) * log(sigmoid( theta' * X(i,:)')))  - (1 - y(i) )* log(1- sigmoid(theta' * X(i,:)'));
     J = J + ( -y(i) * log(sigmoid( X(i,:) *  theta )))  - (1 - y(i) )* log(1- sigmoid(  X(i,:) * theta ));
  
end

J = J/m;   

%加上规格化的削弱分子
% 注意j从 1 开始，所以下面这里写j =2 要跳过 theta(0)
for j=2:size(theta)
     J = J + (lambda * (theta(j)^2)/(2 *m));
end
  
 
 
for j=1:size(theta)
 for i=1:m
       grad(j) =  grad(j)  +  (sigmoid(   X(i,:) * theta)  - y(i) ) * X(i,j);
 end  
 
 grad(j) =  grad(j)/m;
end

%加上规格化的削弱分子
% 注意j从 1 开始，所以下面这里写j =2 要跳过 theta(0)
for j=2:size(theta)
    grad(j) =  grad(j) + (lambda * theta(j))/m;
end



% =============================================================

end
