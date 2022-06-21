% Parameter Initialization 
clc; clear; 

format short
x0 = [50, 50];                  % Initial guess
tol = 1e-3;                   % Maximum tolerance
i = 1;                        % Iteration counter 
S = 0;                        % Initial direction 
X_solutions = [];

% OBjective Function 
syms x1 x2
% f = @(x1, x2) -(x2 + 47).*sin(sqrt( x2+x1./2+47 )) - x1*sin(sqrt( x1-(x2+47) ));  
f = @(x1, x2) x1 - x2 + 2*x1^2 + 2*x1*x2 + x2^2;
% fx = inline(f);

% Gradient
g = gradient(f, [x1, x2]);    % Obtains the gradient of the symblic function
gd = matlabFunction(g);       % Converts the gradient into an anonymus function with two inputs
gradx = @(x) gd(x(1), x(2));  % Converts the gradient into a single input anonymus function 
g_pre = -vpa(subs(g, [x1, x2], [x0(1), x0(2)]));           % Initializes the n-1th gradient

% Hessian 
hess = hessian(f, [x1, x2]);     % Obtains the hessian of the symbolic function
% hd = matlabFunction(h);     % Converts the hessian into an anonymus function with two inputs 
% h = vpa(subs(hess, 1)); 

% Main Algorithm 


while norm(vpa(subs(g, [x1, x2], [x0(1), x0(2)]))) > tol && i < 4     % main CG algorithm

    X_solutions = [X_solutions; x0]; 
    h = vpa(subs(hess, [x1, x2], [x0(1), x0(2)]));
    Gi = -vpa(subs(g, [x1, x2], [x0(1), x0(2)]));
    beta = norm(Gi).^2/norm(g_pre).^2; 
    S = Gi + beta.*S; 
    lambda = Gi'*Gi./(S'*h*S); 
    Xnew = x0 + lambda.*S'; 

    % the following commands create a mesh of the function plot in a 3D
    % plane to allow for plotting the descent trajectory 

    P0 = [x0(1), x0(2), f(x0(1), x0(2))] ;   
    P1 = [Xnew(1), Xnew(2), f(Xnew(1), Xnew(2))] ;
    X = [P0(:,1) P1(:,1)] ;
    Y = [P0(:,2) P1(:,2)] ;
    Z = [P0(:,3) P1(:,3)] ;
    plot3(X',Y',Z')
    hold on
    plot3(X',Y',Z','r')

    [X,Y] = meshgrid(-100:1:100);
%     Z = -(Y + 47).*sin(sqrt(abs( Y+X./2+47 ))) - X.*sin(sqrt(abs( X-(Y+47))));
    Z = X - Y + 2.*X.^2 + 2.*X.*Y + Y.^2;
    s = surface(X,Y,Z,'FaceAlpha',0.5);  
%     hold on
%     z = get(s,'ZData');
%     set(h,'ZData',Z-10)  
%     plot(1:50,'color','r','linewidth',3)

    

    x0 = Xnew; 
    g_pre = Gi; 
    i = i+1; 
end 

x0;
X_solutions;   % the value of the minimizer "x" after each iteration stored in a matrix
