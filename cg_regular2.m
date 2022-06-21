clc; clear; 
x = 0.9;                               % define initial x
max = 5;                               % define maximum iteration                                                      

[x, i, g] = cg(x, max);                

function [x, i, g] = cg(x, maxit)      % run CG and assigns the final minimum point to x, number of iterations to i, and gradient value to g
fplot(@(x) 10 + x^2 - 10*cos(2*pi*x))  % plots the main fucntion 
hold on

tol = 1e-14;                           % define the tolerance 
[f0, g0] = fun(x);                     % runs the objective function and assigns the initial value with x0 to f0, and initial value of the gradient to g0
d0 = g0;                               % initializes the residue
s = -g0;                               % initializes the direction of the first descent trajectory 

for i = 1:maxit                        % main CG algorithm 
    [f,g,h] = fun(x); 
    d = -g;
    lambda = (d*s)/((s*h*s));          % exact descent step length  
    x_new = x + lambda*s;
    p = g;
    g = g0 + lambda*h*s;               % conjugate descent direction wrt to previous direction 
    
    a = fun(x);
    fx = a(1); 
    b = fun(x_new);
    fx_new = b(1);
    plot([x, x_new], [fx, fx_new], "r-")

        if norm(g) < tol  
        fprintf('Convergence reached! x = %g \n', x_new);
        break;   
        end 
   
        if i == maxit
        disp('Maximum number of iterations reached');
        end 
    beta = g^2/g0^2;
    s = -g + beta*s;   


    
    x = x_new;
    d0 = d;
    g0 = g; 
end 
% fprintf('Number of iterations i = %d \n', i);
end 


function [f, g, h] = fun(a)
syms x  
fx = @(x) 10 + x^2 - 10*cos(2*pi*x);
f = fx(a);
d = diff(fx, x); 
g = vpa(subs(d, x, a)); 
der2 = diff(fx, x, 2); 
% h = vpa(subs(der2, x, a)); 
hess = hessian(fx, [x]);
h = vpa(subs(hess, a));
end 
