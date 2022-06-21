clear; clc; 
x = 9.25;
maxit = 100;
[x, i, h, beta, s] = cg(x, maxit);

X = 1:3; 
% Defines a range in we we want to find the global minimum
minima = [];
fx = [];

tic
for k = 1:(length(X)-1)
    % This loop finds all the local minima in the given range. Each minimum
    % point is obtained using gradient descent with strong wolfe conditions
    if abs(cg(X(k+1), maxit) - cg(X(k), maxit)) > 1e-1
        minima = [cg(X(k), maxit); minima];
    end 
end 

k = 1; 

for k = 1:length(minima)
    % This loop takes all the local minima and creates an array containing 
    % each points' function value( e.g. [(x1, y1), (x2, y2),...]').
    f = fun(minima(k));
    fx = [f; fx];
end

[m, I] = min(fx); 
final_x = minima(I); 
error = norm(final_x - 0)
toc

fprintf('The final value of x is: %g \n', final_x);
fprintf('All local minima within this range: \n')
fprintf('%d \n', minima);
cg_plot((final_x + 0.25), maxit);

function [x, i, h, beta, s] = cg_plot(x, maxit) 
fplot(@(x) 1 + (1/4000)*x^2 - cos(x))
hold on
s = 0;
tol = 1e-1;  
[f0, g0] = fun(x); 
d0 = -g0; 
for i = 1:maxit
    [f,g,h] = fun(x); 
    d = -g; 
    beta = d/d0;
    d = -1/d0;
    s = d + beta*s;
    a = lsa(x, d, 1);  
    x_new = x + a*d;
    
    if norm(g) < 1e-1   
        fprintf('Convergence reached! x = %g \n', x_new);
        break;   
    end 
    if i == maxit
        disp('Maximum number of iteration reached');
    end 
    a = fun(x);
    fx = a(1); 
    b = fun(x_new);
    fx_new = b(1);
    plot([x, x_new], [fx, fx_new], "r-")
    
    x = x_new;
    d0 = d;
end 
% fprintf('Number of iterations i = %d \n', i);
end 

function [x, i, h, beta, s] = cg(x, maxit) 

s = 0;
tol = 1e-1;  
[f0, g0] = fun(x); 
d0 = -g0; 
for i = 1:maxit
    [f,g,h] = fun(x); 
    d = -g; 
    beta = d/d0;
    d = -1/d0;
    s = d + beta*s;
    a = lsa(x, d, 1);  
    x_new = x + a*d;
    
    if norm(g) < 1e-1   
        fprintf('Convergence reached! x = %g \n', x_new);
        break;   
    end 
    if i == maxit
        disp('Maximum number of iteration reached');
    end 
    
    x = x_new;
    d0 = d;
end 
% fprintf('Number of iterations i = %d \n', i);
end 

function a = lsa(x, d, a1)                     %  Line search algorithm (same as the one used in cg_w)

c1 = 1e-4; 
c2 = 0.5; 
rho = 0.8; 
amax = 10*a1; 
maxit = 100; 
a0 = 0; 
i = 1;  

[f0, g0] = fun(x);
fold = fun(x+a0*d);

while 1 
    [f, g] = fun(x+a1*d);
    if (f > f0+c1*a1*g0) || ((i>1) && f > fold)
        a = zoom(x, d, a0, a1); 
        return; 
    end 
    if abs(g) <= -c2*g0
        a = a1; 
        return; 
    end 
    if g >= 0 
        a = zoom(x, d, a1, a0); 
        return; 
    end 

    if i == maxit
        disp('Maximum number of iteration for Line Search reached')
        a = a1; 
        return; 
    end 
    i = i + 1; 
    a0 = a1; 
    a1 = rho*a0 + (1-rho)*amax; 
    fold = f; 
end

end 


function alpha = zoom(x, d, alo, ahi)     % Zoom algorithm (same as the one used in cg_w)

c1 = 1e-4; 
c2 = 0.5;
maxit = 20; 

[f0, g0] = fun(x); 
j = 0; 

while 1 
    a = (alo+ahi)/2; 
    [f, g] = fun(x+a*d);
    if (f > f0 + c1*a*g0 || f > fun(x+alo*d))
        ahi = a;
    else 
        if abs(g) <= -c2*g0
            alpha = a; 
            return; 
        end 
        if g*(ahi - alo) >= 0
            ahi = alo; 
        end 
        alo = a; 
    end 
    if j == maxit
        alpha = a; 
        return; 
    end 
    j = j + 1; 
end 
end 


function [f, g, h] = fun(a)
syms x  
fx = @(x) 1 + (1/4000)*x^2 - cos(x);
f = fx(a);
der = diff(fx, x); 
g = vpa(subs(der, x, a)); 
der2 = diff(fx, x, 2); 
% h = vpa(subs(der2, x, a)); 
hess = hessian(fx, [x]);
h = vpa(subs(hess, a));
end 