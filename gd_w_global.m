clear; clc; 
x = 9.25;
maxit = 100;
% ssd_w(x, maxit);

X = -5:5; 
% Defines a range in we we want to find the global minimum
minima = [];
fx = [];
tic
for k = 1:(length(X)-1)
    % This loop finds all the local minima in the given range. Each minimum
    % point is obtained using gradient descent with strong wolfe conditions
    if abs(ssd_w(X(k+1), maxit) - ssd_w(X(k), maxit)) > 1e-1
        minima = [ssd_w(X(k), maxit); minima];
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
toc

fprintf('The final value of x is: %g \n', final_x);
fprintf('All local minima within this range: \n')
fprintf('%d \n', minima);
ssd_plot((final_x + 2), maxit);

function [x, i] = ssd_plot(x, maxit)
% This function runs the ssd_w function once again on the final minimum
% point (+ a marginal displacement to allow for plotting). But this time
% the function is only used to plot the descent direction for the final
% minimum point. 

fplot(@(x) 1 + (1/4000)*x^2 - cos(x))
hold on 

tol = 1e-4;  
[f, g] = fun(x);  
for i = 1:maxit
    d = -g; 
    a = lsa(x, d, 1); 
    x_new = x + a*d; 
    [f, g] = fun(x_new);  
    if norm(g) < tol   
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
    plot([x, x_new], [fx, fx_new], "b-")
    x = x_new;
end 
fprintf('Number of iterations i = %d \n', i);
end 

function [x, i] = ssd_w(x, maxit) 
tol = 1e-4;  
[f, g] = fun(x);  
for i = 1:maxit
    d = -g; 
    a = lsa(x, d, 1); 
    x = x + a*d; 
    [f, g] = fun(x);  
    if norm(g) < tol   
%         fprintf('Convergence reached! x = %g \n', x);
        break;   
    end 
    if i == maxit
%         disp('Maximum number of iteration reached');
    end 
end 
% fprintf('Number of iterations i = %d \n', i);
end 
 

function a = lsa(x, d, a1)                     %  Line search algorithm (same as the one used in gd_w)

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


function alpha = zoom(x, d, alo, ahi)     % Zoom algorithm (same as the one used in gd_w)

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

% x =  -6.2800540280912647;
% [f, g] = fun(x) 
% new = hello(x) 

% The above lines are only written to test the function (fun()) with
% different values of x. 

function [f, g] = fun(a)
syms x  
fx = @(x) 1 + (1/4000)*x^2 - cos(x);
f = fx(a);
der = diff(fx, x); 
g = vpa(subs(der, x, a)); 
end 


%% doodle
% function b = hello(x)
% b = fun(x); 
% end 



 

% syms x 
% 
% f = @(x) x^2 + 1; 
% b = hello(f); 
% 
% function a = hello(f)
% 
% syms x
% d = diff(f, x); 
% g0 = vpa(subs(d, x, 4)); 
% a = g0
% 
% end 

% fplot(@(x) 1 + (1/4000)*x^2 - cos(x))
