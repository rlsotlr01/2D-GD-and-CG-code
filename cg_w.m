clc; clear; 
x = 0.9;                                            % initial value of x
max = 5; 
tic
[x, i, h] = cg(x, max);                             % calls the inexact CG method with given initial value and maximum iterations
toc

function [x, i, h] = cg(x, maxit) 
% This function runs the ssd_w function once again on the final minimum
% point (+ a marginal displacement to allow for plotting). But this time
% the function is only used to plot the descent direction for the final
% minimum point. 

fplot(@(x) 10 + x^2 - 10*cos(2*pi*x))               % this part plots the function 
hold on

tol = 1e-4;  
[f0, g0] = fun(x); 
d0 = -g0; 
s = -g0;

for i = 1:maxit                                     % initiation of the main CG algorithm but with inexact step length (Wolfe Line Search) method
    [f,g,h] = fun(x); 
    d = -g;
    a = lsa(x, d, 1);
%     lambda = -(g*s)/((s*h*s));
    x_new = x + a*d;
    g = -1/g0; 
    
    a = fun(x);
    fx = a(1); 
    b = fun(x_new);
    fx_new = b(1);
    plot([x, x_new], [fx, fx_new], "g-")

        if norm(g) < tol  
        fprintf('Convergence reached! x = %g \n', x_new);
        break;   
        end 
   
        if i == maxit
        disp('Maximum number of iteration reached');
        end 
    beta = g/g0;
    s = -g + beta*s;   


    
    x = x_new;
    d0 = d;
    g0 = g;
end 
% fprintf('Number of iterations i = %d \n', i);
end 

function a = lsa(x, d, a1)                     

% Implementation of Line Search Algorithm with Strong Wolfe conditions
% as found J. Nocedal, S. Wright, Numerical Optimization, 1999 edition
% Algorithm 3.2 on page 59

c1 = 1e-4; 
c2 = 0.5; 
rho = 0.8; 
amax = 10*a1; 
maxit = 20; 
a0 = 0;                  % 0th steplength is 0 
i = 1;  

[f0, g0] = fun(x);        % storing information of function and gradient
                          % at the point for further use

fold = fun(x+a0*d);       % initialization for function with the previous step-length


while 1 
    [f, g] = fun(x+a1*d);  % function with current step-length
    if (f > f0+c1*a1*g0) || ((i>1) && f > fold)   % sufficent decrease check or comparison between f and fold
        a = zoom(x, d, a0, a1);    % a suitable step-length is in [a0,a1]
        return; 
    end 
    if abs(g) <= -c2*g0   % curvature condition
        a = a1;   % current step-length satisfies strong wolfe conditions
        return; 
    end 
    if g >= 0 
        a = zoom(x, d, a1, a0); % suitable step-length is in [a1,a0]
        return; 
    end 

    if i == maxit
        disp('Maximum number of iteration for Line Search reached')
        a = a1; 
        return; 
    end 

    % update for next loop
    i = i + 1; 
    a0 = a1; 
    a1 = rho*a0 + (1-rho)*amax; 
    fold = f; 
end

end 


function alpha = zoom(x, d, alo, ahi)     % Zoom algorithm 

c1 = 1e-4; 
c2 = 0.5;
maxit = 20; 

[f0, g0] = fun(x); 
j = 0; 

while 1 
    a = (alo+ahi)/2; % trial step-length is the middle point of [alo,ahi]
    [f, g] = fun(x+a*d);
    if (f > f0 + c1*a*g0 || f > fun(x+alo*d))  % sufficient decrease or comparison with alo
        ahi = a;   % narrows the interval between [alo,ahi]
    else 
        if abs(g) <= -c2*g0  % curvature condition
            alpha = a;   % a satisfies the strong wolfe conditions
            return; 
        end 
        if g*(ahi - alo) >= 0
            ahi = alo; 
        end 
        alo = a;    % the interval is now [a,alo]
    end 
    if j == maxit
        alpha = a;  % escape condition
        return; 
    end 
    j = j + 1; 
end 
end 


function [f, g, h] = fun(a)
syms x  
fx = @(x) 10 + x^2 - 10*cos(2*pi*x);
f = fx(a);
der = diff(fx, x); 
g = vpa(subs(der, x, a)); 
der2 = diff(fx, x, 2); 
% h = vpa(subs(der2, x, a)); 
hess = hessian(fx, [x]);
h = vpa(subs(hess, a));
end 