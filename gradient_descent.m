A = [2.5409 -0.0113; -0.0113 0.5287];  
b = [1.3864; 0.3719];
x0 = [1; 1];


xk = x0; 
rk = A*xk - b; 
pk = -rk; 

iterate = 1; 
curve_x = [];

while norm(rk) > 1e-6
 
    apk = A*pk; 
    rk_0 = rk; 

    alpha = (rk'*rk)./(pk'*apk); 
    xk = xk + alpha*pk; 
    rk = rk - alpha.*(A*rk);  
    pk = -rk; 
    iterate = iterate+1; 
    curve_x = [curve_x, xk]; 
   
end 

