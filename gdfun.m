function [xk, iterations] = gdfun(A, b, x0)

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

iterations = iterate - 1;

end 