Roozbeh Golchian Khabaz (瑞文) 
Student ID: 2021280373

***********************************

"cg_regular2.m":       Runs regular Conjugate Gradient with exact line search method on a 2D function. The function is costumizable. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"cg_w.m":              Runs regular Conjugate Gradient with Wolfe Line Search method on a 2D function. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"cg_w_global.m":       Runs the modified versin of cg_w.m to allow for a global minimum search. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"gradient_descent.m":  Runs regular Gradient Descent on an n dimensional function. The function and positive definite matrix A
                       as well as matrix B are costumizable. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"gd_w.m":              Runs Gradient Descent with Wolfe Line search method, as found in J. Nocedal, S. Wright, Numerical Optimization, 1999 edition
                       Algorithm 3.2 on pages 59 - 60. 
                                
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"gd_w_global.m":       Runs the modified versin of cg_w.m to allow for a global minimum search. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"prototype_cg.m":      Runs the regular Conjugate Gradient on the test function "x1 - x2 + 2*x1^2 + 2*x1*x2 + x2^2" and plots the descent 
                       trajectory on a 3D mesh grid. 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"cgfun.m", "gdfun.m", "Solutions.m", and "params.mat" can be used together in order to achieve results of both regular CG and GD for a given function 
in one script. "params.mat" needs to be executed first. Next, the "Solutions.m" script can be run to get the results. 
