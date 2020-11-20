# powell_ceres
This repository has the purpose of timing the minimization example https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/powell.cc of the ceres solver (http://ceres-solver.org/). 

Three different methods are used to define cost functions and its derivates.

1. Analytic derivation of jacobians using a SizedCostFunction

2. Automatic differentiation defining a single cost functor with 4 residuals

3. Automatic differenttiation defining 4 cost functors (as in the example given by ceres)


Here are the results on my machine in seconds for 10000 runs:

| Method         | Analytic    | Automatic 1 cost function | Automatic 4 cost functions |
| :------------: | :---------: | :-----------------------: | :------------------------: |
| total time [s] | 0.802201    | 2.1996                    | 2.91902                    |
| mean time [s]  | 0.0000802201 | 0.00021996                | 0.000291902                |


Analytic faster than auto one cost function       by 274.196%  
Analytic faster than original four cost functions by 363.876%  

This shows that even for small problems optimization time strongly depends on the way cost functions are implemented.  
