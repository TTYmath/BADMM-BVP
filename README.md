# BADMM-BVP: Bregman ADMM for Bethe variational problem

This repository contains a MATLAB implementation of the Bregman ADMM for Bethe variational problem (BVP) and Quantum Bethe variational problem (QBVP).

Details of the algorithm can be found in the following paper:  

Yuehaw Khoo, Tianyun Tang, and Kim-Chuan Toh, A Bregman ADMM for Bethe variational problem. 

## Reproduce results in paper

To test the spin glass model, run 'Run_Spin.m'. 

To test the sensor network localization problem, run 'Run_SNL'.

To test the quantum ising model, run 'Run_Ising'.

## Test your own problem

You may also test you own problem using solvers:

'BADMM.m': Bregman ADMM for BVP, 'QT_BADMM.m' Bregman ADMM for QBVP

Please input parameters following the input the the previous three examples.

## Other algorithms

Apart from Bregman ADMM, other algorithms for BVP and QBVP have also been implemented to comparison in numerical experiments. They are:

'CCCP.m': Double-loop algorithm for BVP

Yuille, A. L. (2002). CCCP algorithms to minimize the Bethe and Kikuchi free energies: Convergent alternatives to belief propagation. Neural computation, 14(7), 1691-1722.

'JBP.m': Jacobi Belief propagation for BVP, 'GSBP.m': Gauss-Seidel Belief propagation for BVP

Pearl, J. (2014). Probabilistic reasoning in intelligent systems: networks of plausible inference. Elsevier.

'QT_JBP.m': Jacobi Belief propagation for QBVP, 'QT_GSBP.m': Gauss-Seidel Belief propagation for QBVP

Zhao, J., Bondesan, R., & Luk, W. (2024). Quantum Belief Propagation.

Please cite the above papers if you use these implementations of the algorithms in your work.





