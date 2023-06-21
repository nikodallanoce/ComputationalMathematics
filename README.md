# ComputationalMathematics
Computational Mathematics for Learning and Data Analysis project for the a.y. 2021/2022.
## Group Members
- [Niko Dalla Noce](https://github.com/nikodallanoce)
- [Alessandro Ristori](https://github.com/RistoAle97)
- [Simone Rizzo](https://github.com/simone-rizzo)
## Wildcard Project
(P) is the linear least squares problem
$$\displaystyle \min_{w} \lVert \hat{X}w-\hat{y} \rVert$$
where

$$\hat{X}= \begin{bmatrix} X^T \newline \lambda I \end{bmatrix}, \hat{y} = \begin{bmatrix} y \newline 0 \end{bmatrix},$$

with $X$ the (tall thin) matrix from the ML-cup dataset by prof. Micheli, and $y$ is a random vector.
- (A1) is an algorithm of the class of **limited-memory quasi-Newton methods**.
- (A2) is **thin QR factorization with Householder reflectors**, in the variant where one does not form the matrix $Q$, but stores the Householder vectors $u_k$ and uses them to perform (implicitly) products with $Q$ and $Q^T$.
- (A3) is an algorithm of the class of **Conjugate Gradient methods**.
- (A4) is a **standard momentum descent (heavy ball)** approach.
## Repository structure
```bash
📂ComputationalMathematics
├── 📂1_LBFGS  # Limited-memory quasi-Newton method
│   ├── 📄LBFGS.m # implementation of limited memory BFGS
│   ├── 📄run_lbfgs.m # choose the hyper-parameters and run L-BFGS
│   └── 📄...
├── 📂2_QR  # Thin QR factorization with Householder reflectors
│   ├── 📄check_accuracy_thinqr.m # computes the accuracy of our implementation
│   ├── 📄householder_vector.m # builds the householder reflectors
│   ├── 📄thinqr.m # implementation of thin QR factorization
│   ├── 📄run_qr.m # choose the hyper-parameters and run thin QR
│   └── 📄...
├── 📂3_CG  # Conjugate gradient method
│   ├── 📄cg.m # non-optmized version of conjugate gradient
│   ├── 📄cg_opt.m # optmized implementation of conjugate gradient
│   ├── 📄run_cg.m # choose the hyper-parameters and run conjugate gradient
│   └── 📄...
├── 📂4_SMD  # Standard momentum descent (heavy ball)
│   ├── 📄smd.m # implementation of standard momentum descent
│   ├── 📄run_smd.m # choose the hyper-parameters and run standard momentum descent
│   └── 📄...
├── 📂datasets  # Datasets used by the project
│   └── 🗃️ML-CUP21-TR.csv
├── 📂utilities  # Methods for building the matrices, functions and gradients
│   ├── 📄build_lls.m # builds the function and gradient of lls
│   ├── 📄build_matrices.m # builds the required matrices
│   ├── 📄callback.m # computes the metrics
│   └── 📄compare_scalability # comparison of each method scalability
└── 📄README.md
```
