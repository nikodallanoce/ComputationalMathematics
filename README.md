# ComputationalMathematics
Computational Mathematics for Learning and Data Analysis project for the a.y. 2021/2022.
## Group Members
- [Niko Dalla Noce](https://github.com/nikodallanoce)
- [Alessandro Ristori](https://github.com/RistoAle97)
- [Simone Rizzo](https://github.com/simone-rizzo)
## Wilcard Project
(P) is the linear least squares problem
$$\displaystyle \min_{w} \lVert \hat{X}w-\hat{y} \rVert$$
where
$$\hat{X}= \begin{bmatrix} X^T \newline \lambda I \end{bmatrix}, \hat{y}= \begin{bmatrix} y \newline 0 \end{bmatrix},$$
with $X$ the (tall thin) matrix from the ML-cup dataset by prof. Micheli, and $y$ is a random vector.
- (A1) is an algorithm of the class of **limited-memory quasi-Newton methods**.
- (A2) is **thin QR factorization with Householder reflectors**, in the variant where one does not form the matrix $Q$, but stores the Householder vectors $u_k$ and uses them to perform (implicitly) products with $Q$ and $Q^T$.
- (A3) is an algorithm of the class of **Conjugate Gradient methods**.
- (A4) is a **standard momentum descent (heavy ball)** approach.
## Repository structure
```bash
πComputationalMathematics
βββ π1_LBFGS  # Limited-memory quasi-Newton method
β   βββ π...
βββ π2_QR  # Thin QR factorization with Householder reflectors
β   βββ π...
βββ π3_CG  # Conjugate gradient method
β   βββ π...
βββ π4_SMD  # Standard momentum descent (heavy ball)
β   βββ π...
βββ πdatasets  # Datasets used by the project
β   βββ ποΈ ML-CUP21-TR.csv
βββ πreport  # Project report
β   βββ π...
βββ πutilities  # Methods for building the matrices, functions and gradients
β   βββ πbuild_lls.m # method to build the function and gradient of lls
β   βββ πbuild_matrices.m # method to build the required matrices
βββ πREADME.md
```
