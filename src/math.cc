/*Include Files:*/
#include <iostream>

#include <R.h>
#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

#include "basic.h"
#include "math.h"

mat MathOp::kronecker(const mat & A, const mat & B) {
  int nrowA = A.n_rows;
  int ncolA = A.n_cols;

  int nrowB = B.n_rows;
  int ncolB = B.n_cols;

  mat kron = zeros<mat>(nrowA * nrowB, ncolA * ncolB);
  //  cout << kron.n_rows << " " << kron.n_cols << endl;
  for (int i=0;i<nrowA;i++) {
    for (int j=0;j<ncolA;j++) {
      //      cout << kron.n_rows << " " << kron.n_cols << endl;
      //      cout << A(i,j) << endl;
      kron.submat(i*nrowB, j*ncolB, (i+1)*nrowB-1, (j+1)*ncolB-1) = A(i,j)*B;
      //      cout << kron << endl;
    }
  }
  return kron;
}

mat MathOp::outer(const vec & x, const vec & y) {
  int n = x.n_elem;

  mat xy(n, n);
  for (int i=0;i<n;i++) {
    for (int j=0;j<n;j++) {
      xy(i,j) = x(i)*y(j);
    }
  }
  
  return xy;
}


vec MathOp::vectorize(const mat & A) {
  int n = A.n_elem;
  vec y(n);

  for (int i=0;i<n;i++) {
    y(i) = A(i);
  }

  return y;
}

double MathOp::Logit(double p) {
  return log(p/(1-p));
}

// v: input vectors
// u: orthonormalized output vectors
void MathOp::gramSchmidt(mat & u, const mat & v) {
  const int n = u.n_cols;

  u.col(0) = v.col(0);
  for (int i=0;i<n;i++) {
    u.col(i) = v.col(i);
    for (int j=0;j<i;j++) {
      //u.col(i) -= proj(v.col(i), u.col(j)); // Numerical unstable version
      u.col(i) -= proj(u.col(i), u.col(j)); // Numerical stable version
    }
    u.col(i) = u.col(i)/norm(u.col(i), 2); // Normalize
  }
}

mat MathOp::robustCholesky(const mat & S) {
  mat S_sqrt = chol(S);

  if (S_sqrt.n_elem == 0) {
    Rprintf("Add a small amount to the diagonal before Cholesky\n");
    const mat SAddDiag = 0.99*S + 0.01*diagmat(S);
    S_sqrt = trans(chol(SAddDiag));
  }
  if (S_sqrt.n_elem == 0) {
    Rprintf("Diagonalize before Cholesky\n");
    S_sqrt = trans(chol(diagmat(S)));
  }

  return S_sqrt;
}

vec MathOp::findSteepestDescent(double (*func)(const vec &, int &), vec p, const int npar, const double h) {
  int status;
  vec pder(npar);
  vec pi(npar);
  double f0 = func(p, status);

  for (int i=0;i<npar;i++) {
    pi(i) = p(i);
  }
  for (int i=0;i<npar;i++) {
    pi(i) = p(i) + h;
    double fi = func(pi, status);

    pi(i) = p(i);
    pder(i) = (fi - f0)/h;
  }
  pder = -pder;
  //  pder.print("Steepeste descent direction=");
  //  delete [] pi;

  return pder;
}
