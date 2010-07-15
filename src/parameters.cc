/*Include Files:*/
#include <iostream>


#include <R.h>

#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

#include "basic.h"
#include "math.h"
#include "parameters.h"

double Parameters::minlambda;
double Parameters::maxlambda;

Parameters::Parameters(int nSup) {
  psi=0;
  mu=0;
  omega.set_size(nSup);
  lambda.set_size(nSup);
}

Parameters::Parameters(const double mu_, const double psi_,
		       const vec & omega_, const vec & lambda_, const int transf_, const int check) {
  psi = psi_;
  mu = mu_;
  omega = omega_;
  lambda = lambda_;

  if (check) {
    const int error = checkPars(transf_);
    if (error) {
      Rprintf("Parameters::Parameters. Exit\n");
      exit(-1);
    }
  }
}

Parameters::Parameters(const vec & x, const int transf, const int check) {
  setPars(x, transf, check);
}

void Parameters::setPars(const vec & x, int const transf, const int check) {
  if (transf == NOTRANSF)
    setPars0(x);
  else if (transf==0)
    setPars1(x);
  else
    setPars2(x);
  if (transf != NOTRANSF && check) {
    const int error = checkPars(transf);

    if (error) {
      Rprintf("Parameters::setPars. Exit\n");
      exit(-1);
    }
  }
}
 

void Parameters::setPars1(const vec & x) {
  int nSup = (x.n_elem - 2)/2;

  mu = x(0);
  //  lambda = x.rows(1,nSup);
  lambda = 1/(1 + exp(-x.rows(1,nSup)));
  int i;
  for (i=1;i<nSup;i++) {
    lambda[i] *= lambda[i-1];
  }
  for (i=0;i<nSup;i++) {
    lambda[i] = lambda[i]*(maxlambda - minlambda) + minlambda;
  }

  psi = exp(x(nSup+1));
  omega = exp(x.rows(nSup+2, 2*nSup+1));
}

void Parameters::setPars2(const vec & x) {
  int nSup = (x.n_elem - 2)/2;
  mu = x(0);
  lambda = zeros<vec>(nSup);
  if (nSup == 1) {
    //    lambda = x.rows(1,nSup);
    lambda(0) = minlambda + (maxlambda - minlambda)/(1 + exp(-x(1)));
  }
  else if (nSup == 2) {
    lambda(0) = (maxlambda - minlambda)/(1 + exp(-x(1))) + minlambda;
    lambda(1) = minlambda/(1 + exp(-x(2)));
  }

  psi = exp(x(nSup+1));
  omega = exp(x.rows(nSup+2, 2*nSup+1));
}

vec Parameters::extractParsInv(int transf) {
  vec parvec;

  if (transf == 0)
    parvec = extractParsInv1();
  else
    parvec = extractParsInv2();

  return parvec;
}

vec Parameters::extractParsInv1() {
  const int nSup = lambda.n_elem;
  vec parvec(2*nSup+2);

  parvec(0) = mu;
  double lambdastar = (lambda(0) - minlambda)/(maxlambda-minlambda);
  parvec(1) = Logit(lambdastar);
  for (int i=1;i<nSup;i++) {
    lambdastar = (lambda(i) - minlambda)/(lambda(i-1) - minlambda);
    parvec(i+1) = Logit(lambdastar);
  }

  parvec(nSup+1) = log(psi);
  parvec.rows(nSup+2, 2*nSup+1) = log(omega);

  return parvec;
}
vec Parameters::extractParsInv2() {
  int nSup = lambda.n_elem;
  vec parvec(2*nSup+2);

  parvec(0) = mu;
  if (nSup == 1) {
    parvec(1) = Logit((lambda(0)-minlambda)/(maxlambda-minlambda));
  }
  else if (nSup == 2) {
    parvec(1) = Logit((lambda(0)-minlambda)/(maxlambda-minlambda));
    parvec(2) = Logit(lambda(1)/minlambda);
  }

  parvec(nSup+1) = log(psi);
  parvec.rows(nSup+2, 2*nSup+1) = log(omega);

  return parvec;
}

double Parameters::getPar(const int ind) {
  int nSup = lambda.n_elem;

  double val;
  if (ind == 0) {
    val = mu;
  }
  else if (ind <= nSup) {
    val = lambda(ind-1);
  }
  else if (ind <= nSup+1) {
    val = psi;
  }
  else {
    val = omega(ind-nSup-2);
  }
  return val;
}

int Parameters::checkPars(const int transf) {
  int error = 0;
  int nsup = lambda.n_elem;

  if (isnan(mu)) {
    error = 1;
    Rprintf("Error(Parameters::checkPars): mu is nan\n");
  }
  if (isnan(psi)) {
    error = 1;
    Rprintf("Error(Parameters::checkPars): psi is nan\n");
  }
  else if (psi <= 0.0) {
    error = 1;
    Rprintf("Error(Parameters::checkPars): psi <= 0.0\n");
  }

  for (int i=0;i<nsup;i++) {
    if (isnan(lambda(i))) {
      error = 1;
      Rprintf("Error(Parameters::checkPars): lambda(%1d) is nan\n", i);
    }
  }
  for (int i=1;i<nsup;i++) {
    if (lambda(i) > lambda(i-1)) {
      error = 1;
      Rprintf("Error(Parameters::checkPars): lambda(%1d)=%6.4f < lambda(%1d)=%6.4f\n",
	      i, lambda(i), i-1, lambda(i-1));
    }
  }
  if (transf == 0) {
    for (int i=0;i<nsup;i++) {
      if (lambda(i) < minlambda) {
	error = 1;
	Rprintf("Error(Parameters::checkPars): lambda(%1d)=%6.4f < minlambda\n", i, lambda(i));
      }
      else if (lambda(i) > maxlambda) {
	error = 1;
	Rprintf("Error(Parameters::checkPars): lambda(%1d)=%6.4f > maxlambda\n", i, lambda(i));
      }
    }
  }
  else if (transf == 1 && nsup == 2) {
     if (lambda(0) < minlambda) {
       error = 1;
       Rprintf("Error(Parameters::checkPars): lambda(0)=%6.4f <= minlambda\n", lambda(0));
     }
     else if (lambda(0) > maxlambda) {
       error = 1;
       Rprintf("Error(Parameters::checkPars): lambda(0)=%6.4f >= maxlambda\n", lambda(0));
     }
     if (lambda(1) < 0.0) {
       error = 1;
       Rprintf("Error(Parameters::checkPars): lambda(1)=%6.4f <= 0.0\n", lambda(1));
     }
     else if (lambda(1) > minlambda) {
       error = 1;
       Rprintf("Error(Parameters::checkPars): lambda(1)=%6.4f >= minlambda\n", lambda(1));
     }
  }
  for (int i=0;i<nsup;i++) {
    if (isnan(omega(i))) {
      error = 1;
      Rprintf("Error(Parameters::checkPars): omega(%1d) is nan\n", i);
    }
    if (omega(i) < 0.0) {
      error = 1;
      Rprintf("Error(Parameters::checkPars): omega(%1d)<=0.0\n", i);
    }
  }

  return error;
}

int Parameters::numberOfSuperPositions(const vec & par) {
  const int n = (par.n_elem - 2)/2;

  return n;
}

vec Parameters::asvector() {
  const int nSup = lambda.n_elem;
  const int npar = 2 + 2*nSup; // mu, psi, lambda, omega

  vec x(npar);

  int ind=0;
  x(ind++) = mu;

  for (int i=0;i<nSup;i++) {
    x(ind++) = lambda(i);
  }

  x(ind++) = psi;

  for (int i=0;i<nSup;i++) {
    x(ind++) = omega(i);
  }

  return x;
}

void Parameters::setPars0(const vec & x) {
  const int nSup = (x.n_elem - 2)/2;
  int ind=0;

  mu = x(ind++);

  lambda = zeros<vec>(nSup);
  for (int i=0;i<nSup;i++) {
    lambda(i) = x(ind++);
  }

  psi = x(ind++);

  omega = zeros<vec>(nSup);
  for (int i=0;i<nSup;i++) {
    omega(i) = x(ind++);
  }

  return;
}


mat Parameters::gradient(const int transf) {
  const int nSup = lambda.n_elem;
  const int npar = 2*nSup + 2;
  mat grad = zeros<mat>(npar,npar);

  // mu
  grad(0,0) = 1.0;
  
  // lambda
  int ind = 1;
  if (transf == 1 && nSup == 2) {
    grad(ind,ind) = (lambda(0)-minlambda)*(1-(lambda(0)-minlambda)/(maxlambda-minlambda));
    grad(ind+1,ind+1) = lambda(1)*(1-lambda(1)/minlambda);
  }
  else {
    vec lambdaStar(nSup);
    vec par = this->extractParsInv(transf);
    for (int i=0;i<nSup;i++) {
      lambdaStar(i) = (lambda(i)-minlambda)/(maxlambda-minlambda);
    }
    for (int i=0;i<nSup;i++) {
      for (int j=i;j<nSup;j++) {
	grad(ind+i,ind+j) = (maxlambda-minlambda)* lambdaStar(j) / (1 + exp(par(ind+i)));
      }
    }
  }
  // psi
  ind = nSup+1;
  grad(ind,ind) = psi;

  // omega
  ind = nSup+2;
  for (int i=0;i<nSup;i++)
    grad(ind+i,ind+i) = omega(i);

  return grad;
}

void Parameters::print(const char * str) const {
  Rprintf("%s\n", str);
  print();
}

void Parameters::print() const {
  Rprintf("mu: %8.5f\n", mu);
  Rprintf("xi: %8.5f ", psi);
  Rprintf("lambda: ");
  int n = lambda.n_elem;
  for (int i=0;i<n;i++)
    Rprintf("%8.5f ", lambda(i));
  Rprintf("omega: ");
  for (int i=0;i<n;i++)
    Rprintf("%8.5f ", omega(i));
  Rprintf("\n");
}
