/*Include Files:*/
#include <iostream>


#include <R.h>

#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

#include "basic.h"
#include "math.h"
#include "parametersMulti.h"

vec ParametersMulti::minlambda;
vec ParametersMulti::maxlambda;

ParametersMulti::ParametersMulti(int nSup) {
  q=2;
  p=1;

  mu.set_size(q);
  psi.set_size(p+q);
  omega.set_size(p+q, nSup);
  lambda.set_size(p+q, nSup);
  phi.set_size(4,2);
}

ParametersMulti::ParametersMulti(const vec mu_, const vec psi_,
				 const mat & omega_, const mat & lambda_,
				 const double phi21_, const int transf_) {
  q=2;
  p=1;

  psi = psi_;
  mu = mu_;
  omega = omega_;
  lambda = lambda_;
  phi = zeros<mat>(4,2);
  phi(2,1) = phi21_;

  const int error = checkPars(transf_);
  if (error) {
    Rprintf("ParametersMulti::ParametersMulti. Exit\n");
    exit(-1);
  }
}

ParametersMulti::ParametersMulti(const vec & x, const int transf) {
  q=2;
  p=1;

  setPars(x, transf);
}

void ParametersMulti::setPars(const vec & x, int const transf) {
  if (transf == NOTRANSF)
    setPars0(x);
  else if (transf==0)
    setPars1(x);
  else
    setPars2(x);
  const int error = checkPars(transf);
  if (error) {
    Rprintf("ParametersMulti::setPars. Exit\n");
    exit(-1);
  }
}

int ParametersMulti::numberOfSuperPositions(const vec & par) {
  const int n = (par.n_elem - (p+2*q+1))/(2*(p+q));

  return n;
}

vec ParametersMulti::asvector() {
  const int nSup = lambda.n_cols;
  const int npar = q + q+p + 2*(q+p)*nSup+1; // mu, psi, lambda, omega, phi21

  vec x(npar);

  int ind=0;
  for (int i=0;i<q;i++) {
    x(ind++) = mu(i);
  }

  for (int k=0;k<p+q;k++) {
    for (int i=0;i<nSup;i++) {
      x(ind++) = lambda(k,i);
    }
  }

  for (int k=0;k<p+q;k++) {
    x(ind++) = psi(k);
  }

  for (int k=0;k<p+q;k++) {
    for (int i=0;i<nSup;i++) {
      x(ind++) = omega(k,i);
    }
  }
  x(ind++) = phi(2,1);

  return x;
}

void ParametersMulti::setPars0(const vec & x) {
  const int nSup = (x.n_elem - (p+2*q+1))/(2*(p+q));
  int ind=0;
  for (int i=0;i<q;i++) {
    mu(i) = x(ind++);
  }

  for (int k=0;k<p+q;k++) {
    for (int i=0;i<nSup;i++) {
      lambda(k,i) = x(ind++);
    }
  }

  for (int k=0;k<p+q;k++) {
    psi(k) = x(ind++);
  }

  for (int k=0;k<p+q;k++) {
    for (int i=0;i<nSup;i++) {
      omega(k,i) = x(ind++);
    }
  }
  phi(2,1) = x(ind++);

  return;
}

// x[0..q-1]: mu (q elements)
// x[q..(q+nSup-1)]: lambda(0, ...) First lambda process
// ...
// x[..]: psi (p+q elements)
// x[..]: Omega as lambda
void ParametersMulti::setPars1(const vec & x) {
  int nSup = (x.n_elem - (p+2*q+1))/(2*(p+q));

  mu = x.rows(0,q-1);

  lambda = zeros<mat>(p+q, nSup);

  int ind = q;
  for (int k=0;k<p+q;k++) {
    lambda.row(k) = trans(1/(1 + exp(-x.rows(ind, ind+nSup-1))));
    int i;

    for (i=1;i<nSup;i++) {
      lambda(k,i) *= lambda(k,i-1);
    }

    for (i=0;i<nSup;i++) {
      lambda(k,i) = lambda(k,i)*(maxlambda(k) - minlambda(k)) + minlambda(k);
    }
    ind += nSup;
  }

  psi = exp(x.rows(ind, ind + p + q -1));

  ind += p + q;
  omega = zeros<mat>(p+q, nSup);
  for (int k=0;k<p+q;k++) {
    omega.row(k) = trans(exp(x.rows(ind, ind+nSup-1)));

    ind += nSup;
  }
  phi = zeros<mat>(4,2);
  phi(2,1) = x(ind);
}

void ParametersMulti::setPars2(const vec & x) {
  int nSup = (x.n_elem - (p+2*q+1))/(2*(p+q));

  mu = x.rows(0,q-1);

  int ind = q;

  lambda = zeros<mat>(p+q, nSup);
  for (int k=0;k<p+q;k++) {
    if (nSup == 1) {
      //    lambda = x.rows(1,nSup);
      lambda(k,0) = minlambda(k) + (maxlambda(k) - minlambda(k))/(1 + exp(-x(ind)));
    }
    else if (nSup == 2) {
      lambda(k,0) = (maxlambda(k) - minlambda(k))/(1 + exp(-x(ind))) + minlambda(k);
      lambda(k,1) = minlambda(k)/(1 + exp(-x(ind+1)));
    }
    ind += nSup;
  }

  psi = exp(x.rows(ind, ind + p + q -1));
  ind += (p+q);
  omega = zeros<mat>(p+q, nSup);
  for (int k=0;k<p+q;k++) {
    omega.row(k) = trans(exp(x.rows(ind, ind+nSup-1)));
    ind += nSup;
  }
  phi = zeros<mat>(4,2);
  phi(2,1) = x(ind);
}

vec ParametersMulti::extractParsInv(int transf) {
  vec parvec;

  if (transf == 0)
    parvec = extractParsInv1();
  else
    parvec = extractParsInv2();

  return parvec;
}

vec ParametersMulti::extractParsInv1() {
  int nSup = lambda.n_cols;
  int npar = q + q+p + 2*(q+p)*nSup+1; // mu, psi, lambda, omega, phi21
  vec parvec(npar);

  parvec.rows(0, q-1) = mu;

  int ind = q;
  for (int k=0;k<p+q;k++) {
    double lambdastar = (lambda(k,0) - minlambda(k))/(maxlambda(k)-minlambda(k));

    parvec(ind) = Logit(lambdastar);
    for (int i=1;i<nSup;i++) {
      lambdastar = (lambda(k,i) - minlambda(k))/(lambda(k,i-1) - minlambda(k));
      parvec(ind+i) = Logit(lambdastar);
    }
    ind += nSup;
  }

  parvec.rows(ind, ind+(p+q-1)) = log(psi);
  ind += (p+q);

  for (int k=0;k<p+q;k++) {
    parvec.rows(ind, ind+nSup-1) = trans(log(omega.row(k)));
    ind += nSup;
  }

  parvec(ind) = phi(2,1);

  return parvec;
}


vec ParametersMulti::extractParsInv2() {
  int nSup = lambda.n_cols;
  int npar = q + q+p + 2*(q+p)*nSup+1; // mu, psi, lambda, omega, phi21
  vec parvec(npar);

  parvec.rows(0, q-1) = mu;
  int ind = q;

  for (int k=0;k<p+q;k++) {
    if (nSup == 1) {
      parvec(ind) = Logit((lambda(k,0)-minlambda(k))/(maxlambda(k)-minlambda(k)));
    }
    else if (nSup == 2) {
      parvec(ind) = Logit((lambda(k,0)-minlambda(k))/(maxlambda(k)-minlambda(k)));
      parvec(ind+1) = Logit(lambda(k,1)/minlambda(k));
    }
    ind += nSup;
  }

  parvec.rows(ind, ind+(p+q-1)) = log(psi);
  ind += (p+q);

  for (int k=0;k<p+q;k++) {
    parvec.rows(ind, ind+nSup-1) = trans(log(omega.row(k)));
    ind += nSup;
  }
  parvec(ind) = phi(2,1);

  return parvec;
}

double ParametersMulti::getPar(const int ind) {
  int nSup = lambda.n_cols;

  int mulim = q;
  int lambdalim = mulim + nSup*(q+p);
  int psilim = lambdalim + q+p;

  double val;
  if (ind < mulim) {
    val = mu(ind);
  }
  else if (ind <= lambdalim) {
    int I = ind - mulim;
    int k = trunc(I/nSup);
    int l = I-k*nSup;

    val = lambda(k,l);
  }
  else if (ind <= psilim) {
    val = psi(ind-lambdalim);
  }
  else {
    int I = ind - psilim;
    int k = trunc(I/nSup);
    int l = I-k*nSup;
      
    val = omega(k,l);
  }
  return val;
}

int ParametersMulti::checkPars(const int transf) {
  int error = 0;
  int nsup = lambda.n_cols;

  for (int k=0;k<q;k++) {
    if (isnan(mu(k))) {
      error = 1;
      Rprintf("Error(ParametersMulti::checkPars): mu(%d) is nan\n", k);
    }
  }
  for (int k=0;k<q+p;k++) {
    if (isnan(psi(k))) {
      error = 1;
      Rprintf("Error(ParametersMulti::checkPars): psi(%d) is nan\n", k);
    }
    else if (psi(k) <= 0.0) {
      error = 1;
      Rprintf("Error(ParametersMulti::checkPars): psi(%d) <= 0.0\n", k);
    }
  }

  for (int k=0;k<q+p;k++) {
    for (int i=0;i<nsup;i++) {
      if (isnan(lambda(k,i))) {
	error = 1;
	Rprintf("Error(ParametersMulti::checkPars): lambda(%1d) is nan\n", i);
      }
    }
    for (int i=1;i<nsup;i++) {
      if (lambda(k,i) > lambda(k,i-1)) {
	error = 1;
	Rprintf("Error(ParametersMulti::checkPars): lambda(%1d)=%6.4f < lambda(%1d)=%6.4f\n",
		i, lambda(k,i), i-1, lambda(k,i-1));
      }
    }
    if (transf == 0) {
      for (int i=0;i<nsup;i++) {
	if (lambda(k,i) < minlambda(k)) {
	  error = 1;
	  Rprintf("Error(ParametersMulti::checkPars): lambda(%1d)=%6.4f < minlambda\n", i, lambda(k,i));
	}
	else if (lambda(k,i) > maxlambda(k)) {
	  error = 1;
	  Rprintf("Error(ParametersMulti::checkPars): lambda(%1d)=%6.4f > maxlambda\n", i, lambda(k,i));
	}
      }
    }
    else if (transf == 1 && nsup == 2) {
      if (lambda(k,0) < minlambda(k)) {
	error = 1;
	Rprintf("Error(ParametersMulti::checkPars): lambda(0)=%6.4f <= minlambda\n", lambda(0));
      }
      else if (lambda(k,0) > maxlambda(k)) {
	error = 1;
	Rprintf("Error(ParametersMulti::checkPars): lambda(0)=%6.4f >= maxlambda\n", lambda(0));
      }
      if (lambda(k,1) < 0.0) {
	error = 1;
	Rprintf("Error(ParametersMulti::checkPars): lambda(1)=%6.4f <= 0.0\n", lambda(1));
      }
      else if (lambda(k,1) > minlambda(k)) {
	error = 1;
	Rprintf("Error(ParametersMulti::checkPars): lambda(1)=%6.4f >= minlambda\n", lambda(1));
      }
    }
  }
  for (int k=0;k<p+q;k++) {
    for (int i=0;i<nsup;i++) {
      if (isnan(omega(k,i))) {
	error = 1;
	Rprintf("Error(ParametersMulti::checkPars): omega(%1d) is nan\n", i);
      }
      if (omega(k,i) < 0.0) {
	error = 1;
	Rprintf("Error(ParametersMulti::checkPars): omega(%1d)<=0.0\n", i);
      }
    }
  }

  return error;
}
