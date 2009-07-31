/*Include Files:*/
#include <iostream>

#include <R.h>

#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

#include "basic.h"
#include "optimise.h"


int Optimise::nFuncEval = 0;
int Optimise::nFuncEvalOuter = 0;
int Optimise::nGradEval = 0;

Optimise::Optimise(const double value) {
  //  noImprovementValue = value;
}
Optimise::Optimise() {
  //  noImprovementValue = 1e-10;
}
//Optimise::Optimise() : rho(0.01){
  //  rho = .01;
  //  sigma = .9;
  //  tau1 = 9;
  //  tau2 = .1;
  //  tau3 = .5;
//}


void Optimise::checkGradient(FunctionValue (*func)(const vec &, const int),
			     const vec & par, double eps, double tol, int verbose) {
  vec parh = par;
  int n = par.n_elem;

  FunctionValue lik = func(par, 1);
  //  double f0 = lik.f;
  vec df0 = lik.df;
  vec df(n);
  
  int errorGrad = 0;
  //  double maxDist = 0;
  for (int i=0;i<n;i++) {
    parh(i) = par(i) + eps; 
    lik = func(parh, 0);
    double fh1 = lik.f;
    parh(i) = par(i) + 2*eps;
    lik = func(parh, 0);
    double fh2 = lik.f;

    parh(i) = par(i) - eps; 
    lik = func(parh, 0);
    double fh_1 = lik.f;
    parh(i) = par(i) - 2*eps;
    lik = func(parh, 0);
    double fh_2 = lik.f;

    df(i) = (-fh2 + 8*fh1 - 8*fh_1 + fh_2)/(12*eps);

    double dist = fabs(df(i) - df0(i));
    //    if (maxDist < dist) {
    //      maxDist = dist;
    //    }
    double abstol = tol;
    if (fabs(df0(i)) > 1) {
      abstol *= fabs(df0(i));
    }
    if (dist > abstol) {
      errorGrad = 1;
    }
    
  }


  if (errorGrad) {
    warning_own("Numeric gradient differs from analytic gradient");
  }
  if (errorGrad || verbose) {
    Rprintf("Check gradient, eps =%7.4f\n", eps);
    Rprintf("Parameters\n");
    for (int i=0;i<n;i++) {
      Rprintf("%7.4f ", par(i));
    }
    Rprintf("\n");

    Rprintf("Gradient (analytical)\n");
    for (int i=0;i<n;i++) {
      Rprintf("%7.4f ", df0(i));
    }
    Rprintf("\n");

    Rprintf("Gradient (numeric)\n");
    for (int i=0;i<n;i++) {
      Rprintf("%7.4f ", df(i));
    }
    Rprintf("\n");
  }
}
