/*Include Files:*/
#include <iostream>
#include <limits.h>
#include <float.h>
#include "armadillo"
#include "math.h"

using namespace arma;
using namespace std;

#include "basic.h"
#include "testfuncs.h"
/*Include Files:*/
#include <iostream>

#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

#include "basic.h"
#include "optimise.h"


FILE * DFILE = stderr;
vec ReadData(string filename);

string test;
int npar;

int main(int argc, char **argv)
{
  const int ntest = 2;
  const string testfunc[2] = {"rosen", "ackley"};
  if (argc < 6) {
    cout << "Usage: " << argv[0] << " <testfunction> <gradtol> <noImprovementValue> <print-level> <startvalues>" << endl;
    cout << "Available test functions: ";
    for (int i=0;i<ntest;i++) {
      cout << testfunc[i] << " ";
    }
    cout << endl;
    exit(-1);
  }

  //  string test = argv[1];
  test = argv[1];
  int found = -1;
  for (int i=0;i<ntest;i++) {
    if (test == testfunc[i]) {
      found = i;
      break;
    }
  }

  if (found<0) {
    cout << "Available test functions: ";
    for (int i=0;i<ntest;i++) {
      cout << testfunc[i] << " ";
    }
    cout << endl;
    stop("Test function not found");
  }

  //  const double gradtol=1e-2;
  const double gradtol = atof(argv[2]);
  //  const int print_level=1;
  const double noImprovementValue = atof(argv[3]);
  const int print_level = atoi(argv[4]);
  npar = argc - 5;
  vec par(npar);
  for (int i=0;i<npar;i++) {
    par(i) = atof(argv[i+5]);
  }
  
  //  FunctionValue (*func)(const vec &, const int) = & rosenbrockBanana;

  Optimise opt(noImprovementValue);

  double precision = DBL_EPSILON;
  //  opt.checkGradient(&func, par, 1e-9, 1e-2, 1);
  //  opt.checkGradient(&func, par, 1e-8, 1e-2, 1);
  //  opt.checkGradient(&func, par, 1e-7, 1e-2, 1);
  //  opt.checkGradient(&func, par, 1e-6, 1e-2, 1);
  opt.checkGradient(&func, par, 1e-5, 1e-2, 1);
  //  opt.checkGradient(&func, par, 1e-4, 1e-2, 1);
  //  opt.checkGradient(&func, par, 1e-3, 1e-2, 1);
  Optimise::nFuncEval = 0;
  Optimise::nGradEval = 0;
  //  opt.checkGradient(&func, par, 1e-3);
  //  opt.checkGradient(&func, par, 1e-2);

  //  FunctionValue (*func)(const vec &, const int, const mat &) = &(qlExtern->likelihood);
  mat H;

  opt.optimal(&func, par, print_level, gradtol, H);

  cout << "Number of function evaluations: " << Optimise::nFuncEval << endl;
  cout << "Number of gradient evaluations: " << Optimise::nGradEval << endl;

  //  EstimationObject obj = estimation(y, nSup, 1);
  //  obj.print();

  int nr; 
  int n=npar; 
  double *x; 
  //  fcn_p fcn; 
  void *state;

  double *xpls; 
  double *fpls; 
  double *gpls; 
  int *itrmcd;

  double *a; 
  double *wrk;

  // optif0(nr, n, x, fcn, state, xpls, fpls, gpls, itrmcd, a, wrk);

}

FunctionValue func(const vec & x, const int evaluateGradient) {
  Optimise::nFuncEval++;
  Optimise::nGradEval += evaluateGradient;
  if (test == "rosenbrock")
    return rosenbrockBanana(x, evaluateGradient);
  else if (test =="rosen")
    return rosen(x, evaluateGradient, npar);
  return FunctionValue(0);
}

// Rosenbrock Banana function
FunctionValue rosenbrockBanana(const vec & x, const int evaluateGradient) {
  double x1 = x(0);
  double x2 = x(1);
  double term1 = (x2 - x1 * x1);
  double term2 = (1 - x1);
  double f = 100 * term1*term1 + term2*term2;

  FunctionValue fval;
  if (evaluateGradient) {
    vec df(2);
    df(0)=-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1);
    df(1)= 200 *      (x2 - x1 * x1);
    fval = FunctionValue(f, df);
  }
  else {
    fval = FunctionValue(f);
  }

  return fval;
}

FunctionValue rosen(const vec & x, const int evaluateGradient, const int n) {
  // 
  // Rosenbrock function
  // Matlab Code by A. Hedar (Nov. 23, 2005).
  // The number of variables n should be adjusted below.
  // The default value of n = 2.
  // 
  //  int n = 2;
  double f = 0;
  for (int j=0;j<(n-1);j++) {
    double term1 = (x(j)*x(j) - x(j+1));
    double term2 = (x(j)-1);
    f += 100 * term1*term1 + term2*term2;
  }

  FunctionValue fval;
  if (evaluateGradient) {
    vec df(n);
    df(0) = 400 * x(0) * (x(0)*x(0) - x(1)) - 2 * (1 - x(0));
    df(n-1) = -200 * (x(n-2)*x(n-2) - x(n-1));
    for (int k=1;k<(n-1);k++) {
      df(k) = 400 * x(k) * (x(k)*x(k) - x(k+1)) - 200 * (x(k-1)*x(k-1) - x(k)) - 2 * (1 - x(k));
    }
    fval = FunctionValue(f, df);
  }
  else {
    fval = FunctionValue(f);
  }

  return fval;
}


/*
  FunctionValue ackley(const vec x) {
  // 
  // Ackley function.
  // Matlab Code by A. Hedar (Sep. 29, 2005).
  // The number of variables n should be adjusted below.
  // The default value of n =2.
  // 
  n = 2;
  a = 20; b = 0.2; c = 2*pi;
  s1 = 0; s2 = 0;
  for i=1:n;
  s1 = s1+x(i)^2;
  s2 = s2+cos(c*x(i));
  end
  y = -a*exp(-b*sqrt(1/n*s1))-exp(1/n*s2)+a+exp(1);

  }

  function y = powell(x)  
  // 
  // Powell function 
  // Matlab Code by A. Hedar (Nov. 23, 2005).
  // The number of variables n should be adjusted below.
  // The default value of n = 24.
  // 
  n = 24;
  m = n;
  for i = 1:m/4
  fvec(4*i-3) = x(4*i-3)+10*(x(4*i-2));
  fvec(4*i-2) = sqrt(5)*(x(4*i-1)-x(4*i));
  fvec(4*i-1) = (x(4*i-2)-2*(x(4*i-1)))^2;
  fvec(4*i)   = sqrt(10)*(x(4*i-3)-x(4*i))^2;
  end;
  fvec = fvec';
  y = norm(fvec)^2;

  function y = booth(x)
  // 
  // Booth function 
  // Matlab Code by A. Hedar (Sep. 29, 2005).
  // The number of variables n = 2.
  // 
  y  = (x(1)+2*x(2)-7)^2+(2*x(1)+x(2)-5)^2;

  function y = branin(x)
  // 
  // Branin function 
  // Matlab Code by A. Hedar (Sep. 29, 2005).
  // The number of variables n = 2.
  // 
  y = (x(2)-(5.1/(4*pi^2))*x(1)^2+5*x(1)/pi-6)^2+10*(1-1/(8*pi))*cos(x(1))+10;


  function y = gold(x)
  //  
  // Goldstein and Price function 
  // Matlab Code by A. Hedar (Sep. 29, 2005).
  // The number of variables n = 2.
  // 
  a = 1+(x(1)+x(2)+1)^2*(19-14*x(1)+3*x(1)^2-14*x(2)+6*x(1)*x(2)+3*x(2)^2);
  b = 30+(2*x(1)-3*x(2))^2*(18-32*x(1)+12*x(1)^2+48*x(2)-36*x(1)*x(2)+27*x(2)^2);
  y = a*b;


  function y = perm(x)
  // 
  // Perm function 
  // Matlab Code by A. Hedar (Nov. 23, 2005).
  // The number of variables n should be adjusted below.
  // The default value of n = 4.
  // 
  n = 4;
  b = 0.5;
  s_out = 0;
  for k = 1:n;
  s_in = 0;
  for j = 1:n
  s_in = s_in+(j^k+b)*((x(j)/j)^k-1);
  end
  s_out = s_out+s_in^2;
  end
  y = s_out;

*/
