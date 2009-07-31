/*Include Files:*/
#include <iostream>
#include <cfloat>

#include <R.h>

#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

#include "basic.h"
#include "math.h"
#include "parameters.h"
#include "parametersMulti.h"
#include "optimise.h"
#include "bfgs.h"
#include "conjugate.h"
#include "simulate.h"
#include "ql.h"
#include "indirect.h"

const double Inf = DBL_MAX;

double funcOuter(const vec & par, const int evaluateGradient) {
  Optimise::nFuncEvalOuter++;
  //  Optimise::nGradEval += evaluateGradient;
  //  return qlExtern->likelihood(par, evaluateGradient);
  int status;
  return indirectExtern->distanceMetric(par, status); //, nSim, W, nTimes, parObs);
}

double funcOuter_nr(const vec & p, int & status) {
  //  int n = indirectExtern->dimPar();
  //  Rprintf("funcOuter_nr: n %d\n", n); 
  //  par.print("(funcOuter_nr) par=");

  Optimise::nFuncEvalOuter++;

  double f = indirectExtern->distanceMetric(p, status);

  return f;
}

int n_rosen; // global variable

// Test function
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

  //  x.print("rosen: x=");
  //  Rprintf("rosen: f=%8.4f\n", f);
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

double rosen_nr(const vec & p, int & status) {
  // 
  // Rosenbrock function
  // Matlab Code by A. Hedar (Nov. 23, 2005).
  // The number of variables n should be adjusted below.
  // The default value of n = 2.
  // 
  //  int n = 2;
  const int n = n_rosen;

  Optimise::nFuncEvalOuter++;

  status = EXIT_SUCCESS;

  return rosen(p, 0, n).f;
}

Indirect::Indirect(const vec & y,
		   const int nSim_,
		   const int nSup_,
		   const int nTimes_,
		   const int useStartPar_,
		   const double minlambda_,
		   const double maxlambda_,
		   const double ftol_,
		   const double ftol_weak_,
		   const double gradtol_,
		   const int transf_,
		   const int addPenalty_,
		   const int print_level_,
		   const int useRoptimiser_,
		   const int initialSteepestDescent_) : Simulate(nSup_, nTimes_, print_level_),
							QL(y, minlambda_, maxlambda_, transf_, addPenalty_, useRoptimiser_),
							gradtol(gradtol_), ftol(ftol_), ftol_weak(ftol_weak_),
							print_level(print_level_),
							initialSteepestDescent(initialSteepestDescent_)
{
  nSim = nSim_;
  nTimes = nTimes_;
  useStartPar = useStartPar_;

  distMin = DBL_MAX;
  distMinBracket = DBL_MAX;

  if (print_level_ >= 2) 
    print_level_inner = 1;
  else
    print_level_inner = 0;

  //  gradtol = gradtol_;
  //  printLevel = printLevel_;
}

Indirect::~Indirect() {
}

void Indirect::checkContinuity(const vec & startpar, const int nEval, const double delta,
			       int * indpar, mat & xvals, mat & xvals_transf, mat & funcvals) {
  if (print_level == 1) {
    Rprintf("---- Find QL estimate of observed data\n");
    Rprintf(" (use always own implemented bfgs since we need the hessian matrix)\n");
  }
  if (print_level > 1) {
    startpar.print("startpar=");
    Rprintf("gradtol %6.4f  print_level %d\n", gradtol, print_level);
  }
  const int useRoptimiser_save = useRoptimiser;
  useRoptimiser = 0; // Must use own bfgs to get the hessian matrix
  updatePars = ones<ivec>(startpar.n_elem);
  parFull = startpar;
  EstimationObject res = optimise(startpar, gradtol, print_level, updatePars, func); // Call QL::optimise
  useRoptimiser = useRoptimiser_save;

  parObs =  res.par;
  parObs.print("parObs=");
  W = res.H;
  W.print("W=");

  vec par = startpar;
  int npar = par.n_elem;
  //  const double delta=0.001;

  //  const int nSup=1;
  simulateInit();

  int nparOut = 0;
  for (int i=0;i<npar;i++) {
    if (indpar[i]) {
      nparOut++;
    }
  }

  xvals = zeros<mat>(nparOut,nEval);
  xvals_transf = zeros<mat>(nparOut,nEval);
  funcvals = zeros<mat>(nparOut,nEval);

  int status;
  int k = 0;

  for (int i=0;i<npar;i++) {
    if (indpar[i]) {
      startpar_inner = parObs;
      for (int j=0;j<nEval;j++) {
	par(i) = startpar(i) + delta*(j-(nEval-1.0)/2.0);
	funcvals(k,j) = distanceMetric(par, status);

	Parameters pex(par, transf);
	xvals(k,j) = pex.getPar(i);
	xvals_transf(k,j) = par(i);
      }
      par(i) = startpar(i); // Reset
      k++;
    }
  }

  //  funcvals.print("funcvals=");

  //  return funcvals;
}

void Indirect::test(vec & par) {
  ConjugateDirection optOuter(initialSteepestDescent, print_level);

  Optimise::nFuncEvalOuter = 0;

  n_rosen = par.n_elem;

  mat Hconj;
  int restart;

  int error;
  par = zeros<vec>(n_rosen);
  double fret;
  optOuter.conjugateDirection(&rosen_nr, par, fret, ftol, ftol_weak, Hconj, restart, error);

  if (error) {
    Rprintf("Error in conjugate direction.\n");
    //    vec y = trans(Z0.row(0));
    //    saveToFile(y, 0);
    //    exit(-1);
  }
  
}


EstimationObject Indirect::indirectInference(const vec & startpar) {
  if (print_level == 1) {
    Rprintf("---- Find QL estimate of observed data\n");
    Rprintf(" (use always own implemented bfgs since we need the hessian matrix)\n");
  }
  if (print_level >= 2) {
    startpar.print("startpar=");
    Rprintf("gradtol %6.4f  print_level %d\n", gradtol, print_level);
  }
  const int useRoptimiser_save = useRoptimiser;
  useRoptimiser = 0; // Must use own bfgs to get the hessian matrix
  updatePars = ones<ivec>(startpar.n_elem);
  parFull = startpar;
  EstimationObject res = optimise(startpar, gradtol, print_level_inner, updatePars, func); // Call QL::optimise
  useRoptimiser = useRoptimiser_save;

  parObs =  res.par;
  W = res.H;
  if (print_level >= 2) {
    parObs.print("parObs=");
    W.print("W=");
  }
  
  if (!validParsForSimulation(parObs, transf)) {
    Rprintf("QL estimate too extreme to be an initial value for indirect inference\n");
    Rprintf("Indirect inference aborted\n");
    EstimationObject noRes;
    return noRes;
  }
  //  EstimationObject res = EstimationObject(par, parStart, H, gradtol);

  if (print_level >= 1) {
    Rprintf("Number of evaluations in quasi likelihood estimation of observed data\n");
    Rprintf("nFuncEval %d\n", Optimise::nFuncEval);
    Rprintf("nGradEval %d\n", Optimise::nGradEval);
  }

  ConjugateDirection optOuter(initialSteepestDescent, print_level);

  Optimise::nFuncEvalOuter = 0;

  const int nPar = parObs.n_elem;

  mat parsim(nPar, nSim);
  vec parsimSum = zeros<vec>(nPar);

  vec funcvals(nSim);

  mat Hconj;
  int restart;
  int nSimAll = 0;
  for (int k=0;k<nSim;k++) {
    if (print_level >= 1) {
      Rprintf("Simulation iteration %d\n", k);
    }
    simulateInit(); // Simulates new epsilon. Sets new seed

    vec parvec = parObs; // Start values
    startpar_inner = parObs;
    int error;
    double fret;
    restart=0;
    optOuter.conjugateDirection(&funcOuter_nr, parvec, fret, ftol, ftol_weak, Hconj, restart, error);
    //    n_rosen = 2;
    //    vec parvec_rosen = zeros<vec>(n_rosen);
    //    optOuter.conjugateDirection(&rosen_nr, parvec_rosen, print_level, gradtol, Hconj, restart, error);
    if (error) {
      Rprintf("Error in conjugate direction. Write out simulated data\n");
      vec y = trans(Z0.row(0));
      saveToFile(y, 0);
      //      exit(-1);
    }
    nSimAll++;
    if (restart) {
      Rprintf("Restart: Do a new simulation\n");
      k--;
      continue;
    }
    parsim.col(k) = parvec;
    parsimSum = parsimSum + parsim.col(k);
    funcvals(k) = fret;
  }

  if (print_level >= 2) {
    funcvals.print("funcvals=");
  }

  vec parsimMean = (1.0/nSim) * parsimSum;
  parsimMean.print("parsimMean=");
  if (print_level >= 1) {
    cout << "nFuncEval " << Optimise::nFuncEval << endl;
    cout << "nGradEval " << Optimise::nGradEval << endl;
  }

  res = EstimationObject(parsimMean, startpar, parsim, Hconj, gradtol, nSimAll, EXIT_SUCCESS);

  if (print_level >= 2) {
    res.print(transf);
  }
  return res;
}


double Indirect::distanceMetric(const vec & par, int & status) {
  status = EXIT_SUCCESS;

  int npar = par.n_elem;
  int error = 0;
  for (int i=0;i<npar;i++) {
    if (isnan(par(i))) {
      error = 1;
      break;
    }
  }
  if (error) {
    par.print("par=");
    Rprintf("Indirect::distanceMetric: nan in parameter vector\n");
    return Inf;
  }

  Parameters pex(par, transf);
  if (print_level > 2) {
    Rprintf("Enter Indirect::distanceMetric\n");
    pex.print();
  }

  const double deltaT = 1;
  const int resetSeed = 1;
  vec s2;
  if (pex.checkPars(transf)) {
    return Inf;
  }
    
  vec ysim = simulate(pex.mu, pex.lambda, pex.psi,
		      pex.omega, nTimes, deltaT, resetSeed, s2);

  if (ysim.n_elem == 0) {
    return Inf;
  }

  setDataLogReturns(ysim);

  vec startpar = startpar_inner;
  parFull = startpar;
  updatePars = ones<ivec>(startpar.n_elem);
  if (print_level >= 2) {
    startpar.print("Start parameters innner optimisation:");
  }
  EstimationObject res = optimise(startpar, gradtol, print_level_inner, updatePars, func); // Call QL::optimise
  if (res.status == EXIT_FAILURE) {
    status = EXIT_FAILURE;
    return 0;
  }
  const double limit = 10.0;
  vec startpar_diff = res.par - startpar_inner;
  if (norm(startpar_diff, 1) < limit) {
    startpar_inner = res.par;
  }
  else if (print_level >= 2) {
    Rprintf("Difference between estimate from innner optimisation and inner start values too large\n");
    Rprintf("Inner start parameters not updated\n");
  }
  
  vec parsim =  res.par;

  vec pardiff = parsim - parObs;
  double dist = as_scalar(trans(pardiff) * W * pardiff);

  if (dist < distMinBracket) {
    distMinBracket = dist;
    startpar_inner_min_mnbrak = res.par;
  }
  if (dist < distMin) {
    distMin = dist;
    startpar_inner_min = res.par;
    est_inner_min = par;
  }

  if (print_level >= 3) {
    Rprintf("Quit Indirect::distanceMetric\n");
  }
  if (print_level > 1) {
    rowvec tmp = trans(par);
    tmp.print("par(theta)=");
    tmp = trans(parsim);
    tmp.print("parsim-QL =");
    tmp = trans(parObs);
    tmp.print("parObs-QL =");
    tmp = trans(pardiff);
    tmp.print("pardiff-QL=");
  //  W.print("W=");
    Rprintf("outer_func= %6.4f\n", dist);
  }
  return dist;
}
