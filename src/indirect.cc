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
#define MAX(A,B) ((A) > (B) ? (A) : (B))

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
		   const double gradMax_,
		   const int initialSteepestDescent_,
		   const int ITMAX_,
		   const int saveDraws_,
		   const int scoreCriterium_,
		   const int optWeightMatrix_) : QL(y, minlambda_, maxlambda_, transf_, addPenalty_, useRoptimiser_, gradMax_, nSup_, nTimes_, print_level_, saveDraws_),
						 gradtol(gradtol_), ftol(ftol_), ftol_weak(ftol_weak_),
						 print_level(print_level_),
						 initialSteepestDescent(initialSteepestDescent_),
						 ITMAX(ITMAX_),
						 scoreCriterium(scoreCriterium_),
						 optWeightMatrix(optWeightMatrix_)
{
  nSim = nSim_;
  nTimes = nTimes_;
  useStartPar = useStartPar_;

  distMin = DBL_MAX;
  distMinBracket = DBL_MAX;

  if (print_level_ >= 3) 
    print_level_inner = 2;
  else if (print_level_ >= 2) 
    print_level_inner = 1;
  else
    print_level_inner = 0;

  //  gradtol = gradtol_;
  //  printLevel = printLevel_;
}

Indirect::~Indirect() {
}

void Indirect::checkContinuity(const vec & startpar, const int nEval, const double delta,
			       int * indpar, mat & xvals, mat & xvals_transf, mat & funcvals,
			       const int profileGradient) {
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
  if (scoreCriterium) {
    if (optWeightMatrix) { // Check this
      mat gr = quasiLikelihood_individual(res.par);
      mat I = gr * trans(gr); // / nTimes;
      W =inv(I);
    }
    else
      W = inv(res.H);
  }
  else {
    W = res.H;
  }

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
  if (profileGradient)
    nparOut++;

  xvals = zeros<mat>(nparOut,nEval);
  xvals_transf = zeros<mat>(nparOut,nEval);
  funcvals = zeros<mat>(nparOut,nEval);

  int status;
  int k = 0;

  vec grad;
  if (profileGradient) {
    const double h = 0.01; // NB: must be tuned
    startpar_inner = parObs;
    grad = findSteepestDescent(&funcOuter_nr, startpar, npar, h);
    grad = grad/norm(grad, 2);
    Rprintf("Check: norm(grad)=%8.6f\n", norm(grad, 2));

  }

  for (int i=0;i<npar+profileGradient;i++) {
    int getProfile = 1;
    if (i<npar) {
      if (!indpar[i])
	getProfile = 0;
    }
    if (getProfile) {
      if (i<npar) 
	Rprintf("Parameter %d\n", i);
     else
	Rprintf("Gradient direction\n");
      startpar_inner = parObs;
      for (int j=0;j<nEval;j++) {
	if (j % 5 == 0) {
	  Rprintf("%d ", j);
	}
	if (i<npar)
	  par(i) = startpar(i) + delta*(j-(nEval-1.0)/2.0);
	else
	  par = startpar + delta*(j-(nEval-1.0)/2.0)*grad;
	funcvals(k,j) = funcOuter_nr(par, status);

	Parameters pex(par, transf);
	if (i<npar) {
	  xvals(k,j) = pex.getPar(i);
	  xvals_transf(k,j) = par(i);
	}
	else {
	  xvals(k,j) = delta*(j-(nEval-1.0)/2.0);
	  xvals_transf(k,j) = xvals(k,j);
	}
      }
      Rprintf("\n");
      if (i<npar)
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

  int iter;
  int error;
  par = zeros<vec>(n_rosen);
  double fret;
  optOuter.conjugateDirection(&rosen_nr, par, fret, ftol, ftol_weak, Hconj, restart, iter, ITMAX, error);

  if (error) {
    Rprintf("Error in conjugate direction.\n");
    //    vec y = trans(Z0.row(0));
    //    saveToFile(y, 0);
    //    exit(-1);
  }
  
}


EstimationObject Indirect::indirectInference(const vec & startpar, const int useQLAsStartPar) {
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
  const mat Hi = inv(res.H);
  sdObs = sqrt(Hi.diag());

  if (scoreCriterium) {
    if (optWeightMatrix) { // Check this
      mat gr = quasiLikelihood_individual(res.par);
      mat I = gr * trans(gr); // / nTimes;
      W =inv(I);
    }
    else
      W = inv(res.H);
  }
  else {
    W = res.H;
  }

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

  ivec iter(nSim);
  ivec iterAll(nSim);

  vec funcvals(nSim);
  vec funcvalsAll(nSim);

  mat Hconj;
  int restart;
  int nSimAll = 0;
  for (int k=0;k<nSim;k++) {
    if (print_level >= 1) {
      Rprintf("Simulation iteration %d (number of success %d)\n", nSimAll, k);
    }
    simulateInit(); // Simulates new epsilon. Sets new seed

    vec parvec;
    if (useQLAsStartPar) {
      parvec = parObs; // Start values
      startpar_inner = parObs;
    }
    else {
      parvec = startpar; // Start values
      startpar_inner = startpar;
    }
    int error;
    double fret;
    restart=0;
    int iter_powell;
    optOuter.conjugateDirection(&funcOuter_nr, parvec, fret, ftol, ftol_weak, Hconj, restart, iter_powell, ITMAX, error);
    //    n_rosen = 2;
    //    vec parvec_rosen = zeros<vec>(n_rosen);
    //    optOuter.conjugateDirection(&rosen_nr, parvec_rosen, print_level, gradtol, Hconj, restart, error);
    if (error) {
      Rprintf("Error in conjugate direction. Write out simulated data\n");
      vec y = trans(Z0.row(0));
      saveToFile(y, 0);
      //      exit(-1);
    }

    if (nSimAll >= nSim) {
      ivec iterAll_copy = iterAll;
      vec funcvalsAll_copy = funcvalsAll;

      iterAll.set_size(nSimAll+1);
      funcvalsAll.set_size(nSimAll+1);
      iterAll.rows(0, nSimAll-1) = iterAll_copy;
      funcvalsAll.rows(0, nSimAll-1) = funcvalsAll_copy;
    }
    funcvalsAll(nSimAll) = fret;
    iterAll(nSimAll) = iter_powell;
    nSimAll++;

    if (restart) {
      Rprintf("Restart: Do a new simulation\n");
      k--;
      continue;
    }
    parsim.col(k) = parvec;
    parsimSum = parsimSum + parsim.col(k);
    funcvals(k) = fret;
    iter(k) = iter_powell;
  }

  if (print_level >= 2) {
    funcvals.print("funcvals=");
  }

  vec parsimMean = (1.0/nSim) * parsimSum;
  if (print_level >= 1) {
    parsimMean.print_trans("Indirect inference estimate=");
    cout << "nFuncEval " << Optimise::nFuncEval << endl;
    cout << "nGradEval " << Optimise::nGradEval << endl;
  }

  res = EstimationObject(parsimMean, startpar, parsim, Hconj, gradtol, funcvals,
			 funcvalsAll, iter, iterAll, nSimAll, EXIT_SUCCESS);

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

  const int checkPars=0;
  Parameters pex(par, transf, checkPars);
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

  double dist;
  if (!scoreCriterium) {
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
    dist = as_scalar(trans(pardiff) * W * pardiff);

    
    if (dist < distMinBracket) {
      distMinBracket = dist;
      startpar_inner_min_mnbrak = res.par;
    }
    
    if (dist < distMin) {
      startpar_inner_min = res.par;
    }

    if (print_level > 1) {
      rowvec tmp = trans(parsim);
      tmp.print("parsim-QL =");
      tmp = trans(pardiff);
      tmp.print("pardiff-QL=");
    }
  }
  else { // Score criterium
    const int evaluateGradient=1;
    //    FunctionValue fval = quasiLikelihood(parObs, evaluateGradient);
    FunctionValue fval = func(parObs, evaluateGradient);
    vec df = fval.df;
    dist = as_scalar(trans(df) * W * df);
  }

  if (dist < distMin) {
    distMin = dist;
    est_inner_min = par;
  }

 if (print_level >= 3) {
    Rprintf("Quit Indirect::distanceMetric\n");
  }
  if (print_level > 1) {
    rowvec tmp = trans(par);
    tmp.print("par(theta)=");
    tmp = trans(parObs);
    tmp.print("parObs-QL =");
    Rprintf("outer_func= %6.4f\n", dist);
  }
  return dist;
}

mat Indirect::computeBgradient(const vec & par, const vec & sdQL, const vec & startpar_QL,
			       vec & res0, mat & H) {
  const int debug=0;
  int npar = par.n_elem;
  mat bder(npar, npar);
  double h=0.01;
  for (int k=0;k<1;k++) {
    if (debug)
      Rprintf("h=%6.4f\n", h);
 
    //  print_level = 2; //  For debugging purpose
 
    vec par2 = par;
    Parameters pex0(par, transf);
    computeQLSimEstimate(pex0, startpar_QL, res0, H);
    if (debug)
      res0.print_trans("res0=");
    for (int i=0;i<npar;i++) {
      const double pari = par(i);
      par2(i) = pari + h*sdQL(i);
      Parameters pex(par2, transf);
      par2(i) = pari;

      vec res;
      mat Htmp;
      computeQLSimEstimate(pex, res0, res, Htmp);

      //  if (res.status == EXIT_FAILURE) {
      //    status = EXIT_FAILURE;
      //    return 0;
      //  }
      //      res.print("res=");
      vec diff = res - res0;
      //      diff.print_trans("diff=");
      bder.col(i) = (res - res0)/(h*sdQL(i));      
    }
    if (debug)
      bder.print("bder=");
    h *= 2;
  }

  return bder;
}

void Indirect::computeQLSimEstimate(const Parameters & pex, const vec & startpar, vec & est, mat & H) {
  simulateAndSetData(pex);

  parFull = startpar;
  updatePars = ones<ivec>(startpar.n_elem);
  if (print_level >= 2) {
    startpar.print("Start innner optimisation (computing the derivative of b):");
  }
  EstimationObject res = optimise(startpar, gradtol, print_level_inner, updatePars, func); // Call QL::optim

  est = res.par;
  H = res.H;
}

vec Indirect::simulateData(const Parameters & pex) {
  const int resetSeed = 1;
  const double deltaT = 1;
  vec s2;
  vec ysim = simulate(pex.mu, pex.lambda, pex.psi,
		      pex.omega, nTimes, deltaT, resetSeed, s2);

  return ysim;
}

void Indirect::simulateAndSetData(const Parameters & pex) {
  vec ysim = simulateData(pex);
  setDataLogReturns(ysim);
}


void Indirect::confidenceIntervalsIndirect(const EstimationObject & obj,
					   Parameters & sd,
					   Parameters & lower, Parameters & upper,
					   Parameters & lowerUn, Parameters & upperUn, 
					   const int sandwichIndirect) {

  if (sandwichIndirect) {
    //	  vec estQL_transf = indirectExtern->bestQLEstimate();
    //	  estQL_transf.print("estQL_transf=");
    vec estQLsim;
    mat estHsim;
    //    indirectExtern->QLestimates(parObs, sdObs);
    const vec startpar = parObs;// QL estimate
    //    startpar.print("startpar=");
    mat bder = computeBgradient(obj.par, sdObs, startpar, estQLsim, estHsim);
    //      parObs.print_trans("parObs");
    //      estQLsim.print_trans("estQLsim");

    //      Parameters pex0(estQLsim, *transf);
    Parameters pex0(obj.par, transf);
    simulateAndSetData(pex0);
    mat grQLsim = quasiLikelihood_individual(estQLsim);

    if (print_level >= 2) {
      bder.print("bder=");
    }
    mat HiII = computeSandwichMatrixIndirect(estHsim, grQLsim, bder, nObs, nTimes);

    if (print_level >= 2) {
      HiII.print("Sandwich (Indirect inference)=");
    }

    confidenceIntervals(obj.par, HiII, sd, lower, upper, lowerUn, upperUn);
  }
  else {
    confidenceIntervalsIndirect(obj, sd, lower, upper, lowerUn, upperUn);
  }
}

void Indirect::confidenceIntervalsIndirect(const EstimationObject & obj,
					   Parameters & sd,
					   Parameters & lower, Parameters & upper,
					   Parameters & lowerUn, Parameters & upperUn) {
  const double q0_975 = 1.959964;

  //  const int nsimIndirect = obj.parsim.n_cols;
  vec estimate = mean(obj.parsim, 1);
  vec sd_transf = stddev(obj.parsim, 0, 1);
  
  vec lowervec = estimate - q0_975*sd_transf;
  vec uppervec = estimate + q0_975*sd_transf;

  lowerUn.setPars(lowervec, NOTRANSF);
  upperUn.setPars(uppervec, NOTRANSF);

  lower.setPars(lowervec, transf);
  upper.setPars(uppervec, transf);

  const int npar = estimate.n_elem;
  const int nsup = Parameters::numberOfSuperPositions(estimate);
  Parameters x(nsup);
  const int nsimSD=1000;
  vec sumx = zeros<vec>(npar);
  vec sumx2 = zeros<vec>(npar);
  //  GetRNGstate();
  for (int i=0;i<nsimSD;i++) {
    // Draw random normal numbers with mean=estimate, sd=sd_transf
    const vec eps = normal(npar);
    const vec u = estimate + eps % sd_transf;

    // Transform u
    x.setPars(u, transf);
    const vec xvec = x.asvector();
    sumx = sumx + xvec;
    sumx2 = sumx2 + xvec%xvec;
  }
  //  PutRNGstate();
  vec varvec = (sumx2 - (1.0/nsimSD)*sumx%sumx)/(nsimSD-1.0);
  //  varvec.print("sdvec=");
  vec sdvec = sqrt(varvec);
  //  sdvec.print("sdvec=");
  sd.setPars(sdvec, NOTRANSF);
  //  sd.print();
}


mat Indirect::computeSandwichMatrixIndirect(const mat & H, mat & gr, const mat & bder0,
					    const int nObs, const int nTimes) {

  constrainGradient(gr);

  mat I = gr * trans(gr) / nTimes;
  mat Ji = inv(H) * nTimes;

  //  const int npar = H.n_rows;
  const double lowerLimit = 0.25;
  const double upperLimit = 1/lowerLimit;

  /*
  mat bder = bder0;
  for (int i=0;i<npar;i++) {
    double aL = 0.0;
    double aU = 0.0;
    const double bder_diag_ii = bder(i,i);
    if (bder_diag_ii < lowerLimit) {
      aL = (lowerLimit - bder_diag_ii)/(1-bder_diag_ii);
      Rprintf("Lower truncate bder,i=%d, bder_diag_ii=%6.4f, aL=%6.4f\n", i, bder_diag_ii, aL);
    }
    if (bder_diag_ii > upperLimit) {
      aU = (upperLimit - bder_diag_ii)/(1-bder_diag_ii);
      Rprintf("Upper truncate bder,i=%d, bder_diag_ii=%6.4f, aU=%6.4f\n", i, bder_diag_ii, aU);
    }
    const double a = MAX(aL, aU);
    //    Rprintf("Truncate bder, aU=%6.4f\n", a);
    bder(i,i) = a + (1-a)*bder(i,i);
  }
  */

  double aL = 0.0;
  double aU = 0.0;
  const vec bder_diag = diagvec(bder0);
  const double bder_diag_min = min(bder_diag);
  const double bder_diag_max = max(bder_diag);
  if (bder_diag_min < lowerLimit) {
    aL = (lowerLimit - bder_diag_min)/(1-bder_diag_min);
    if (print_level >= 2) {
      Rprintf("Lower truncate bder, bder_diag_min=%6.4f, aL=%6.4f\n", bder_diag_min, aL);
    }
  }
  if (bder_diag_max > upperLimit) {
    aU = (upperLimit - bder_diag_max)/(1-bder_diag_max);
    if (print_level >= 2) {
      Rprintf("Upper truncate bder, bder_diag_max=%6.4f, aU=%6.4f\n", bder_diag_max, aU);
    }
  }
  const double a = MAX(aL, aU);
  //    Rprintf("Truncate bder, aU=%6.4f\n", a);
  const mat bder = a + (1-a)*bder0;

  //  bder.print("bder=");
  mat Bi = inv(bder);
  

  const double nObs_double = nObs;
  mat S = (1.0 + nObs_double/nTimes) * Bi*Ji*I*Ji*trans(Bi) / nObs_double; // Assume S=1

  if (0) {
    Ji.print("Ji_ii=");
    I.print("I_ii=");
    Bi.print("Bi=");
    Rprintf("nObs=%d, nTimes=%d\n", nObs, nTimes);
    const mat Sdiag = sqrt(S.diag());
    Sdiag.print_trans("sqrt(S.diag_ql())=");
  }

  return S;
}
