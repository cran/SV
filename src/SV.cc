/*Include Files:*/
#include <iostream>

#include <R.h>
//#include <Rmath.h>

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

int ParametersMulti::q;
int ParametersMulti::p;

QL * qlExtern;
Indirect * indirectExtern;
EstimationObject IndirectEstimation(int nSup, int nSim, int print_level);

FILE * DFILE = stderr;
vec ReadData(string filename);
mat ReadDataMulti(string filename);
vec SetParVec(double * par, double mu, double psi, double * lambda, double * omega,
	      int nSup, int useParVec, int transf);
vec SetParVecMulti(double * par, double * mu, double * psi, double * lambda, double * omega,
		   double phi21, int nSup, int useParVec, int transf);


//
extern "C" {

  void CheckContinuity(char ** datfile, int * nSup,
		       int * nTimes,
		       double * par, int * indpar,
		       double * mu, double * psi,
		       double * lambda, double * omega,
		       double * minlambda, double * maxlambda,
		       double * H,
		       int * nFuncEval, int * nGradEval,
		       double * gradtol, int * nObs,
		       int * transf, int * useParVec,
		       int * addPenalty, int * checkGrad,
		       int * print_level,
		       int * nEval, double * delta, double * xOut, double * xOut_transf, double * fOut,
		       int * useRoptimiser, int * initialSteepestDescent)
  {
    const int ITMAX = 200;
    const double gradMax = 1000*1000;
    vec y = ReadData(*datfile);
    int ny = y.n_elem;

    if (*print_level >= 1) {
      Rprintf("Number of observations: %d\n", ny);
    }
    // Truncate observed time series to last 'nObs' observations
    if (*nObs < ny && *nObs > 0) {
      if (*print_level >= 1) {
	Rprintf("Truncate series to the %d last observations ", *nObs);
      }
      y = y.rows(ny-(*nObs),ny-1);
      if (*print_level >= 1) {
	Rprintf("Number of observations: %d\n", y.n_elem);
      }
    }

    *nObs = y.n_elem - 1; // Log-return data have one less element

    if (*nTimes == -1) {
      *nTimes = *nObs;
    }
    else if (*nTimes < *nObs) {
      Rprintf("Error: nObs > nTimes\nTerminating Check continuity\n");

      return;
    }

    Optimise::nFuncEval = 0;
    Optimise::nGradEval = 0;

    const int useStartPar = 0; // Not in use??

    const int nSim=1;
    const double ftol = 0.001;
    const double ftol_weak = 1.0;
    const int saveDraws = 0;
    indirectExtern = new Indirect(y, nSim, *nSup, *nTimes, useStartPar,
				  *minlambda, *maxlambda, ftol, ftol_weak, *gradtol,
				  *transf, *addPenalty, *print_level, *useRoptimiser, gradMax,
				  *initialSteepestDescent, ITMAX, saveDraws);
    qlExtern = indirectExtern;

    const vec startpar = SetParVec(par, *mu, *psi, lambda, omega, *nSup,
				   *useParVec, *transf);
    if (*print_level >= 2) {
      startpar.print_trans("startpar (unrestricted scale)=");
      Parameters par_debug(startpar, *transf);
      par_debug.print("startpar (original scale)=");
    }

    mat funcvals;
    mat xvals;
    mat xvals_transf;
    indirectExtern->checkContinuity(startpar, *nEval, *delta, indpar, xvals, xvals_transf, funcvals);

    const int npar = funcvals.n_rows;
    const int m = funcvals.n_cols;

    *nFuncEval = Optimise::nFuncEval;
    *nGradEval = Optimise::nGradEval;

    int index=0;
    for (int i=0;i<npar;i++) {
      for (int j=0;j<m;j++) {
	xOut[index] = xvals(i,j);
	xOut_transf[index] = xvals_transf(i,j);
	fOut[index++] = funcvals(i,j);
      }
    }
  }

  void SimulateVolatility(int * nSup,
			  int * nSim,
			  int * nTimes,
			  double * par,
			  double * mu, double * psi,
			  double * lambda, double * omega,
			  double * minlambda_, double * maxlambda_,
			  int * transf, int * useParVec,
			  int * print_level,
			  double * logYRet_out,
			  double * s2_out)
  {
    Parameters::minlambda = *minlambda_;
    Parameters::maxlambda = *maxlambda_;
    const vec parvec = SetParVec(par, *mu, *psi, lambda, omega, *nSup,
				 *useParVec, *transf);
    Parameters pex(parvec, *transf);

    const int resetSeed=0;
    const double deltaT=1.0;
    vec s2;
    Simulate sim(*nSup, *nTimes, *print_level);

    int ind=0;
    for (int isim=0;isim<*nSim;isim++) {
      sim.simulateInit(); // Simulates new epsilon. Sets new seed
      const vec logYRet = sim.simulate(pex.mu, pex.lambda, pex.psi,
				       pex.omega, *nTimes, deltaT, resetSeed, s2);

      if (logYRet.n_elem == 0) {
	return;
      }
      
      for (int i=0;i<(*nTimes);i++) {
	s2_out[ind] = s2(i);
	logYRet_out[ind++] = logYRet(i);
      }
    }

  }

  void SimulateVolatilityMulti(int * nSup,
			       int * nSim,
			       int * nTimes,
			       double * par,
			       double * mu, double * psi,
			       double * lambda, double * omega,
			       double * phi21,
			       double * minlambda_, double * maxlambda_,
			       int * transf, int * useParVec,
			       int * print_level,
			       double * logYRet_out,
			       double * s2_out)
  {
    const int p=ParametersMulti::p;
    const int q=ParametersMulti::q;
    vec minlambdavec(p+q);
    vec maxlambdavec(p+q);
    for (int k=0;k<p+q;k++) {
      minlambdavec(k) = minlambda_[k];
      maxlambdavec(k) = maxlambda_[k];
    }
    ParametersMulti::minlambda = minlambdavec;
    ParametersMulti::maxlambda = maxlambdavec;

    const vec parvec = SetParVecMulti(par, mu, psi, lambda, omega, *phi21, *nSup,
				      *useParVec, *transf);
    ParametersMulti pex(parvec, *transf);

    const int resetSeed=0;
    const double deltaT=1.0;
    mat s2;
    Simulate sim(p, q, *nSup, *nTimes, *print_level);
    sim.simulateInit(); // Simulates new epsilon. Sets new seed
    const mat logYRet = sim.simulateMulti(pex.mu, pex.lambda, pex.psi,
					  pex.omega, pex.phi(2,1), *nTimes, deltaT, resetSeed, s2);
    

    if (logYRet.n_elem == 0) {
      return;
    }

    int ind=0;
    for (int k=0;k<p+q;k++) {
      for (int i=0;i<(*nTimes);i++) {
	s2_out[ind++] = s2(k,i);
      }
    }
    ind=0;
    for (int k=0;k<q;k++) {
      for (int i=0;i<(*nTimes);i++) {
	logYRet_out[ind++] = logYRet(k,i);
      }
    }

  }

  void QuasiLikelihood(char ** datfile, int * nSup,
		       double * par,
		       double * mu, double * psi,
		       double * lambda, double * omega,
		       double * minlambda, double * maxlambda,
		       double * HiUnCorr,
		       double * HiRob,
		       int * nFuncEval, int * nGradEval,
		       double * gradtol, int * nObs,
		       int * transf, int * useParVec,
		       int * addPenalty, int * checkGrad,
		       int * print_level,
		       int * useRoptimiser,
		       int * updatePars_,
		       int * sandwich,
		       double * gradMax,
		       int * nIter)
  {
    vec y = ReadData(*datfile);
    const int n = y.n_elem;

    if (*print_level >= 1) {
      Rprintf("Number of observations: %d\n", y.n_elem);
    }

    if (*nObs < n && *nObs > 0) {
      if (*print_level >= 1) {
	Rprintf("Truncate series to the %d last observations ", *nObs);
      }
      y = y.rows(n-(*nObs),n-1);
      if (*print_level >= 1) {
	Rprintf("Number of observations: %d\n", y.n_elem);
      }
    }

    *nObs = y.n_elem - 1; // Log-return data have one less element

    qlExtern = new QL(y, *minlambda, *maxlambda, *transf, *addPenalty, *useRoptimiser, *gradMax);


    const vec parvec = SetParVec(par, *mu, *psi, lambda, omega, *nSup,
				 *useParVec, *transf);

    if (*print_level >= 1) {
      parvec.print_trans("Start parameters (unrestricted)=");
      Parameters par_debug(parvec, *transf);
      par_debug.print("Start parameters (restricted)=");
    }

    int npar = parvec.n_elem;
    ivec updatePars(npar);

    if (*checkGrad) {
      updatePars = ones<ivec>(npar); // check gradient for all parameters
      qlExtern->setUpdates(updatePars, parvec);
      qlExtern->checkGradient(&func, parvec, 1e-4, 1e-2, 1);
    }

    Optimise::nFuncEval = 0;
    Optimise::nGradEval = 0;
    for (int i=0;i<npar;i++) { // Set the parameters to estimate
      updatePars(i) = updatePars_[i];
    }
    EstimationObject obj = qlExtern->optimise(parvec,
					      *gradtol,
					      *print_level,
					      updatePars,
					      func);
    *nIter = obj.nIter;


    // Extract mu, lambda, psi, omega
    Parameters sd(*nSup);
    Parameters lower(*nSup);
    Parameters upper(*nSup);
    Parameters lowerUn(*nSup);
    Parameters upperUn(*nSup);

    mat Hi;
    qlExtern->confidenceIntervals(obj.par, obj.H, Hi, sd, lower, upper, lowerUn, upperUn, *sandwich);
    
    // Extract mu, lambda, psi, omega, phi21
    Parameters parObj(obj.par, *transf);

    mu[0] = parObj.mu;
    mu[1] = sd.mu;
    mu[2] = lower.mu;
    mu[3] = upper.mu;

    psi[0] = parObj.psi;
    psi[1] = sd.psi;
    psi[2] = lower.psi;
    psi[3] = upper.psi;

    
    int ind=0;

    for (int i=0;i<(*nSup);i++) {
      lambda[ind] = parObj.lambda(i);
      lambda[(*nSup)+ind] = sd.lambda(i);
      lambda[2*(*nSup)+ind] = lower.lambda(i);
      lambda[3*(*nSup)+ind] = upper.lambda(i);

      omega[ind] = parObj.omega(i);
      omega[(*nSup)+ind] = sd.omega(i);
      omega[2*(*nSup)+ind] = lower.omega(i);
      omega[3*(*nSup)+ind] = upper.omega(i);

      ind++;
    }
    ind=0;
    const mat Hi_uncorr = inv(obj.H);
    for (int i=0;i<npar;i++) {
      for (int j=0;j<npar;j++) {
	HiUnCorr[ind] = Hi_uncorr(i,j);
	HiRob[ind++] = Hi(i,j);
      }
    }
  
    *nFuncEval = Optimise::nFuncEval;
    *nGradEval = Optimise::nGradEval;

    if (*print_level >= 1) {
      Rprintf("nFuncEval %d   nGradEval %d\n", *nFuncEval, *nGradEval);
      obj.print(*transf);
    }
  }

  void QuasiLikelihoodMulti(char ** datfile, int * nSup,
			    double * par,
			    double * mu, double * psi,
			    double * lambda, double * omega,
			    double * phi21,
			    double * minlambda, double * maxlambda,
			    double * H,
			    int * nFuncEval, int * nGradEval,
			    double * gradtol, int * nObs,
			    int * transf, int * useParVec,
			    int * addPenalty, int * checkGrad,
			    int * print_level,
			    int * useRoptimiser,
			    int * updatePars_,
			    int * sandwich,
			    double * gradMax,
			    int * nIter)
  {
    mat y = ReadDataMulti(*datfile);
    const int ny = y.n_rows;

    if (*print_level >= 1) {
      Rprintf("Number of observations: %d\n", ny);
    }

    if (*nObs < ny && *nObs > 0) {
      if (*print_level >= 1) {
	Rprintf("Truncate series to the %d last observations ", *nObs);
      }
      y = y.rows(ny-(*nObs),ny-1);
      if (*print_level >= 1) {
	Rprintf("Number of observations: %d\n", y.n_elem);
      }
    }

    *nObs = y.n_elem - 1; // Log-return data have one less element

    ParametersMulti::p = 1;
    ParametersMulti::q = 2;

    const int p_q = ParametersMulti::p + ParametersMulti::q;
    
    vec minlambdavec(p_q);
    vec maxlambdavec(p_q);
    for (int k=0;k<p_q;k++) {
      minlambdavec(k) = minlambda[k];
      maxlambdavec(k) = maxlambda[k];
    }
    qlExtern = new QL(y, minlambdavec, maxlambdavec, *transf, *addPenalty, *useRoptimiser, *gradMax);


    const vec parvec = SetParVecMulti(par, mu, psi, lambda, omega, *phi21,
				      *nSup, *useParVec, *transf);

    if (*print_level >= 1) {
      parvec.print_trans("Start parameters (unrestricted):");
      ParametersMulti par_debug(parvec, *transf);
      par_debug.print("Start parameters (restricted):");
    }

    int npar = parvec.n_elem;
    ivec updatePars(npar);

    if (*checkGrad) {
      updatePars = ones<ivec>(npar); // check gradient for all parameters
      qlExtern->setUpdates(updatePars, parvec);
      qlExtern->checkGradient(&funcMulti, parvec, 1e-6, 1e-2, 1);
    }
    

    Optimise::nFuncEval = 0;
    Optimise::nGradEval = 0;
    for (int i=0;i<npar;i++) { // Set the parameters to estimate
      updatePars(i) = updatePars_[i];
    }
    if (0) {
      qlExtern->setUpdates(updatePars, parvec);     
      Rprintf("QuasiLikelihood check: func %6.4f\n", funcMulti(parvec, 0).f);
    }

    EstimationObject obj = qlExtern->optimise(parvec,
					      *gradtol,
					      *print_level,
					      updatePars,
					      funcMulti);
    *nIter = obj.nIter;

    ParametersMulti sd(*nSup);
    ParametersMulti lower(*nSup);
    ParametersMulti upper(*nSup);

     // Extract confidence intervals for parameters
    ParametersMulti lowerUn(*nSup);
    ParametersMulti upperUn(*nSup);
    mat Hi;
    qlExtern->confidenceIntervalsMulti(obj.par, obj.H, Hi, sd, lower, upper, lowerUn, upperUn, *sandwich);

    // Extract mu, lambda, psi, omega, phi21
    ParametersMulti parObj(obj.par, *transf);
    for (int k=0;k<ParametersMulti::q;k++) {
      mu[k] = parObj.mu(k);
      mu[ParametersMulti::q+k] = sd.mu(k);
      mu[2*ParametersMulti::q+k] = lower.mu(k);
      mu[3*ParametersMulti::q+k] = upper.mu(k);
    }
    
    int ind=0;
    const int nsup_pq = (*nSup)*(p_q);
    for (int k=0;k<p_q;k++) {
      psi[k] = parObj.psi(k);
      psi[p_q+k] = sd.psi(k);
      psi[2*(p_q)+k] = lower.psi(k);
      psi[3*(p_q)+k] = upper.psi(k);

      for (int i=0;i<(*nSup);i++) {
	lambda[ind] = parObj.lambda(k,i);
	lambda[nsup_pq+ind] = sd.lambda(k,i);
	lambda[2*nsup_pq+ind] = lower.lambda(k,i);
	lambda[3*nsup_pq+ind] = upper.lambda(k,i);

	omega[ind] = parObj.omega(k,i);
	omega[nsup_pq+ind] = sd.omega(k,i);
	omega[2*nsup_pq+ind] = lower.omega(k,i);
	omega[3*nsup_pq+ind] = upper.omega(k,i);

	ind++;
      }
    }
    ind=0;
    for (int i=0;i<npar;i++) {
      for (int j=0;j<npar;j++) {
	H[ind++] = obj.H(i,j);
      }
    }
    phi21[0] = parObj.phi(2,1);
    phi21[1] = sd.phi(2,1);
    phi21[2] = lower.phi(2,1);
    phi21[3] = upper.phi(2,1);
    
    *nFuncEval = Optimise::nFuncEval;
    *nGradEval = Optimise::nGradEval;

    if (*print_level >= 1) {
      Rprintf("nFuncEval %d   nGradEval %d\n\n", *nFuncEval, *nGradEval);
      obj.printMulti(*transf);
    }
  }

  void IndirectInference(char ** datfile, int * nSup, int * nSim,
			 int * nTimes,
			 double * par,
			 double * mu, double * psi,
			 double * lambda, double * omega,
			 double * muSim, double * psiSim,
			 double * lambdaSim, double * omegaSim,
			 double * minlambda, double * maxlambda,
			 double * H,
			 int * nFuncEval, int * nGradEval,
			 int * nFuncEvalOuter,
			 double * ftol, double * ftol_weak,
			 double * gradtol, int * nObs,
			 int * transf, int * useParVec,
			 int * addPenalty, int * checkGrad,
			 int * print_level,
			 int * useRoptimiser,
			 int * initialSteepestDescent,
			 int * ITMAX,
			 int * nSimAll,
			 double * funcval,
			 int * niter,
			 int * convergence,
			 char ** simfile,
			 int * error,
			 int * useQLAsStartPar,
			 double * gradMax)
  {
    vec y = ReadData(*datfile);
    const int ny = y.n_elem;

    if (*print_level >= 1) {
      Rprintf("Number of observations: %d\n", y.n_elem);
    }
    // Truncate observed time series to last 'nObs' observations
    if (*nObs < ny && *nObs > 0) {
      if (*print_level >= 1) {
	Rprintf("Truncate series to the %d last observations ", *nObs);
      }
      y = y.rows(ny-(*nObs),ny-1);
      if (*print_level >= 1) {
	Rprintf("Number of observations: %d\n", y.n_elem);
      }
    }

    *nObs = y.n_elem - 1; // Log-return data have one less element

    if (*nTimes == -1) {
      *nTimes = *nObs;
    }
    else if (*nTimes < *nObs) {
      Rprintf("Error: nObs > nTimes\nTerminating Indirect inference\n");

      return;
    }
    

    int sandwichIndirect;
    if (*nSim > 1)
      sandwichIndirect = 0;
    else
      sandwichIndirect = 1;


    const int useStartPar = 0; // Not in use??
    const int saveDraws = 0;
    indirectExtern = new Indirect(y, *nSim, *nSup, *nTimes, useStartPar,
				  *minlambda, *maxlambda, *ftol, *ftol_weak, *gradtol,
				  *transf, *addPenalty, *print_level,
				  *useRoptimiser, *gradMax, *initialSteepestDescent, *ITMAX, saveDraws);
    qlExtern = indirectExtern;

    const vec startpar = SetParVec(par, *mu, *psi, lambda, omega, *nSup,
				   *useParVec, *transf);
    if (*print_level >= 2) {
      startpar.print_trans("Start parameters (unrestricted)=");
      Parameters par_debug(startpar, *transf);
      par_debug.print("Start parameters (restricted)=");
    }

    Optimise::nFuncEval = 0;
    Optimise::nGradEval = 0;
    Optimise::nFuncEvalOuter = 0;

    EstimationObject obj = indirectExtern->indirectInference(startpar, *useQLAsStartPar);

    if (obj.status == EXIT_FAILURE) {
      if (*print_level >= 1) {
	Rprintf("Indirect inference aborted\n");
      }
      *error = 1;
      return;
    }

    Parameters sd(*nSup);
    Parameters lower(*nSup);
    Parameters upper(*nSup);
    Parameters lowerUn(*nSup);
    Parameters upperUn(*nSup);
    Parameters sd2(*nSup);
    Parameters lower2(*nSup);
    Parameters upper2(*nSup);

    indirectExtern->confidenceIntervalsIndirect(obj, sd, lower, upper, lowerUn, upperUn, sandwichIndirect);

    Parameters parObj(obj.par, *transf);

    if (strcmp(*simfile, "") != 0) { // Write to simfile
      if (*print_level >= 1) {
	Rprintf("Writes to file simulated data corresponding to the indirect inference estimate\n");
      }
      vec ysim = indirectExtern->simulateData(parObj);
      writeData(ysim, *simfile);
    }

    // Extract mu, lambda, psi, omega
    mu[0] = parObj.mu;
    mu[1] = sd.mu;
    mu[2] = lower.mu;
    mu[3] = upper.mu;
    if (sandwichIndirect && 0) {
      mu[4] = sd2.mu;
      mu[5] = lower2.mu;
      mu[6] = upper2.mu;
    }

    psi[0] = parObj.psi;
    psi[1] = sd.psi;
    psi[2] = lower.psi;
    psi[3] = upper.psi;
    if (sandwichIndirect && 0) {
      psi[4] = sd2.psi;
      psi[5] = lower2.psi;
      psi[6] = upper2.psi;
    }

    int ind=0;

    for (int i=0;i<(*nSup);i++) {
      lambda[ind] = parObj.lambda(i);
      lambda[(*nSup)+ind] = sd.lambda(i);
      lambda[2*(*nSup)+ind] = lower.lambda(i);
      lambda[3*(*nSup)+ind] = upper.lambda(i);
      if (sandwichIndirect && 0) {
	lambda[4*(*nSup)+ind] = sd2.lambda(i);
	lambda[5*(*nSup)+ind] = lower2.lambda(i);
	lambda[6*(*nSup)+ind] = upper2.lambda(i);
      }

      omega[ind] = parObj.omega(i);
      omega[(*nSup)+ind] = sd.omega(i);
      omega[2*(*nSup)+ind] = lower.omega(i);
      omega[3*(*nSup)+ind] = upper.omega(i);
      if (sandwichIndirect && 0) {
	omega[4*(*nSup)+ind] = sd.omega(i);
	omega[5*(*nSup)+ind] = lower.omega(i);
	omega[6*(*nSup)+ind] = upper.omega(i);
      }

      ind++;
    }

    *nSimAll = obj.nSimAll;

    int k=0;
    for (int i=0;i<(*nSim);i++) {
      Parameters parSim(obj.parsim.col(i), *transf);
      muSim[i] = parSim.mu;
      psiSim[i] = parSim.psi;
      for (int j=0;j<(*nSup);j++) {
	lambdaSim[k] = parSim.lambda(j);
	omegaSim[k] = parSim.omega(j);
	k++;
      }
    }

    for (int i=0;i<(*nSimAll);i++) {
      funcval[i] = obj.funcvalAll(i);
      niter[i] = obj.iterAll(i);
      if (obj.funcvalAll(i) > (*ftol_weak))
	convergence[i] = 0;
      else if (obj.funcvalAll(i) > (*ftol))
	convergence[i] = 1;
      else
	convergence[i] = 2;
    }

    *nFuncEval = Optimise::nFuncEval;
    *nGradEval = Optimise::nGradEval;
    *nFuncEvalOuter = Optimise::nFuncEvalOuter;

    if (*print_level >= 2) {
      Rprintf("nFuncEval %d nGradEval %d nFuncEvalOuter %d\n", *nFuncEval, *nGradEval, *nFuncEvalOuter);
    }
  }

  //  Simulation study for univariate data
  void SimulationStudy(int * nRep, int * methods,
		       int * nSup, int * nSimIndirect,
		       int * nObs,
		       int * nTimes, double * par,
		       double * mu, double * psi,
		       double * lambda, double * omega,
		       int * covmu, int * covpsi,
		       int * covlambda, int * covomega,
		       double * funcvals, int * iters,
		       int * nsimIndTot,
		       double * minlambda, double * maxlambda,
		       int * nFuncEval, int * nGradEval,
		       int * nFuncEvalOuter,
		       double * ftol, double * ftol_weak,
		       double * gradtol,
		       int * transf,
		       int * addPenalty,
		       int * print_level,
		       int * useRoptimiser,
		       int * initialSteepestDescent,
		       int * ITMAX,
		       int * error,
		       char ** savefile, char ** savefile2,
		       int * sandwich,
		       int * sandwichIndirect,
		       int * writeSimDataToFile,
		       double * gradMax)
  {
    Simulate simObs(*nSup, *nObs, *print_level);
    const int useParVec = 0;
    const int resetSeed = 0;
    const double deltaT = 1.0;
    const int useStartPar = 0; // Not in use??

    vec y = zeros<vec>(1); // dummy
    const int saveDraws = 0;
    indirectExtern = new Indirect(y, *nSimIndirect, *nSup, *nTimes, useStartPar,
				  *minlambda, *maxlambda, *ftol, *ftol_weak, *gradtol,
				  *transf, *addPenalty, *print_level,
				  *useRoptimiser, *gradMax, *initialSteepestDescent, *ITMAX, saveDraws);
    qlExtern = indirectExtern;
    const vec parvec = SetParVec(par, *mu, *psi, lambda, omega, *nSup,
				 useParVec, *transf);
    Parameters pex(parvec, *transf);
    Parameters truepar(parvec, NOTRANSF);
    
    ivec updatePars = ones<ivec>(parvec.n_elem);
    const int npar = parvec.n_elem;

    const int iQL = 0;
    const int iII = 1;
    int nMethods=0;
    const int nMaxMethods=2;
    for (int i=0;i<nMaxMethods;i++) {
      if (methods[i] == 1)
	nMethods++;
    }

    int nExtra = 0;
    if (methods[iII]) {
      nExtra = 0; // 1;
    }

    mat mumat(*nRep, 2*nMethods+nExtra);
    mat psimat(*nRep, 2*nMethods+nExtra);
    mat lambdamat(*nRep, (2*nMethods+nExtra)*(*nSup));
    mat omegamat(*nRep, (2*nMethods+nExtra)*(*nSup));
    imat coverageMu(*nRep, nMethods+nExtra);
    imat coveragePsi(*nRep, nMethods+nExtra);
    imat coverageLambda(*nRep, (nMethods+nExtra)*(*nSup));
    imat coverageOmega(*nRep, (nMethods+nExtra)*(*nSup));
    imat itervec(*nRep, *nSimIndirect);
    mat funcvalvec(*nRep, *nSimIndirect);
    ivec nSimAll(*nRep);

    ofstream ost(*savefile);
    ost.precision(8);
    ofstream ost2(*savefile2);
    ost2.precision(8);

    for (int i=0; i<(*nRep); i++) {
      Rprintf("Replicate %d ========\n\n", i);
      // Simulate data, write to file
      simObs.simulateInit(); // Simulates new epsilon. Sets new seed
      vec s2;
      // Simulates nObs log-returns observations
      vec ysim = simObs.simulate(pex.mu, pex.lambda, pex.psi,
				 pex.omega, *nObs, deltaT, resetSeed, s2);
      if (*writeSimDataToFile) {
	char nc[20];
	sprintf(nc, "%d", i);
	char filename[80];
	strcpy(filename,"sim_tmp");
	strcat(filename, nc);
	strcat(filename,".dat");

	writeData(ysim, filename);
      }
      qlExtern->setDataLogReturns(ysim);
      qlExtern->setNObs(ysim.n_elem);

      simObs.cleanup();

      Parameters sd(*nSup);
      Parameters lowerUn(*nSup);
      Parameters upperUn(*nSup);
      Parameters lower(*nSup);
      Parameters upper(*nSup);

      int ind1=0;
      int ind2=0;
      int ind3=0;
      int ind4=0;

      EstimationObject objQL;
      EstimationObject objII;
      mat HiQL;
      mat grQL;
      vec sdQL_transf;
       if (methods[iQL]) {
	// QL
	objQL = qlExtern->optimise(parvec,
				   *gradtol,
				   *print_level,
				   updatePars,
				   func);
      }
      if (methods[iII]) {
	objII = indirectExtern->indirectInference(parvec);
	if (objII.status == EXIT_FAILURE) {
	  Rprintf("Try a new replicate %d\n", i);
	  i--;
	  continue;
	}
       }

      if (methods[iQL]) {
	Parameters pexQL(objQL.par, *transf);
	pexQL.print("QL estimated");
	qlExtern->confidenceIntervals(objQL.par, objQL.H, HiQL, sd, lower, upper, lowerUn, upperUn, *sandwich);

	// Estimates
	mumat(i,ind1) = pexQL.mu;
	psimat(i,ind1++) = pexQL.psi;
	lambdamat.submat(i,0, i, *nSup-1) = trans(pexQL.lambda);
	omegamat.submat(i,0, i, *nSup-1) = trans(pexQL.omega);
	ind2 += *nSup;

	// SD
	mumat(i,ind1) = sd.mu;
	psimat(i,ind1++) = sd.psi;
	lambdamat.submat(i,ind2, i, ind2+(*nSup)-1) = trans(sd.lambda);
	omegamat.submat(i,ind2, i, ind2+(*nSup)-1) = trans(sd.omega);
	ind2 += *nSup;
     
	// Coverage
	coverageMu(i,ind3) = (truepar.mu >= lowerUn.mu) && (truepar.mu <= upperUn.mu);
	coveragePsi(i,ind3++) = (truepar.psi >= lowerUn.psi) && (truepar.psi <= upperUn.psi);

	for (int j=0;j<*nSup;j++) {
	  coverageLambda(i,j) = (truepar.lambda(j) >= lowerUn.lambda(j)) && (truepar.lambda(j) <= upperUn.lambda(j));
	  coverageOmega(i,j) = (truepar.omega(j) >= lowerUn.omega(j)) && (truepar.omega(j) <= upperUn.omega(j));
	}
	ind4 += *nSup;
      }

      if (methods[iII]) {
	indirectExtern->confidenceIntervalsIndirect(objII, sd, lower, upper, lowerUn, upperUn, *sandwichIndirect);

	Parameters pexII(objII.par, *transf);
	pexII.print("II estimate");

	// Estimates
	mumat(i,ind1) = pexII.mu;
	psimat(i,ind1++) = pexII.psi;
	lambdamat.submat(i,ind2, i, ind2+(*nSup)-1) = trans(pexII.lambda);
	omegamat.submat(i,ind2, i, ind2+(*nSup)-1) = trans(pexII.omega);
	ind2 += *nSup;

	// SD
	mumat(i,ind1) = sd.mu;
	psimat(i,ind1++) = sd.psi;
	lambdamat.submat(i,ind2, i, ind2+(*nSup)-1) = trans(sd.lambda);
	omegamat.submat(i,ind2, i, ind2+(*nSup)-1) = trans(sd.omega);
	ind2 += *nSup;

  
	// Coverage
	coverageMu(i,ind3) = (truepar.mu >= lowerUn.mu) && (truepar.mu <= upperUn.mu);
	coveragePsi(i,ind3++) = (truepar.psi >= lowerUn.psi) && (truepar.psi <= upperUn.psi);

	for (int j=0;j<*nSup;j++) {
	  coverageLambda(i,ind4+j) = (truepar.lambda(j) >= lowerUn.lambda(j)) && (truepar.lambda(j) <= upperUn.lambda(j));
	  coverageOmega(i,ind4+j) = (truepar.omega(j) >= lowerUn.omega(j)) && (truepar.omega(j) <= upperUn.omega(j));
	}
	ind4 += *nSup;
 
	funcvalvec.row(i) = trans(objII.funcval);
	nSimAll(i) = objII.nSimAll;


	ost2 << i << " ";
	ost2 << nSimAll(i)  << " ";
	for (int j=0;j<nSimAll(i);j++) ost2 << objII.funcvalAll(j) << " ";
	for (int j=0;j<nSimAll(i);j++) ost2 << objII.iterAll(j) << " ";
	ost2 << endl;
      }
      // Save to file
      ost << i << " ";
      for (int j=0;j<2*nMethods;j++) ost << mumat(i, j) << " ";
      for (int j=0;j<2*nMethods;j++) ost << psimat(i, j) << " ";
      for (int j=0;j<2*nMethods*(*nSup);j++) ost << lambdamat(i, j) << " ";
      for (int j=0;j<2*nMethods*(*nSup);j++) ost << omegamat(i, j) << " ";
      for (int j=0;j<nMethods;j++) ost << coverageMu(i,j) << " ";
      for (int j=0;j<nMethods;j++) ost << coveragePsi(i,j) << " ";
      for (int j=0;j<nMethods*(*nSup);j++) ost << coverageLambda(i,j) << " ";
      for (int j=0;j<nMethods*(*nSup);j++) ost << coverageOmega(i,j) << " ";
      for (int j=0;j<*nSimIndirect;j++) ost << funcvalvec(i,j) << " ";
      for (int j=0;j<*nSimIndirect;j++) ost << itervec(i,j) << " ";
      //      for (int j=0;j<*nSimIndirect;j++) ost << niter(i,j) << " ";
      ost << nSimAll(i) << " ";
      if (methods[iQL]) {
	const mat Hi_uncorr = inv(objQL.H);
	for (int j=0;j<npar;j++) ost << sqrt(Hi_uncorr(j,j)) << " ";
	for (int j=0;j<npar;j++) ost << sqrt(HiQL(j,j)) << " ";
      }
      if (methods[iII]) {
	for (int j=0;j<*nSimIndirect;j++) for (int k=0;k<npar;k++) ost << objII.parsim(k,j) << " ";
      }
      ost << endl;
    }
    ost.close();
    ost2.close();

    int ind1=0;
    int ind2=0;
    int ind3=0;
    int ind4=0;
    int ind5=0;
    for (int i=0;i<*nRep;i++) {
      for (int j=0;j<2*nMethods;j++) {
	mu[ind1] = mumat(i,j);
	psi[ind1++] = psimat(i,j);
      }
      for (int j=0;j<nMethods;j++) {
	covmu[ind2] = coverageMu(i,j);
	covpsi[ind2++] = coveragePsi(i,j);
      }
      for (int j=0;j<2*nMethods*(*nSup);j++) {
	lambda[ind3] = lambdamat(i,j);
	omega[ind3++] = omegamat(i,j);
      }
      for (int j=0;j<nMethods*(*nSup);j++) {
	covlambda[ind4] = coverageLambda(i,j);
	covomega[ind4++] = coverageOmega(i,j);
      }
      for (int j=0;j<*nSimIndirect;j++) {
	funcvals[ind5] = funcvalvec(i,j);
	iters[ind5++] = itervec(i,j);
      }
      nsimIndTot[i] = nSimAll(i);
    }
  }


  void test(double * par, int * npar,
	    int * nFuncEvalOuter,
	    int * print_level,
	    int * initialSteepestDescent)
  {

    const int useStartPar = 0; // Not in use??
    const double gradMax = 1000*1000;
    const int useRoptimiser = 0;
    const double minlambda = 0.1;
    const double maxlambda = 1.0;
    const int transf = 1;
    const double gradtol = 1e-4;
    const double ftol = 1e-4;
    const double ftol_weak = 1e-3;
    const int ITMAX = 200;
    const int nSim = 1;
    const int nSup = 1;
    const int addPenalty = 0;
    const int nTimes = 1;
    vec y = zeros<vec>(2);
    // nTimes --> nTimes - 1 ???????
    //    qlExtern = new QL(y, *minlambda, *maxlambda, *transf, *addPenalty);
    const int saveDraws = 0;
    indirectExtern = new Indirect(y, nSim, nSup, nTimes, useStartPar,
				  minlambda, maxlambda, ftol, ftol_weak, gradtol,
				  transf, addPenalty, *print_level,
				  useRoptimiser, gradMax, *initialSteepestDescent, ITMAX, saveDraws);
    qlExtern = indirectExtern;

    vec startpar(*npar);
    for (int i=0;i<(*npar);i++) {
      startpar(i) = par[i];
    }
    if (*print_level >= 2) {
      startpar.print_trans("startpar=");
    }

    Optimise::nFuncEvalOuter = 0;

    indirectExtern->test(startpar);
    
    *nFuncEvalOuter = Optimise::nFuncEvalOuter;

    if (*print_level >= 1) {
      Rprintf("nFuncEvalOuter %d\n", *nFuncEvalOuter);
    }

    for (int i=0;i<(*npar);i++) {
      par[i] = startpar(i);
    }

  }
} // End extern "C"

vec ReadData(string filename) {
  ifstream ist(filename.c_str());

  if (!ist) {
    //    error("Can't open input file", filename);
    cout << "Can't open input file " << filename << endl;
    exit(-1);
  }

  int index = 0;
  int nSize=1000;
  int nDeltaSize = 1000;

  vec y(nSize);
  double y0;
  while(ist >> y0) {
    y(index++) = y0;

    if (index >= nSize) {
      nSize += nDeltaSize;
      vec ycopy = y.rows(0, index-1);
      y.set_size(nSize);
      y.rows(0, index-1) = ycopy;
    }
  }
  if (0) {
    vec ysub = y.rows(0,10);
    ysub.print_trans("y=");
  }

  return y.rows(0, index-1);
}

mat ReadDataMulti(string filename) {
  ifstream ist(filename.c_str());

  if (!ist) {
    //    error("Can't open input file", filename);
    cout << "Can't open input file " << filename << endl;
    exit(-1);
  }

  int index = 0;
  int nSize=1000;
  int nDeltaSize = 1000;

  const int nvar=2;
  mat y(nSize, nvar);
  double y0;
  //  ist >> y0;
  int nZeroLines = 0;
  while(ist >> y0) {
    y(index,0) = y0;
    double y1;
    ist >> y1;
    y(index,1) = y1;
    if (y0 != 0.0 && y1 != 0.0) {
      index++;
    }
    else {
      nZeroLines++;
      continue;
    }
    //    if (index < 10)
    //      cout << y(index-1) << endl;
    if (index >= nSize) {
      mat tmp = y.row(index-1);
      nSize += nDeltaSize;

      mat ycopy = y.rows(0, index-1);
      y.set_size(nSize,nvar);
      y.rows(0, index-1) = ycopy;
    }
  }

  if (0) {
    mat ysub = y.rows(0,10);
    ysub.print_trans("y=");
    Rprintf("Number of lines removed: %d\n", nZeroLines);
  }

  return y.rows(0, index-1); // Possible memory leak;
}

vec SetParVec(double * par, double mu, double psi, double * lambda, double * omega,
	      int nSup, int useParVec, int transf) {
  int dim = 2*nSup+2;
  vec parvec;
  if (useParVec) {
    parvec = zeros<vec>(dim);
    for (int i=0;i<dim;i++) {
      parvec(i) = par[i];
    }
  }
  else {
    vec lambdavec(nSup);
    vec omegavec(nSup);
    for (int i=0;i<nSup;i++) {
      lambdavec(i) = lambda[i];
      omegavec(i) = omega[i];
    }
    Parameters par(mu, psi, omegavec, lambdavec, transf);
    parvec = par.extractParsInv(transf);
  }

  return parvec;
}

vec SetParVecMulti(double * par, double * mu, double * psi, double * lambda, double * omega,
		   double phi21, int nSup, int useParVec, int transf) {
  vec parvec;
  const int p=ParametersMulti::p;
  const int q=ParametersMulti::q;
  const int p_q = p+q;
  if (useParVec) {
    int npar = q + q+p + 2*(q+p)*nSup+1; // mu, psi, lambda, omega, phi21
    parvec = zeros<vec>(npar);
    for (int i=0;i<npar;i++) {
      parvec(i) = par[i];
    }
  }
  else {
    vec muvec(q);
    vec psivec(p_q);
    mat lambdamat(p_q, nSup);
    mat omegamat(p_q, nSup);
    int ind=0;
    for (int k=0;k<q;k++) {
      muvec(k) = mu[k];
    }
    for (int k=0;k<p_q;k++) {
      psivec(k) = psi[k];
      for (int i=0;i<nSup;i++) {
	lambdamat(k,i) = lambda[ind];
	omegamat(k,i) = omega[ind++];
      }
    }
    ParametersMulti par(muvec, psivec, omegamat, lambdamat, phi21, transf);
    parvec = par.extractParsInv(transf);
  }

  return parvec;
}
