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
void ConfidenceIntervals(const vec & estimate, const mat & Hi,
			 Parameters & sd,
			 Parameters & lower, Parameters & upper,
			 const int transf);
void ConfidenceIntervalsMulti(const vec & estimate, const mat & Hi,
			      ParametersMulti & sd,
			      ParametersMulti & lower, ParametersMulti & upper,
			      const int transf);
mat ComputeSandwichMatrix(mat & H, mat & gr);

//
extern "C" {

  void CheckContinuity(char ** datfile, int * nSup,
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
    vec y0 = ReadData(*datfile);

    int nTimes = y0.n_elem;
    //vec y = y0.rows(1, nTimes-1);
    vec y = y0;
    nTimes = y.n_elem;
   //  y.print("y=");

    if (*print_level >= 1) {
      Rprintf("Number of observations: %d\n", y.n_elem);
    }

    if (*nObs < nTimes && *nObs > 0) {
      if (*print_level >= 1) {
	Rprintf("Truncate series to the %d last observations ", *nObs);
      }
      y = y.rows(nTimes-(*nObs),nTimes-1);
      if (*print_level >= 1) {
	Rprintf("Number of observations: %d\n", y.n_elem);
      }
      nTimes = y.n_elem;
    }

    const int useStartPar = 0; // Not in use??

    const int nSim=1;
    const double ftol = 0.1;
    const double ftol_weak = 1.0;
    indirectExtern = new Indirect(y, nSim, *nSup, nTimes, useStartPar,
				  *minlambda, *maxlambda, ftol, ftol_weak, *gradtol,
				  *transf, *addPenalty, *print_level, *useRoptimiser,
				  *initialSteepestDescent);
    qlExtern = indirectExtern;

    const vec startpar = SetParVec(par, *mu, *psi, lambda, omega, *nSup,
				   *useParVec, *transf);
    if (*print_level >= 2) {
      startpar.print("startpar=");
      Parameters par_debug(startpar, *transf);
      par_debug.print();
    }

    mat funcvals;
    mat xvals;
    mat xvals_transf;
    indirectExtern->checkContinuity(startpar, *nEval, *delta, indpar, xvals, xvals_transf, funcvals);

    const int npar = funcvals.n_rows;
    const int m = funcvals.n_cols;

    int index=0;
    for (int i=0;i<npar;i++) {
      for (int j=0;j<m;j++) {
	xOut[index] = xvals(i,j);
	xOut_transf[index] = xvals_transf(i,j);
	fOut[index++] = funcvals(i,j);
      }
    }
    //    Rprintf("index= %d\n", index);
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
    sim.simulateInit(); // Simulates new epsilon. Sets new seed
    const vec logYRet = sim.simulate(pex.mu, pex.lambda, pex.psi,
				     pex.omega, *nTimes, deltaT, resetSeed, s2);



    if (logYRet.n_elem == 0) {
      return;
    }

    for (int i=0;i<(*nTimes);i++) {
      s2_out[i] = s2(i);
      logYRet_out[i] = logYRet(i);
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
		       double * H,
		       int * nFuncEval, int * nGradEval,
		       double * gradtol, int * nObs,
		       int * transf, int * useParVec,
		       int * addPenalty, int * checkGrad,
		       int * print_level,
		       int * useRoptimiser,
		       int * updatePars_,
		       int * sandwich)
  {
    vec y0 = ReadData(*datfile);

    int n = y0.n_elem;
    vec y = y0;
    //  y.print("y=");
    n = y.n_elem;

    if (*print_level >= 1) {
      Rprintf("Number of observations: %d\n", y.n_elem);
      //      Rprintf("Transf %d\n", *transf);
      //      Rprintf("Useparvec %d\n", *useParVec);
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

    qlExtern = new QL(y, *minlambda, *maxlambda, *transf, *addPenalty, *useRoptimiser);


    const vec parvec = SetParVec(par, *mu, *psi, lambda, omega, *nSup,
				 *useParVec, *transf);

    if (*print_level >= 1) {
      parvec.print("Start parameters=");
      Parameters par_debug(parvec, *transf);
      par_debug.print();
    }
    //    Rprintf("Before qlExtern->optimise\n");

    int npar = parvec.n_elem;
    ivec updatePars(npar);

    if (*checkGrad) {
      updatePars = ones<ivec>(npar); // check gradient for all parameters
      qlExtern->setUpdates(updatePars, parvec);
      qlExtern->checkGradient(&func, parvec, 1e-4, 1e-2, 1);
    }

    //    Rprintf("QuasiLikelihood check: func %6.4f\n", func(parvec, 0).f);

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
    // Extract mu, lambda, psi, omega
    Parameters sd(*nSup);
    Parameters lower(*nSup);
    Parameters upper(*nSup);
    mat Hi;
    if (*sandwich) {
      //      qlExtern->checkGradient(&func, obj.par, 1e-6, 1e-2, 1);

      mat gr = qlExtern->quasiLikelihood_individual(obj.par);
      //      vec gr_mean = mean(gr, 1);
      //      gr_mean.print("gr_mean=");
      //      mat gr_var = var(gr, 0, 1);
      //      gr_var.print("gr_var=");

      Hi = ComputeSandwichMatrix(obj.H, gr);
      //      Hi.print("Hi=");
    }
    else {
      Hi = inv(obj.H);
    }
    // Extract confidence intervals for parameters
    ConfidenceIntervals(obj.par, Hi, sd, lower, upper, *transf);
    

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
    for (int i=0;i<npar;i++) {
      for (int j=0;j<npar;j++) {
	H[ind++] = obj.H(i,j);
      }
    }
  
    //    Parameters parObj(obj.par, *transf);
    //    *mu = parObj.mu;
    //    *psi = parObj.psi;
    //    for (int i=0;i<(*nSup);i++) {
    //      lambda[i] = parObj.lambda(i);
    //      omega[i] = parObj.omega(i);
    //    }
    
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
			    int * sandwich)
  {
    mat y = ReadDataMulti(*datfile);

    ParametersMulti::p = 1;
    ParametersMulti::q = 2;

    const int p_q = ParametersMulti::p + ParametersMulti::q;

    int n = y.n_rows;
    //    vec y = y0;
    //  y.print("y=");
    //    n = y.n_elem;

    if (*print_level >= 1) {
      Rprintf("Number of observations: %d\n", n);
      //      Rprintf("Transf %d\n", *transf);
      //      Rprintf("Useparvec %d\n", *useParVec);
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

    
    vec minlambdavec(p_q);
    vec maxlambdavec(p_q);
    for (int k=0;k<p_q;k++) {
      minlambdavec(k) = minlambda[k];
      maxlambdavec(k) = maxlambda[k];
    }
    qlExtern = new QL(y, minlambdavec, maxlambdavec, *transf, *addPenalty, *useRoptimiser);


    const vec parvec = SetParVecMulti(par, mu, psi, lambda, omega, *phi21,
				      *nSup, *useParVec, *transf);

    if (*print_level >= 1) {
      cout << "Start parameters (transformed): " << trans(parvec);
      ParametersMulti par_debug(parvec, *transf);
      cout << "Start parameters (original scale): " << endl;
      par_debug.print();
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

    ParametersMulti sd(*nSup);
    ParametersMulti lower(*nSup);
    ParametersMulti upper(*nSup);
    mat Hi;
     if (*sandwich) {
       //      qlExtern->checkGradient(&funcMulti, obj.par, 1e-6, 1e-2, 1);

      mat gr = qlExtern->quasiLikelihoodMulti_individual(obj.par);
      //      vec gr_mean = mean(gr, 1);
      //      gr_mean.print("gr_mean=");
      //      mat gr_var = var(gr, 0, 1);
      //      gr_var.print("gr_var=");

      Hi = ComputeSandwichMatrix(obj.H, gr);
      //      Hi.print("Hi=");
    }
    else {
      Hi = inv(obj.H);
    }
     // Extract confidence intervals for parameters
    ConfidenceIntervalsMulti(obj.par, Hi, sd, lower, upper, *transf);

    // Extract confidence intervals for parameters
    

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
			 int * nSimAll)
  {
    vec y0 = ReadData(*datfile);

    int nTimes = y0.n_elem;
    vec y = y0;//.rows(1, nTimes-1);
    //  y.print("y=");

    if (*print_level >= 1) {
      Rprintf("Number of observations: %d\n", y.n_elem);
    }
    if (*nObs < nTimes && *nObs > 0) {
      if (*print_level >= 1) {
	Rprintf("Truncate series to the %d last observations ", *nObs);
      }
      y = y.rows(nTimes-(*nObs),nTimes-1);
      if (*print_level >= 1) {
	Rprintf("Number of observations: %d\n", y.n_elem);
      }
    }

    const int useStartPar = 0; // Not in use??

    indirectExtern = new Indirect(y, *nSim, *nSup, nTimes, useStartPar,
				  *minlambda, *maxlambda, *ftol, *ftol_weak, *gradtol,
				  *transf, *addPenalty, *print_level,
				  *useRoptimiser, *initialSteepestDescent);
    qlExtern = indirectExtern;

    const vec startpar = SetParVec(par, *mu, *psi, lambda, omega, *nSup,
				   *useParVec, *transf);
   if (*print_level >= 2) {
     startpar.print("startpar=");
     Parameters par_debug(startpar, *transf);
     par_debug.print();
   }

    Optimise::nFuncEval = 0;
    Optimise::nGradEval = 0;
    Optimise::nFuncEvalOuter = 0;

    EstimationObject obj = indirectExtern->indirectInference(startpar);

    if (obj.status == EXIT_FAILURE) {
      if (*print_level >= 1) {
	Rprintf("Indirect inference aborted\n");
      }
      return;
    }

    // Extract mu, lambda, psi, omega
    Parameters parObj(obj.par, *transf);
    *mu = parObj.mu;
    *psi = parObj.psi;
    for (int i=0;i<(*nSup);i++) {
      lambda[i] = parObj.lambda(i);
      omega[i] = parObj.omega(i);
    }

    *nSimAll = obj.nSimAll;

    int k=0;
    for (int i=0;i<(*nSimAll);i++) {
      Parameters parSim(obj.parsim.col(i), *transf);
      muSim[i] = parSim.mu;
      psiSim[i] = parSim.psi;
      for (int j=0;j<(*nSup);j++) {
	lambdaSim[k] = parSim.lambda(j);
	omegaSim[k] = parSim.omega(j);
	k++;
      }
    }

    *nFuncEval = Optimise::nFuncEval;
    *nGradEval = Optimise::nGradEval;
    *nFuncEvalOuter = Optimise::nFuncEvalOuter;

    if (*print_level >= 2) {
      Rprintf("nFuncEval %d nGradEval %d nFuncEvalOuter %d\n", *nFuncEval, *nGradEval, *nFuncEvalOuter);
    }
  }

  void SimulationStudy(int * nRep, int * methods,
		       int * nSup, int * nSimIndirect,
		       int * nTimes,
		       double * mu, double * psi,
		       double * lambda, double * omega,
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
		       int * error)
  {
    Rprintf("SimulationStudy not yet implemented\n");
 //    for (int i=0; i<(*nRep); i++) {
      // Simulate data, write to file

      // QL + Indirect inference

      // Keep estimates
  //      muvec(i) = 0.0;
  //      psivec(i) = 0.0;
  //     lambdamat.row(i) = 0.0;
  //     omegamat.row(i) = 0.0;
  //     funcval(i) = 0.0;
  //   }
  
  }


  void test(double * par, int * npar,
	   int * nFuncEvalOuter,
	   int * print_level,
	   int * initialSteepestDescent)
  {

    const int useStartPar = 0; // Not in use??
    const int useRoptimiser = 0;
    const double minlambda = 0.1;
    const double maxlambda = 1.0;
    const int transf = 1;
    const double gradtol = 1e-4;
    const double ftol = 1e-4;
    const double ftol_weak = 1e-3;
    const int nSim = 1;
    const int nSup = 1;
    const int addPenalty = 0;
    const int nTimes = 1;
    vec y = zeros<vec>(2);
    // nTimes --> nTimes - 1 ???????
    //    qlExtern = new QL(y, *minlambda, *maxlambda, *transf, *addPenalty);
    indirectExtern = new Indirect(y, nSim, nSup, nTimes, useStartPar,
				  minlambda, maxlambda, ftol, ftol_weak, gradtol,
				  transf, addPenalty, *print_level,
				  useRoptimiser, *initialSteepestDescent);
    qlExtern = indirectExtern;

    vec startpar(*npar);
    for (int i=0;i<(*npar);i++) {
      startpar(i) = par[i];
    }
    if (*print_level >= 2) {
      startpar.print("startpar=");
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
  //  ist >> y0;
  while(ist >> y0) {
    y(index++) = y0;
    //    if (index < 10)
    //      cout << y(index-1) << endl;
    if (index >= nSize) {
      nSize += nDeltaSize;
      vec ycopy = y.rows(0, index-1);
      y.set_size(nSize);
      y.rows(0, index-1) = ycopy;
    }
	
    //    ist >> y0;
  }
  if (0) {
    vec ysub = y.rows(0,10);
    ysub.print("y=");
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
    ysub.print("y=");
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
    
    // Print parameters
    //    par.print();
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

void ConfidenceIntervals(const vec & estimate, const mat & Hi,
			 Parameters & sd,
			 Parameters & lower, Parameters & upper,
			 const int transf) {
  const double q0_975 = 1.959964;
  //  mat Hi = inv(H);
  const vec sd_transf = sqrt(Hi.diag());
  vec lowervec = estimate - q0_975*sd_transf;
  vec uppervec = estimate + q0_975*sd_transf;

  lower.setPars(lowervec, transf);
  upper.setPars(uppervec, transf);

  const int npar = estimate.n_elem;
  const int nsup = Parameters::numberOfSuperPositions(estimate);
  Parameters x(nsup);
  const int nsim=10000;
  vec sumx = zeros<vec>(npar);
  vec sumx2 = zeros<vec>(npar);
  GetRNGstate();
  for (int i=0;i<nsim;i++) {
    // Draw random normal numbers with mean=estimate, sd=sd_transf
    const vec eps = Simulate::normal(npar);
    const vec u = estimate + eps % sd_transf;

    // Transform u
    x.setPars(u, transf);
    const vec xvec = x.asvector();
    sumx = sumx + xvec;
    sumx2 = sumx2 + xvec%xvec;
  }
  PutRNGstate();
  vec varvec = (sumx2 - (1.0/nsim)*sumx%sumx)/(nsim-1.0);
  //  varvec.print("sdvec=");
  vec sdvec = sqrt(varvec);
  //  sdvec.print("sdvec=");
  sd.setPars(sdvec, NOTRANSF);
  //  sd.print();
}

void ConfidenceIntervalsMulti(const vec & estimate, const mat & Hi,
			      ParametersMulti & sd,
			      ParametersMulti & lower, ParametersMulti & upper,
			      const int transf) {
  const double q0_975 = 1.959964;
  //  mat Hi = inv(H);
  const vec sd_transf = sqrt(Hi.diag());
  vec lowervec = estimate - q0_975*sd_transf;
  vec uppervec = estimate + q0_975*sd_transf;

  const int npar = estimate.n_elem;
  const int nsup = ParametersMulti::numberOfSuperPositions(estimate);
  ParametersMulti x(nsup);
  const int nsim=1000;
  vec sumx = zeros<vec>(npar);
  vec sumx2 = zeros<vec>(npar);
  GetRNGstate();
  for (int i=0;i<nsim;i++) {
    // Draw random normal numbers with mean=estimate, sd=sd_transf
    const vec eps = Simulate::normal(npar);
    const vec u = estimate + eps % sd_transf;

    // Transform u
    x.setPars(u, transf);
    const vec xvec = x.asvector();
    sumx = sumx + xvec;
    sumx2 = sumx2 + xvec%xvec;
  }
  PutRNGstate();
  vec varvec = (sumx2 - (1.0/nsim)*sumx%sumx)/(nsim-1.0);
  //  varvec.print("sdvec=");
  vec sdvec = sqrt(varvec);
  //  sdvec.print("sdvec=");
  sd.setPars(sdvec, NOTRANSF);
  //  sd.print();

  lower.setPars(lowervec, transf);
  upper.setPars(uppervec, transf);
}

mat ComputeSandwichMatrix(mat & H, mat & gr) {
  const double gradMax = 1000.0;
  rowvec grt = sum(gr % gr,0);
  int ap = gr.n_cols;
  int ind=0;
  for (int t=0;t<ap;t++) {
    if (grt(t) < gradMax) {
      gr.col(ind++) = gr.col(t);
    }
    //    else {
    //      Rprintf("grt(%d)=%6.4f\n", t, grt(t));
    //    }
  }

  //  Rprintf("ind=%d, ap=%d\n", ind, ap);
  gr = gr.cols(0, ind-1);
  double mult = ((double) ap)/((double) ind);
  gr = mult * gr;

  mat I = gr * trans(gr);
  mat Hi = inv(H);
  
  mat S = Hi*I*Hi;

  return S;
}
