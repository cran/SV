/*Include Files:*/
//#include <iostream>

#include <R.h>
#include <R_ext/Applic.h>

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
#include "simulate.h"
#include "ql.h"

//#define MAX(A,B) ((A) > (B) ? (A) : (B))

const double Inf = 1000000000;
vec grad_Ropt;

#define MAX2(A) ((A) > (0) ? (A)*(A) : (0))
#define MAX3(A) ((A) > (0) ? (A)*(A)*(A) : (0))

typedef double optimfn(int n, double *par, void *ex);
typedef void optimgr(int n, double *par, double *gr, void *ex);

FunctionValue func(const vec & par, const int evaluateGradient) {
  Optimise::nFuncEval++;
  Optimise::nGradEval += evaluateGradient;
  //  Rprintf("func:\n");
  //  par.print("par=");
  return qlExtern->quasiLikelihood(par, evaluateGradient);
}
FunctionValue funcMulti(const vec & par, const int evaluateGradient) {
  Optimise::nFuncEval++;
  Optimise::nGradEval += evaluateGradient;
  //  Rprintf("func:\n");
  //  par.print("par=");
  return qlExtern->quasiLikelihoodMulti(par, evaluateGradient);
}

double func_toRopt(int n, double *par, void * ex) {
  int evaluateGradient = 1;
  Optimise::nFuncEval++;
  Optimise::nGradEval += evaluateGradient;
  vec parvec = zeros<vec>(n);
  for (int i=0;i<n;i++)
    parvec(i) = par[i];
  //  int evaluateGradient = reinterpret_cast<int> (clientData);
  FunctionValue fval = qlExtern->quasiLikelihood(parvec, evaluateGradient);
  grad_Ropt = fval.df;
  return fval.f;
}

void grad_toRopt(int n, double *par, double *gr, void *ex) {
  for (int i=0;i<n;i++) {
    gr[i] = grad_Ropt(i);
  }
}

//QL::QL(const double minlambda_, const double maxlambda_, int useRoptimiser_) {
//  Parameters::minlambda = minlambda_;
//  Parameters::maxlambda = maxlambda_;
//  useRoptimiser = useRoptimiser_;

//  penaltyMin = -7.0;
//  penaltyMax = 3.0;
//}

QL::QL(const vec & y, const double minlambda_, const double maxlambda_,
       const int transf_, const int addPenalty_, int useRoptimiser_,
       const double gradMax_, const int nSup_, const int nTimes_, const int print_level_, const int saveDraws_) : Simulate(nSup_, nTimes_, print_level_, saveDraws_), gradMax(gradMax_) {
  setData(y);
  nObs = Z0.n_cols;

  Parameters::minlambda = minlambda_;
  Parameters::maxlambda = maxlambda_;
  useRoptimiser = useRoptimiser_;

  transf = transf_;
  addPenalty = addPenalty_;

  if (transf == 0) {
    penaltyMin = -7.0;
    penaltyMax = 3.0;
  }
  else {
    penaltyMin = -3.0;
    penaltyMax = 3.0;
  }
}

QL::QL(const vec & y, const double minlambda_, const double maxlambda_,
       const int transf_, const int addPenalty_, int useRoptimiser_,
       const double gradMax_) : Simulate(1, 0, 0), gradMax(gradMax_) {
  setData(y);
  nObs = Z0.n_cols;

  Parameters::minlambda = minlambda_;
  Parameters::maxlambda = maxlambda_;
  useRoptimiser = useRoptimiser_;

  transf = transf_;
  addPenalty = addPenalty_;

  if (transf == 0) {
    penaltyMin = -7.0;
    penaltyMax = 3.0;
  }
  else {
    penaltyMin = -3.0;
    penaltyMax = 3.0;
  }
}

// Multidimensional y
QL::QL(const mat & y, const vec minlambda_, const vec maxlambda_,
       const int transf_, const int addPenalty_, int useRoptimiser_,
       const double gradMax_, const int p_, const int q_,
       const int nSup_, const int nTimes_, const int print_level_, const int saveDraws_) :  Simulate(p_, q_, nSup_, nTimes_, print_level_, saveDraws_), gradMax(gradMax_)  {
  setDataMulti(y);
  nObs = Z0.n_cols;

  ParametersMulti::minlambda = minlambda_;
  ParametersMulti::maxlambda = maxlambda_;
  useRoptimiser = useRoptimiser_;

  transf = transf_;
  addPenalty = addPenalty_;

  if (transf == 0) {
    penaltyMin = -5.0;
    penaltyMax = 3.0;
  }
  else {
    penaltyMin = -3.0;
    penaltyMax = 3.0;
  }
}

// Multidimensional y
QL::QL(const mat & y, const vec minlambda_, const vec maxlambda_,
       const int transf_, const int addPenalty_, int useRoptimiser_,
       const double gradMax_) : Simulate(1, 0, 0), gradMax(gradMax_)  {
  setDataMulti(y);
  nObs = Z0.n_cols;

  ParametersMulti::minlambda = minlambda_;
  ParametersMulti::maxlambda = maxlambda_;
  useRoptimiser = useRoptimiser_;

  transf = transf_;
  addPenalty = addPenalty_;

  if (transf == 0) {
    penaltyMin = -5.0;
    penaltyMax = 3.0;
  }
  else {
    penaltyMin = -3.0;
    penaltyMax = 3.0;
  }
}

QL::~QL() {
}

void QL::setData(const vec & y) {
  int ny = y.n_elem;

  int nr = ny-1;

  Z0.set_size(2,nr);
  for (int j=0;j<nr;j++) {
    Z0(0,j) = 100*(log(y(j+1)) - log(y(j)));
    Z0(1,j) = Z0(0,j) * Z0(0,j);
  }
}

void QL::setDataMulti(const mat & y) {
  int nr = y.n_rows;
  int nc = y.n_cols;

  nr = nr-1;

  Z0.set_size(2*nc,nr);
  int ind = 0;
  for (int j=0;j<nr;j++) {
    vec ydiff(2);
    int eqZero=0;
    for (int l=0;l<nc;l++) {
      ydiff(l) = 100*(log(y(j+1,l)) - log(y(j,l)));
      if (ydiff(l) == 0.0) {
	eqZero = 1;
	break;
      }
    }
    if (!eqZero) {
      for (int l=0;l<nc;l++) {
	Z0(2*l,ind) = ydiff(l);
	Z0(2*l+1,ind) = ydiff(l)*ydiff(l);
      }
      ind++;
    }
  }
  Z0 = Z0.cols(0, ind-1); // Possible memory leak
  if (0) {
    mat Z0sub = Z0.cols(0,11);
    Z0sub.print("Z0 (first)=");
    Rprintf("Z0: %d:%d\n", Z0.n_rows, Z0.n_cols);
    const int ap = Z0.n_cols;
    Z0sub = Z0.cols(ap-12,ap-1);
    Z0sub.print("Z0 (last)=");
    vec Z0mean = mean(Z0, 1);
    Z0mean.print_trans("Z0mean=");
  }
}

void QL::setDataLogReturns(const vec & yret) {
  int ny = yret.n_elem;

  int nr = ny;

  Z0.set_size(2,nr);
  for (int j=0;j<nr;j++) {
    Z0(0,j) = yret(j); //100*(log(y(j+1)) - log(y(j)));
    Z0(1,j) = Z0(0,j) * Z0(0,j);
  }
}

void QL::setUpdates(const ivec updatePars_, const vec par_) {
  updatePars = updatePars_;
  parFull = par_;
}

vec QL::setReducedPar(vec par0) {
  const int nparfull = par0.n_elem;
  const int npar = sum(updatePars);
  vec par(npar);
  int index=0;
  for (int i=0;i<nparfull;i++) {
    if (updatePars(i)) {
      par(index++) = par0(i);
    }
  }

  return par;
}

vec QL::setFullPar(vec parReduced) {
  int npar = updatePars.n_elem;
  vec par(npar);
  int index=0;
  for (int i=0;i<npar;i++) {
    if (updatePars(i)) {
      par(i) = parReduced(index++);
    }
    else {
      par(i) = parFull(i);
    }
  }
  return par;
}

EstimationObject QL::optimise(const vec & startpar,
			      const double gradtol,
			      const int print_level,
			      const ivec & updatePars_,
			      FunctionValue (*funcToOpt)(const vec &, const int)) {
  int status = EXIT_SUCCESS;
  mat H;
  //  const int ownOpt = 0;

  //  Rprintf("Enter QL::optimise\n");

  // Set parameters to be estimated
  setUpdates(updatePars_, startpar);
  vec par = setReducedPar(startpar);

  const int npar = par.n_elem;

  if (print_level >= 1) {
    Rprintf("Start parameters\n");
    for (int i=0;i<npar;i++) {
      Rprintf(" %6.4f ", par(i));
    }
    Rprintf("\n");
  }
  
  int nIter=0;
  if (!useRoptimiser) {
    status = opt.bfgs(funcToOpt, par, print_level, gradtol, H, nIter);
    //    H.print("H after bfgs:");
  }
  else {
    double * x = new double[npar];
    for (int i=0;i<npar;i++)
      x[i] = par(i);
    double Fmin;
    int maxit = 10000;
    int trace=6;

    int * mask = (int *) R_alloc(npar, sizeof(int));
    for (int i = 0; i < npar; i++) mask[i] = 1;

    double abstol = -Inf;
    double reltol= 1e-8;
    int nREPORT = 1;
    void *ex = 0;
    int fncount;
    int grcount;
    int fail;
    vmmin(npar, x, &Fmin, func_toRopt, grad_toRopt, maxit, trace, mask,
	  abstol, reltol, nREPORT, ex, &fncount, &grcount, &fail); // Does not return the hessian matrix
	  
    for (int i=0;i<npar;i++)
      par(i) = x[i];
    if (print_level >= 1) {
      Rprintf("fncount %d   grcount %d\n", fncount, grcount);
    }

    delete [] x;
  }

  vec par2 = setFullPar(par);
  EstimationObject res = EstimationObject(par2, startpar, H, gradtol, status, nIter);

  //  Rprintf("Quit QL::optimise\n");
  return res;
}


FunctionValue QL::quasiLikelihood(const vec & parReduced, const int evaluateGradient) {
  mat a, a_;
  mat V, V_;
  double f;

  mat b;
  mat W;
  mat Bt;

  mat Qu;
  mat sigma;
  mat fii;

  vec df;

  //  Rprintf("Enter QL:quasiLikelihood\n");

  vec par = setFullPar(parReduced);

  const int checkPars=0;
  Parameters pex(par, transf, checkPars);
  if (pex.checkPars(transf)) {
    f = Inf;
    const int npar = par.n_elem;
    df = 1000.0 * ones<vec>(npar);
    df = setReducedPar(df);
    FunctionValue fval = FunctionValue(f, df);
    return fval;
  }

  //  Rprintf("QL:quasiLikelihood check: mean(Z0(0,)) %6.4f var(Z0(0,)) %6.4f\n", mean(Z0.row(0)), var(Z0.row(0)));

  //  Rprintf("Enter QL::quasiLikelihood\n");
  //  par.print("par=");
  filter(par, Z0, a, a_, V, V_, Qu, sigma, fii, f);

  if (evaluateGradient) {
    if (1) { //if (is.finite(filter$f)) {
      smoother(par, a, a_, V, V_, b, W, Bt);
      
      df = gradient(par, b, W, Bt, Qu, sigma, fii);
      //      df.print("df=");
      if (0) {//(any(is.na(df))) {
	df.print_trans("df =");
	stop_own("Error in gradient");
      }
    }
    else {
      //    df = rep(-Inf, length(par))
    }
  }

  df = setReducedPar(df);
  FunctionValue fval = FunctionValue(f, df);

  return fval;
}

mat QL::quasiLikelihood_individual(const vec & parReduced) {
  mat a, a_;
  mat V, V_;
  double f;

  mat b;
  mat W;
  mat Bt;

  mat Qu;
  mat sigma;
  mat fii;

  vec df;
  mat gr;

  vec par = setFullPar(parReduced);
  filter(par, Z0, a, a_, V, V_, Qu, sigma, fii, f);
  smoother(par, a, a_, V, V_, b, W, Bt);
      
  df = gradientIndividual(par, b, W, Bt, Qu, sigma, fii, gr);

  df = setReducedPar(df);

  return gr;
}

FunctionValue QL::quasiLikelihoodMulti(const vec & parReduced, const int evaluateGradient) {
  mat a, a_;
  mat V, V_;
  double f;

  mat b;
  mat W;
  mat Bt;

  mat Qu;
  mat sigma;
  mat fii;

  vec tau;
  mat gama;
  mat Gi;
  mat F1;

  vec df;

  const int debug=0;

  if (debug) Rprintf("Enter QL:quasiLikelihoodMulti\n");

  vec par = setFullPar(parReduced);

  const int checkPars=0;
  ParametersMulti pex(par, transf, checkPars);
  if (pex.checkPars(transf)) {
    f = Inf;
    const int npar = par.n_elem;
    df = 1000.0 * ones<vec>(npar);
    df = setReducedPar(df);
    FunctionValue fval = FunctionValue(f, df);
    return fval;
  }

  if (debug) Rprintf("QL:quasiLikelihoodMulti check: mean(Z0(0,)) %6.4f var(Z0(0,)) %6.4f\n", mean(Z0.row(0)), var(Z0.row(0)));

  //  Rprintf("Enter QL::quasiLikelihood\n");
  //par.print("par=");
  filter_multivariat(par, Z0, a, a_, V, V_, Qu, sigma, fii, tau, gama, Gi, F1, f);

  if (evaluateGradient) {
    if (1) { //if (is.finite(filter$f)) {
      smoother_multivariate(par, a, a_, V, V_, fii, b, W, Bt);
      
      df = gradient_multivariat(par, b, W, Bt, Qu, sigma, fii, Gi, F1, tau, gama);
      //      df.print("df=");
      if (0) {//(any(is.na(df))) {
	df.print_trans("df =");
	stop_own("Error in gradient");
      }
    }
    else {
      //    df = rep(-Inf, length(par))
    }
  }

  df = setReducedPar(df);
  FunctionValue fval = FunctionValue(f, df);

  if (debug) Rprintf("Quit QL::quasiLikelihoodMulti\n");

  //exit(-1);

  return fval;
}

mat QL::quasiLikelihoodMulti_individual(const vec & parReduced) {
  mat a, a_;
  mat V, V_;
  double f;

  mat b;
  mat W;
  mat Bt;

  mat Qu;
  mat sigma;
  mat fii;

  vec tau;
  mat gama;
  mat Gi;
  mat F1;

  vec df;
  mat gr;

  const int debug=0;

  if (debug) Rprintf("Enter QL:quasiLikelihoodMulti_individual\n");

  vec par = setFullPar(parReduced);

  if (debug) Rprintf("QL:quasiLikelihoodMulti_individual check: mean(Z0(0,)) %6.4f var(Z0(0,)) %6.4f\n", mean(Z0.row(0)), var(Z0.row(0)));

  filter_multivariat(par, Z0, a, a_, V, V_, Qu, sigma, fii, tau, gama, Gi, F1, f);
  smoother_multivariate(par, a, a_, V, V_, fii, b, W, Bt);
      
  df = gradient_multivariat_individual(par, b, W, Bt, Qu, sigma, fii, Gi, F1, tau, gama, gr);
 
  df = setReducedPar(df);

  if (debug) Rprintf("Quit QL::quasiLikelihoodMulti_individual\n");

  return gr;
}


vec QL::gradient(const vec& par,
		 const mat & b, const mat & W, const mat & BBt,
		 const mat & Qu, const mat & sigma, const mat & fii) {
  //  Rprintf("Enter QL::gradient\n");
  Parameters par_extract(par, transf);
  double mu = par_extract.mu;
  vec lambda = par_extract.lambda;
  double psi = par_extract.psi;
  vec omega = par_extract.omega;

  const int nSup = (par.n_elem - 2)/2;
  const int ap = Z0.n_cols;
  const int nEq = Z0.n_rows;
  int nSup_fullmodel; // number of terms with exact state space model representation
  if (nSup == 1)
    nSup_fullmodel = nSup;
  else
    nSup_fullmodel = nSup - 1;
  const int nSup_approx = nSup - nSup_fullmodel;
  const int nLatent = 2*nSup_fullmodel + nSup_approx;
  vec expMinusLambda = exp(-lambda);
  vec expMinusTwoLambda = exp(-2.0*lambda);
  vec oneMinusExpMinusLambda = 1 - expMinusLambda;
  vec oneMinusExpMinusTwoLambda = 1 - expMinusTwoLambda;

  vec tau(nEq); //= "mu accu(psi)";
  tau(0)= mu;
  tau(1) = psi + mu*mu;

  mat gama = zeros<mat>(nEq, nLatent);
  for (int i=0;i<nSup_fullmodel;i++) {
    gama(1,2*i) = 1;
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    gama(1,i+nSup_fullmodel) = 1;
  }

  //  mat Qu = zeros<mat>(nLatent, nLatent);
  /*
  for (int i=0;i<nSup_fullmodel;i++) {
    Qu(2*i,2*i) = 2*omega(i)*(-1.5-0.5*expMinusTwoLambda(i) + 2*expMinusLambda(i) + lambda(i));
    Qu(2*i,2*i+1) = 2*omega(i)*(oneMinusExpMinusLambda(i) -0.5*oneMinusExpMinusTwoLambda(i));
    Qu(2*i+1,2*i) = Qu(2*i,2*i+1);
    Qu(2*i+1,2*i+1) = 2*omega(i)*oneMinusExpMinusTwoLambda(i);
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    Qu(nSup_fullmodel+i, nSup_fullmodel+i) = 2*omega(i)*lambda(i);
  }
  

  */
  const vec lambda2 = lambda % lambda;
  const vec lambda3 = lambda2 % lambda;

  mat GG_ = gama;


  //  Qu.print("Qu=");
  
  mat iQu = zeros<mat>(nLatent, nLatent);
  for (int i=0;i<nSup_fullmodel;i++) {
    int i0 = 2*i;
    int i1 = 2*i+1;
    mat Qu_sub = Qu.submat(i0,i0,i1,i1);
    //    Qu_sub.print("Qu_sub=");
    iQu.submat(i0,i0,i1,i1) = inv(Qu_sub);
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    int i0 = i+nSup_fullmodel;
    iQu(i0,i0) = 1/Qu(i0,i0);
  }
  mat isigma = inv(sigma);
  
  //  iQu.print("iQu=");

  //  isigma.print("isigma=");
  //  iQu.print("iQu=");
  //  tau.print("tau=");
  //  gama.print("gama=");
  //  ##derG = 0;
  
  vec dertau = zeros<vec>(nEq);
  vec derSigma = zeros<vec>(nEq*nEq);
  vec * derQu = new vec[nSup];
  mat * derfii = new mat[nSup];
  int dim_sub = 2;
  for (int i=0;i<nSup;i++) {
    if (i == nSup_fullmodel)
      dim_sub = 1;
    derQu[i] = zeros<vec>(dim_sub*dim_sub);
    derfii[i] = zeros<mat>(dim_sub,dim_sub);
  }

  //  Rprintf("Start loop\n");
  //  Rprintf("ap=%d\n", ap);
  for (int t=0;t<ap;t++) {//do until t>ap;
    //    Rprintf("t=%d\n", t);
    int t2 = nLatent*(t-1) - 1;
    int t1 = nLatent*t - 1;
    int t0 = nLatent*(t+1) - 1;
    
    
    vec Rt = Z0.col(t) - tau - GG_ * b.col(t);

    mat WW = GG_ * W.cols(t1+1,t0) * trans(GG_);

    mat Ht =  Rt * trans(Rt) + WW;

    dertau = dertau + isigma * Rt; // check the below line!
    derSigma = derSigma + 0.5*(-vectorize(isigma) + kronecker(isigma, isigma) * vectorize(Ht));

    if (t>0) {
      mat VBt = W.cols((t1+1),t0) * trans(BBt.cols(t1+1,t0));
      mat Wt_ = W.cols(t2+1,t1);
      mat Wt = W.cols(t1+1,t0);

      Ht =  (b.col(t) - fii * b.col(t-1))  *  trans(b.col(t) - fii * b.col(t-1)) +
	Wt - fii * trans(VBt) - VBt * trans(fii) + fii * Wt_ * trans(fii);

      int inc = 2; // dimension of sub model
      int i0 = 0;
      for (int i=0;i<nSup;i++) { // loop over full model + approx model super pos terms
	if (i==nSup_fullmodel) // turn to approx model
	  inc = 1;
	int i1 = i0+inc-1;

	//	mat b_out01 = outer(b.col(t), b.col(t-1));
	//	mat b_out11 = outer(b.col(t-1), b.col(t-1));
	mat b_out01 = outer(b.submat(i0,t,i1, t), b.submat(i0,t-1,i1,t-1));
	mat b_out11 = outer(b.submat(i0,t-1,i1, t-1), b.submat(i0,t-1,i1,t-1));

	mat iQu_sub = iQu.submat(i0,i0, i1, i1);
	mat VBt_sub = VBt.submat(i0,i0, i1, i1);
	mat fii_sub = fii.submat(i0,i0, i1, i1);
	mat Wt_sub = Wt_.submat(i0,i0, i1, i1);
	//	fii_sub.print("fii_sub=");
	derfii[i] = derfii[i] + iQu_sub * (b_out01 + VBt_sub - fii_sub*(b_out11 + Wt_sub));
	//	derfii[i].print("derfii[i]=");
	mat Ht_sub = Ht.submat(i0,i0,i1,i1);
	derQu[i] = derQu[i] + 0.5*(-vectorize(iQu_sub) + kronecker(iQu_sub,iQu_sub)* vectorize(Ht_sub));
	//	derQu[i].print("derQu[i]=");

	i0 += inc;
      }
    }
  }
  //  dertau.print_trans("dertau=");
  //  derSigma.print("derSigma=");

  // diagonalelementene i deriverte mhp. Qu
  //  vec derfii_diag = derfii.diag();     // diagonalelementene i deriverte mhp. Fii 

  //  for (int i=0;i<nSup;i++) { // loop over full model + approx model super po
  //    derfii[i].print("derfii=");
  //    derQu[i].print("derQu=");
  //  }

  mat d_tau = zeros<mat>(2*nSup + 2, 2);
  d_tau(0,0) = 1;
  d_tau(0,1) = 2*mu;
  d_tau(nSup+1, 1) = 1; // wrt eps

  //  d_tau.print("d_tau=");

  double twoSumPsi = 2*psi;

  rowvec d_Sig_mu = zeros<rowvec>(nEq*nEq);
  d_Sig_mu(1) = twoSumPsi;
  d_Sig_mu(2) = twoSumPsi;
  d_Sig_mu(3) = 4*mu*twoSumPsi;

  //  d_Sig_mu.print("d_Sig_mu=");

  vec * d_fii_lam = new vec[nSup];
  dim_sub = 2;
  for (int i=0;i<nSup;i++) {
    if (i == nSup_fullmodel)
      dim_sub = 1;
    d_fii_lam[i] = zeros<vec>(dim_sub);

    if (dim_sub == 1) {
      d_fii_lam[i](0) = -expMinusLambda(i);
    }
    else {
      d_fii_lam[i](0) = (1/lambda(i))*expMinusLambda(i) -
	(1/lambda2(i))*oneMinusExpMinusLambda(i);
      d_fii_lam[i](1) = -expMinusLambda(i);
    }
    //    d_fii_lam[i].print("d_fii_lam[i]=");
  }

  
  vec * d_Q_lam = new vec[nSup];
  dim_sub = 2;
  for (int i=0;i<nSup;i++) {
    if (i == nSup_fullmodel)
      dim_sub = 1;

    d_Q_lam[i] = zeros<vec>(dim_sub*dim_sub);
    if (dim_sub == 1) {
      d_Q_lam[i](0) = 2*omega(i);
    }
    else {
      d_Q_lam[i](0) = -(2/lambda3(i))*(-1.5 - 0.5*expMinusTwoLambda(i) + 2*expMinusLambda(i) + lambda(i)) + (1/lambda2(i))*(expMinusTwoLambda(i) - 2*expMinusLambda(i) + 1);
      d_Q_lam[i](1) = -(1/lambda2(i))*(oneMinusExpMinusLambda(i)-0.5*oneMinusExpMinusTwoLambda(i)) + (1/lambda(i))*(expMinusLambda(i) - expMinusTwoLambda(i));
      d_Q_lam[i](2) = d_Q_lam[i](1);
      d_Q_lam[i](3) = expMinusTwoLambda(i);

      d_Q_lam[i] = 2*omega(i) * d_Q_lam[i];

    }
    //    d_Q_lam[i].print("d_Q_lam[i]=");
  }

  mat d_Sig_lam = zeros<mat>(nSup,nEq*nEq);
  d_Sig_lam.col(3) = -8*(omega/lambda3) % (expMinusLambda-1+lambda)
    + 4*(omega/lambda2) % oneMinusExpMinusLambda;
  //  d_Sig_lam.print("d_Sig_lam=");

  rowvec d_Sig_psi(nEq*nEq);
  d_Sig_psi(0) = 1;
  d_Sig_psi(1) = 2*mu;
  d_Sig_psi(2) = 2*mu;
  d_Sig_psi(3) = 4*mu*mu + 4*psi;

  //  d_Sig_psi.print("d_Sig_psi=");

  //  d_Q_omega = diag(2*lambda, nrow=nSup, ncol=nSup);
  vec * d_Q_omega = new vec[nSup];
  int i0 = 0;
  dim_sub = 2;
  for (int i=0;i<nSup;i++) {
    if (i == nSup_fullmodel)
      dim_sub = 1;
    int i1 = i0+dim_sub-1;
    
    mat Qu_sub = Qu.submat(i0, i0, i1, i1);
    d_Q_omega[i] = (1/omega[i])*vectorize(Qu_sub);
    i0 += dim_sub;
    //    d_Q_omega[i].print("d_Q_omega=");
  }

  mat d_Sig_omega = zeros<mat>(nSup, 4);
  d_Sig_omega.col(3) = 4*(expMinusLambda -1+lambda)/lambda2;
  //  d_Sig_omega.print("d_Sig_omega=");

  mat cA_ = zeros<mat>(nSup, nSup);
  if (transf == 1 && nSup == 2) {
    cA_(0,0) = (lambda(0)-Parameters::minlambda)*(1-(lambda(0)-Parameters::minlambda)/(Parameters::maxlambda-Parameters::minlambda));
    cA_(1,1) = lambda(1)*(1-lambda(1)/Parameters::minlambda);
  }
  else {
    vec lambdaStar(nSup);
    for (int i=0;i<nSup;i++) {
      lambdaStar(i) = (lambda(i)-Parameters::minlambda)/(Parameters::maxlambda-Parameters::minlambda);
    }
    for (int i=0;i<nSup;i++) {
      for (int j=i;j<nSup;j++) {
	cA_(i,j) = (Parameters::maxlambda-Parameters::minlambda)* lambdaStar(j) / (1 + exp(par(i+1)));
      }
    }
  }

  //  cA_.print("cA_=");

  vec gr = d_tau * dertau;
  //  d_Sig_psi.print("d_Sig_psi=");
  //  gr.print("gr=");
  gr(0) = as_scalar(gr(0) + (d_Sig_mu * derSigma));   // mhp. mu
  gr(1+nSup) = as_scalar(gr(1+nSup) + d_Sig_psi * derSigma); // mhp. psi

  //  gr.print("gr=");
  gr(1+nSup) = psi * gr(1+nSup);

  const vec sigma_prod_lambda = d_Sig_lam * derSigma;
  const vec sigma_prod_omega = d_Sig_omega * derSigma;
  dim_sub = 2;
  for (int i=0;i<nSup;i++) {
    if (i==nSup_fullmodel)
      dim_sub = 1;
    const vec derfii_sub = derfii[i].col(dim_sub-1);
    double tmp1 = accu(d_fii_lam[i] % derfii_sub);
    const double tmp2 = accu(d_Q_lam[i] % derQu[i]);
    const vec tmp3 = d_Sig_lam.row(i)*derSigma;
    gr(i+1) = gr(i+1) + tmp1 + tmp2 + sigma_prod_lambda(i); // wrt lambda
    tmp1 = accu(d_Q_omega[i] % derQu[i]);
    gr(2+nSup+i) = gr(2+nSup+i) + tmp1 + sigma_prod_omega(i);  // wrt omega
  }

  gr.rows(1, nSup) = cA_ * gr.rows(1,nSup);  // deriverte mhp par[2:nSup+1]
  
  gr.rows((2+nSup),(1+2*nSup)) = omega % gr.rows((2+nSup),(1+2*nSup));

  vec df = -1 * gr;
  //  df.print("df=");
  
  //  Rprintf("Quit QL::gradient\n");

  if (addPenalty) {
    //    if (nSup > 1) {
    const int indLambda = 1;
    for (int i=0;i<nSup;i++) {
      double pen = 300*(MAX2(par(indLambda+i)-penaltyMax)-MAX2(penaltyMin-par(indLambda+i)));
      if (pen != 0) {
	//	Rprintf("pen1 %6.4f pen2 %6.4f\n", pen1, pen2);
	//	par.print("par=");
	//	df.print("df=");
	df(indLambda+i) += pen;
	//	df.print("df=");
      }
    }
  }

  delete [] derQu;
  delete [] derfii;
  delete [] d_fii_lam;
  delete [] d_Q_lam;
  delete [] d_Q_omega;

  return df;
}

vec QL::gradientIndividual(const vec& par,
			   const mat & b, const mat & W, const mat & BBt,
			   const mat & Qu, const mat & sigma, const mat & fii,
			   mat & gr) {
  //  Rprintf("Enter QL::gradient\n");
  Parameters par_extract(par, transf);
  double mu = par_extract.mu;
  vec lambda = par_extract.lambda;
  double psi = par_extract.psi;
  vec omega = par_extract.omega;

  const int nSup = (par.n_elem - 2)/2;
  const int ap = Z0.n_cols;
  const int nEq = Z0.n_rows;
  int nSup_fullmodel; // number of terms with exact state space model representation
  if (nSup == 1)
    nSup_fullmodel = nSup;
  else
    nSup_fullmodel = nSup - 1;
  const int nSup_approx = nSup - nSup_fullmodel;
  const int nLatent = 2*nSup_fullmodel + nSup_approx;

  const int npar = par.n_elem;
  gr.set_size(npar, ap);

  vec expMinusLambda = exp(-lambda);
  vec expMinusTwoLambda = exp(-2.0*lambda);
  vec oneMinusExpMinusLambda = 1 - expMinusLambda;
  vec oneMinusExpMinusTwoLambda = 1 - expMinusTwoLambda;

  vec tau(nEq); //= "mu accu(psi)";
  tau(0)= mu;
  tau(1) = psi + mu*mu;

  mat gama = zeros<mat>(nEq, nLatent);
  for (int i=0;i<nSup_fullmodel;i++) {
    gama(1,2*i) = 1;
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    gama(1,i+nSup_fullmodel) = 1;
  }

  const vec lambda2 = lambda % lambda;
  const vec lambda3 = lambda2 % lambda;

  mat GG_ = gama;


  //  Qu.print("Qu=");
  
  mat iQu = zeros<mat>(nLatent, nLatent);
  for (int i=0;i<nSup_fullmodel;i++) {
    int i0 = 2*i;
    int i1 = 2*i+1;
    mat Qu_sub = Qu.submat(i0,i0,i1,i1);
    //    Qu_sub.print("Qu_sub=");
    iQu.submat(i0,i0,i1,i1) = inv(Qu_sub);
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    int i0 = i+nSup_fullmodel;
    iQu(i0,i0) = 1/Qu(i0,i0);
  }
  mat isigma = inv(sigma);
  
  //  iQu.print("iQu=");

  //  isigma.print("isigma=");
  //  iQu.print("iQu=");
  //  tau.print("tau=");
  //  gama.print("gama=");
  //  ##derG = 0;
  
  vec dertau = zeros<vec>(nEq);
  vec derSigma = zeros<vec>(nEq*nEq);
  vec * derQu = new vec[nSup];
  mat * derfii = new mat[nSup];
  int dim_sub = 2;
  for (int i=0;i<nSup;i++) {
    if (i == nSup_fullmodel)
      dim_sub = 1;
    derQu[i] = zeros<vec>(dim_sub*dim_sub);
    derfii[i] = zeros<mat>(dim_sub,dim_sub);
  }

  //  Rprintf("Start loop\n");
  //  Rprintf("ap=%d\n", ap);
  for (int t=0;t<ap;t++) {//do until t>ap;
    //    Rprintf("t=%d\n", t);
    int t2 = nLatent*(t-1) - 1;
    int t1 = nLatent*t - 1;
    int t0 = nLatent*(t+1) - 1;
    
    
    vec Rt = Z0.col(t) - tau - GG_ * b.col(t);

    mat WW = GG_ * W.cols(t1+1,t0) * trans(GG_);

    mat Ht =  Rt * trans(Rt) + WW;

    dertau = isigma * Rt; // check the below line!
    derSigma = 0.5*(-vectorize(isigma) + kronecker(isigma, isigma) * vectorize(Ht));

    if (t>0) {
      mat VBt = W.cols((t1+1),t0) * trans(BBt.cols(t1+1,t0));
      mat Wt_ = W.cols(t2+1,t1);
      mat Wt = W.cols(t1+1,t0);

      Ht =  (b.col(t) - fii * b.col(t-1))  *  trans(b.col(t) - fii * b.col(t-1)) +
	Wt - fii * trans(VBt) - VBt * trans(fii) + fii * Wt_ * trans(fii);

      int inc = 2; // dimension of sub model
      int i0 = 0;
      for (int i=0;i<nSup;i++) { // loop over full model + approx model super pos terms
	if (i==nSup_fullmodel) // turn to approx model
	  inc = 1;
	int i1 = i0+inc-1;

	//	mat b_out01 = outer(b.col(t), b.col(t-1));
	//	mat b_out11 = outer(b.col(t-1), b.col(t-1));
	mat b_out01 = outer(b.submat(i0,t,i1, t), b.submat(i0,t-1,i1,t-1));
	mat b_out11 = outer(b.submat(i0,t-1,i1, t-1), b.submat(i0,t-1,i1,t-1));

	mat iQu_sub = iQu.submat(i0,i0, i1, i1);
	mat VBt_sub = VBt.submat(i0,i0, i1, i1);
	mat fii_sub = fii.submat(i0,i0, i1, i1);
	mat Wt_sub = Wt_.submat(i0,i0, i1, i1);
	//	fii_sub.print("fii_sub=");
	derfii[i] = iQu_sub * (b_out01 + VBt_sub - fii_sub*(b_out11 + Wt_sub));
	//	derfii[i].print("derfii[i]=");
	mat Ht_sub = Ht.submat(i0,i0,i1,i1);
	derQu[i] = 0.5*(-vectorize(iQu_sub) + kronecker(iQu_sub,iQu_sub)* vectorize(Ht_sub));
	//	derQu[i].print("derQu[i]=");

	i0 += inc;
      }
    }

    //  dertau.print_trans("dertau=");
    //  derSigma.print("derSigma=");

    // diagonalelementene i deriverte mhp. Qu
    //  vec derfii_diag = derfii.diag();     // diagonalelementene i deriverte mhp. Fii 

    //  for (int i=0;i<nSup;i++) { // loop over full model + approx model super po
    //    derfii[i].print("derfii=");
    //    derQu[i].print("derQu=");
    //  }

    mat d_tau = zeros<mat>(2*nSup + 2, 2);
    d_tau(0,0) = 1;
    d_tau(0,1) = 2*mu;
    d_tau(nSup+1, 1) = 1; // wrt eps

    //  d_tau.print("d_tau=");

    double twoSumPsi = 2*psi;

    rowvec d_Sig_mu = zeros<rowvec>(nEq*nEq);
    d_Sig_mu(1) = twoSumPsi;
    d_Sig_mu(2) = twoSumPsi;
    d_Sig_mu(3) = 4*mu*twoSumPsi;

    //  d_Sig_mu.print("d_Sig_mu=");

    vec * d_fii_lam = new vec[nSup];
    dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i == nSup_fullmodel)
	dim_sub = 1;
      d_fii_lam[i] = zeros<vec>(dim_sub);

      if (dim_sub == 1) {
	d_fii_lam[i](0) = -expMinusLambda(i);
      }
      else {
	d_fii_lam[i](0) = (1/lambda(i))*expMinusLambda(i) -
	  (1/lambda2(i))*oneMinusExpMinusLambda(i);
	d_fii_lam[i](1) = -expMinusLambda(i);
      }
      //    d_fii_lam[i].print("d_fii_lam[i]=");
    }

  
    vec * d_Q_lam = new vec[nSup];
    dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i == nSup_fullmodel)
	dim_sub = 1;

      d_Q_lam[i] = zeros<vec>(dim_sub*dim_sub);
      if (dim_sub == 1) {
	d_Q_lam[i](0) = 2*omega(i);
      }
      else {
	d_Q_lam[i](0) = -(2/lambda3(i))*(-1.5 - 0.5*expMinusTwoLambda(i) + 2*expMinusLambda(i) + lambda(i)) + (1/lambda2(i))*(expMinusTwoLambda(i) - 2*expMinusLambda(i) + 1);
	d_Q_lam[i](1) = -(1/lambda2(i))*(oneMinusExpMinusLambda(i)-0.5*oneMinusExpMinusTwoLambda(i)) + (1/lambda(i))*(expMinusLambda(i) - expMinusTwoLambda(i));
	d_Q_lam[i](2) = d_Q_lam[i](1);
	d_Q_lam[i](3) = expMinusTwoLambda(i);

	d_Q_lam[i] = 2*omega(i) * d_Q_lam[i];

      }
      //    d_Q_lam[i].print("d_Q_lam[i]=");
    }

    mat d_Sig_lam = zeros<mat>(nSup,nEq*nEq);
    d_Sig_lam.col(3) = -8*(omega/lambda3) % (expMinusLambda-1+lambda)
      + 4*(omega/lambda2) % oneMinusExpMinusLambda;
    //  d_Sig_lam.print("d_Sig_lam=");

    rowvec d_Sig_psi(nEq*nEq);
    d_Sig_psi(0) = 1;
    d_Sig_psi(1) = 2*mu;
    d_Sig_psi(2) = 2*mu;
    d_Sig_psi(3) = 4*mu*mu + 4*psi;

    //  d_Sig_psi.print("d_Sig_psi=");

    //  d_Q_omega = diag(2*lambda, nrow=nSup, ncol=nSup);
    vec * d_Q_omega = new vec[nSup];
    int i0 = 0;
    dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i == nSup_fullmodel)
	dim_sub = 1;
      int i1 = i0+dim_sub-1;
    
      mat Qu_sub = Qu.submat(i0, i0, i1, i1);
      d_Q_omega[i] = (1/omega[i])*vectorize(Qu_sub);
      i0 += dim_sub;
      //    d_Q_omega[i].print("d_Q_omega=");
    }

    mat d_Sig_omega = zeros<mat>(nSup, 4);
    d_Sig_omega.col(3) = 4*(expMinusLambda -1+lambda)/lambda2;
    //  d_Sig_omega.print("d_Sig_omega=");

    mat cA_ = zeros<mat>(nSup, nSup);
    if (transf == 1 && nSup == 2) {
      cA_(0,0) = (lambda(0)-Parameters::minlambda)*(1-(lambda(0)-Parameters::minlambda)/(Parameters::maxlambda-Parameters::minlambda));
      cA_(1,1) = lambda(1)*(1-lambda(1)/Parameters::minlambda);
    }
    else {
      vec lambdaStar(nSup);
      for (int i=0;i<nSup;i++) {
	lambdaStar(i) = (lambda(i)-Parameters::minlambda)/(Parameters::maxlambda-Parameters::minlambda);
      }
      for (int i=0;i<nSup;i++) {
	for (int j=i;j<nSup;j++) {
	  cA_(i,j) = (Parameters::maxlambda-Parameters::minlambda)* lambdaStar(j) / (1 + exp(par(i+1)));
	}
      }
    }

    //  cA_.print("cA_=");

    gr.col(t) = d_tau * dertau;
    //  d_Sig_psi.print("d_Sig_psi=");
    //  gr.print("gr=");
    gr(0,t) = as_scalar(gr(0,t) + (d_Sig_mu * derSigma));   // mhp. mu
    gr(1+nSup,t) = as_scalar(gr(1+nSup,t) + d_Sig_psi * derSigma); // mhp. psi

    //  gr.print("gr=");
    gr(1+nSup,t) = psi * gr(1+nSup,t);

    const vec sigma_prod_lambda = d_Sig_lam * derSigma;
    const vec sigma_prod_omega = d_Sig_omega * derSigma;
    dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i==nSup_fullmodel)
	dim_sub = 1;
      const vec derfii_sub = derfii[i].col(dim_sub-1);
      double tmp1 = accu(d_fii_lam[i] % derfii_sub);
      const double tmp2 = accu(d_Q_lam[i] % derQu[i]);
      const vec tmp3 = d_Sig_lam.row(i)*derSigma;
      gr(i+1,t) = gr(i+1,t) + tmp1 + tmp2 + sigma_prod_lambda(i); // wrt lambda
      tmp1 = accu(d_Q_omega[i] % derQu[i]);
      gr(2+nSup+i,t) = gr(2+nSup+i,t) + tmp1 + sigma_prod_omega(i);  // wrt omega
    }

    gr.submat(1,t, nSup,t) = cA_ * gr.submat(1,t,nSup,t);  // deriverte mhp par[2:nSup+1]
  
    gr.submat((2+nSup),t,(1+2*nSup),t) = omega % gr.submat((2+nSup),t,(1+2*nSup),t);


    if (addPenalty) {
      const int indLambda = 1;
      for (int i=0;i<nSup;i++) {
	double pen = 300*(MAX2(par(indLambda+i)-penaltyMax)-MAX2(penaltyMin-par(indLambda+i)));
	if (pen != 0) {
	  gr(indLambda+i,t) -= pen/ap;
	}
      }
    }
  }

  vec df = zeros<vec>(par.n_elem); // = -1 * gr;
  for (int t=0;t<ap;t++) {//do until t>ap;
    df = df - gr.col(t);
  }
  //  df.print("df (should be zero)=");

  return df;
}

vec QL::gradient_multivariat(const vec & par,
			     const mat & b, const mat & W, const mat & BBt,
			     const mat & Qu, const mat & sigma, const mat & fii,
			     const mat & Gi, const mat & F1,
			     const vec & tau, const mat & gama) {
  const int debug=0;
  if (debug) Rprintf("Enter QL::gradient_multivariat\n");
  if (0) {
    exit(-1);
  }
  ParametersMulti par_extract(par, transf);
  const vec mu = par_extract.mu;
  const mat lambda = par_extract.lambda;
  const vec psi = par_extract.psi;
  const mat omega = par_extract.omega;
  const double phi21 = par_extract.phi(2,1);

  const int npar = par.n_elem;
  const int p = ParametersMulti::p;
  const int q = ParametersMulti::q;
  

  const int nSup = par_extract.lambda.n_cols;
  const int ap = Z0.n_cols;
  const int nEq = Z0.n_rows;
  const int nGamma = gama.n_cols;
  int nSup_fullmodel; // number of terms with exact state space model representation
  if (nSup == 1)
    nSup_fullmodel = nSup;
  else
    nSup_fullmodel = nSup - 1;
  const int nSup_approx = nSup - nSup_fullmodel;
  const int nLatent = 2*nSup_fullmodel + nSup_approx;
  const mat expMinusLambda = exp(-lambda);
  const mat expMinusTwoLambda = exp(-2.0*lambda);
  const mat oneMinusExpMinusLambda = 1 - expMinusLambda;
  const mat oneMinusExpMinusTwoLambda = 1 - expMinusTwoLambda;

  const mat lambda2 = lambda % lambda;
  const mat lambda3 = lambda2 % lambda;

  mat GG_ = gama;

  //diagrv(eye(2),psi[3]|(sumr(4*(omega[3,.]./lamda[3,.].^2).*(exp(-lamda[3,.])-1+lamda[3,.]))+2*psi[3]^2));
  mat Omeg1 = eye<mat>(2,2);
  Omeg1(0,0) = psi(2);
  const rowvec Omeg1_tmp1 = 4*(omega.row(2)/lambda2.row(2));
  const rowvec Omeg1_tmp2 =  lambda.row(2) -oneMinusExpMinusLambda.row(2);
  const rowvec Omeg1_tmp = Omeg1_tmp1 %  Omeg1_tmp2;
  //  double tmp = accu(4*(omega.row(2)/lambda2.row(2)) % (-oneMinusExpMinusLambda.row(2) + lambda.row(2)));
  Omeg1(1,1) = sum(Omeg1_tmp) + 2*psi(2)*psi(2);

  //  Qu.print("Qu=");
  
  mat iQu = zeros<mat>(nLatent*(q+p), nLatent*(q+p));
  for (int k=0;k<q+p;k++) {
     int I0 = k*nLatent;
     for (int i=0;i<nSup_fullmodel;i++) {
       int i0 = I0+2*i;
       int i1 = I0+2*i+1;
       mat Qu_sub = Qu.submat(i0,i0,i1,i1);
       //    Qu_sub.print("Qu_sub=");
       iQu.submat(i0,i0,i1,i1) = inv(Qu_sub);
     }
     for (int i=nSup_fullmodel;i<nSup;i++) {
       int i0 = I0+i+nSup_fullmodel;
       iQu(i0,i0) = 1/Qu(i0,i0);
     }
  }

  mat isigma = inv(sigma);
  
  if (debug) iQu.print("iQu=");

  //  isigma.print("isigma=");
  //  iQu.print("iQu=");
  //  tau.print("tau=");
  //  gama.print("gama=");
  //  ##derG = 0;
  
  vec dertau = zeros<vec>(nEq);
  vec derSigma = zeros<vec>(nEq*nEq);
  mat derG = zeros<mat>(2*q,nGamma);

  vec * derQu = new vec[nSup*(p+q)];
  mat * derfii = new mat[nSup*(p+q)];
  int ind=0;
  for (int k=0;k<q+p;k++) {
    int dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i == nSup_fullmodel)
	dim_sub = 1;
      derQu[ind] = zeros<vec>(dim_sub*dim_sub);
      derfii[ind++] = zeros<mat>(dim_sub,dim_sub);
    }
  }

  if (debug) Rprintf("Start loop\n");
  if (debug) Rprintf("ap=%d\n", ap);

  for (int t=0;t<ap;t++) {//do until t>ap;
    if (t==3 && debug)
      exit(-1);

    if (debug==2) Rprintf("t=%d\n", t);
    int t2 = nGamma*(t-1) - 1;
    int t1 = nGamma*t - 1;
    int t0 = nGamma*(t+1) - 1;
    
    
    vec Rt = Z0.col(t) - tau - GG_ * b.col(t);

    mat WW = GG_ * W.cols(t1+1,t0) * trans(GG_);

    mat Ht =  Rt * trans(Rt) + WW;

    dertau = dertau + isigma * Rt; // check the below line!
    derSigma = derSigma + 0.5*(-vectorize(isigma) + kronecker(isigma, isigma) * vectorize(Ht));

    derG = derG + isigma*(Rt*trans(b.col(t)) - GG_*W.cols(t1+1, t0));

    if (debug==2) dertau.print_trans("dertau=");
    if (debug==2) derSigma.print("derSigma=");
    if (debug==2) derG.print("derG=");

    if (t>0) {
      mat VBt = W.cols((t1+1),t0) * trans(BBt.cols(t1+1,t0));
      mat Wt_ = W.cols(t2+1,t1);
      mat Wt = W.cols(t1+1,t0);

      Ht =  (b.col(t) - fii * b.col(t-1))  *  trans(b.col(t) - fii * b.col(t-1)) +
	Wt - fii * trans(VBt) - VBt * trans(fii) + fii * Wt_ * trans(fii);

      ind = 0;
      for (int k=0;k<q+p;k++) {
	int inc = 2; // dimension of sub model
	int I0 = k*nLatent;
	int i0 = I0;
	for (int i=0;i<nSup;i++) { // loop over full model + approx model super pos terms
	  if (i==nSup_fullmodel) // turn to approx model
	    inc = 1;
	  int i1 = i0+inc-1;

	  //	mat b_out01 = outer(b.col(t), b.col(t-1));
	  //	mat b_out11 = outer(b.col(t-1), b.col(t-1));
	  mat b_out01 = outer(b.submat(i0,t,i1, t), b.submat(i0,t-1,i1,t-1));
	  mat b_out11 = outer(b.submat(i0,t-1,i1, t-1), b.submat(i0,t-1,i1,t-1));

	  mat iQu_sub = iQu.submat(i0,i0, i1, i1);
	  mat VBt_sub = VBt.submat(i0,i0, i1, i1);
	  mat fii_sub = fii.submat(i0,i0, i1, i1);
	  mat Wt_sub = Wt_.submat(i0,i0, i1, i1);
	  //	fii_sub.print("fii_sub=");
	  derfii[ind] = derfii[ind] + iQu_sub * (b_out01 + VBt_sub - fii_sub*(b_out11 + Wt_sub));
	  if (debug==2) Rprintf("(k,i)=(%d,%d)\n", k,i);
	  if (debug==2) derfii[ind].print("derfii[ind]=");
	  mat Ht_sub = Ht.submat(i0,i0,i1,i1);
	  derQu[ind] = derQu[ind] + 0.5*(-vectorize(iQu_sub) + kronecker(iQu_sub,iQu_sub)* vectorize(Ht_sub));
	  if (debug==2) derQu[ind].print("derQu[ind]=");
	  ind++;

	  i0 += inc;
	}
      }
    }
  }
  if (debug) Rprintf("Loop finished\n");
  if (debug) dertau.print_trans("dertau=");
  if (debug) derSigma.print("derSigma=");
  if (debug) derG.print("derG=");
  for (ind=0;ind<(p+q)*nSup;ind++) {
    if (debug) derfii[ind].print("derfii[ind]=");
    if (debug) derQu[ind].print("derQu[ind]=");
  }

  // diagonalelementene i deriverte mhp. Qu
  //  vec derfii_diag = derfii.diag();     // diagonalelementene i deriverte mhp. Fii 

  //  for (int i=0;i<nSup;i++) { // loop over full model + approx model super po
  //    derfii[i].print("derfii=");
  //    derQu[i].print("derQu=");
  //  }

  // d_tau_my=(1~2*my[1]~zeros(1,2))| (zeros(1,2)~1~2*my[2]);
  mat d_tau = zeros<mat>(npar, nEq);
  // wrt mu_1 and mu_2
  d_tau(0,0) = 1;
  d_tau(0,1) = 2*mu(0);
  d_tau(1,2) = 1;
  d_tau(1,3) = 2*mu(1);
  // wrt psi_1, psi_2, psi_3
  const int ind_psi = q + (p+q)*nSup;
  d_tau(ind_psi,1) = 1;
  d_tau(ind_psi+1,3) = 1;
  d_tau(ind_psi+2,1) = 1;
  d_tau(ind_psi+2,3) = phi21;
  // wrt phi21
  d_tau(npar-1, 3) = psi(2);

  if (debug) d_tau.print("d_tau=");

  vec twoSumPsi = 2*psi;

  mat d_Sig1_mu = zeros<mat>(2,nEq);
  mat d_Sig2_mu = zeros<mat>(2,nEq);
  d_Sig1_mu(0,1) = twoSumPsi(0);
  d_Sig1_mu(0,2) = twoSumPsi(0);
  d_Sig1_mu(0,3) = 4*mu(0)*twoSumPsi(0);

  d_Sig2_mu(1,1) = twoSumPsi(1);
  d_Sig2_mu(1,2) = twoSumPsi(1);
  d_Sig2_mu(1,3) = 4*mu(1)*twoSumPsi(1);

  rowvec d_Sig2_phi21 = zeros<rowvec>(nEq);
  d_Sig2_phi21(3) = 8*psi(1)*psi(2)*phi21;

  if (debug) d_Sig1_mu.print("d_Sig1_mu=");
  if (debug) d_Sig2_mu.print("d_Sig2_mu=");
  if (debug) d_Sig2_phi21.print("d_Sig2_phi21=");

  vec * d_fii_lam = new vec[(p+q)*nSup];
  ind = 0;
  for (int k=0;k<p+q;k++) {
    int dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i == nSup_fullmodel)
	dim_sub = 1;
      d_fii_lam[ind] = zeros<vec>(dim_sub);

      if (dim_sub == 1) {
	d_fii_lam[ind](0) = -expMinusLambda(k,i);
      }
      else {
	d_fii_lam[ind](0) = (1/lambda(k,i))*expMinusLambda(k,i) -
	  (1/lambda2(k,i))*oneMinusExpMinusLambda(k,i);
	d_fii_lam[ind](1) = -expMinusLambda(k,i);
      }
      if (debug) d_fii_lam[ind].print("d_fii_lam[ind]=");
      ind++;
    }
  }
  
  vec * d_Q_lam = new vec[(p+q)*nSup];
  ind = 0;
  for (int k=0;k<p+q;k++) {
    int dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i == nSup_fullmodel)
	dim_sub = 1;

      d_Q_lam[ind] = zeros<vec>(dim_sub*dim_sub);
      if (dim_sub == 1) {
	d_Q_lam[ind](0) = 2*omega(k,i);
      }
      else {
	d_Q_lam[ind](0) = -(2/lambda3(k,i))*(-1.5 - 0.5*expMinusTwoLambda(k,i) + 2*expMinusLambda(k,i) + lambda(k,i)) + (1/lambda2(k,i))*(expMinusTwoLambda(k,i) - 2*expMinusLambda(k,i) + 1);
	d_Q_lam[ind](1) = -(1/lambda2(k,i))*(oneMinusExpMinusLambda(k,i)-0.5*oneMinusExpMinusTwoLambda(k,i)) + (1/lambda(k,i))*(expMinusLambda(k,i) - expMinusTwoLambda(k,i));
	d_Q_lam[ind](2) = d_Q_lam[ind](1);
	d_Q_lam[ind](3) = expMinusTwoLambda(k,i);

	d_Q_lam[ind] = 2*omega(k,i) * d_Q_lam[ind];
      }
      if (debug) d_Q_lam[ind].print("d_Q_lam[ind]=");
      ind++;
    }
  }

  mat d_Sig1_lam = zeros<mat>(3*nSup,4); // check what is 4?
  mat d_Sig2_lam = zeros<mat>(3*nSup,4);

  //d_Sig1_lam=zeros(3*p__,1)~zeros(3*p__,1)~zeros(3*p__,1)~vec((-8*omega[1,.].*(lamda[1,.].^(-3)).*(exp(-lamda[1,.])-1+lamda[1,.])+4*omega[1,.].*(lamda[1,.].^(-2)).*(-exp(-lamda[1,.])+1) )|zeros(2,p__));  

  d_Sig1_lam.submat(0,3,nSup-1,3) = trans(-8*(omega.row(0)/lambda3.row(0)) % (lambda.row(0) - oneMinusExpMinusLambda.row(0))
					  + 4*(omega.row(0)/lambda2.row(0)) % oneMinusExpMinusLambda.row(0));
  //  d_Sig_lam.print("d_Sig_lam=");

  d_Sig2_lam.submat(nSup,3,2*nSup-1,3) = trans(-8*(omega.row(1)/lambda3.row(1)) % (lambda.row(1) - oneMinusExpMinusLambda.row(1))
					  + 4*(omega.row(1)/lambda2.row(1)) % oneMinusExpMinusLambda.row(1));




  mat d_Sig1_psi = zeros<mat>(p+q, nEq);
  mat d_Sig2_psi = zeros<mat>(p+q, nEq);
  d_Sig1_psi(0,0) = 1;
  d_Sig1_psi(0,1) = 2*mu(0);
  d_Sig1_psi(0,2) = 2*mu(0);
  d_Sig1_psi(0,3) = 4*mu(0)*mu(0) + 4*psi(0) + 4*psi(2);
  d_Sig1_psi(2,3) = 4*psi(0);

  d_Sig2_psi(1,0) = 1;
  d_Sig2_psi(1,1) = 2*mu(1);
  d_Sig2_psi(1,2) = 2*mu(1);
  d_Sig2_psi(1,3) = 4*mu(1)*mu(1) + 4*psi(1) + 4*psi(2)*phi21*phi21;
  d_Sig2_psi(2,3) = 4*psi(1)*phi21*phi21;

  //  d_Sig_psi.print("d_Sig_psi=");

  //  d_Q_omega = diag(2*lambda, nrow=nSup, ncol=nSup);
  vec * d_Q_omega = new vec[(p+q)*nSup];
  ind =0;
  for (int k=0;k<p+q;k++) {
    int I0 = k*nLatent;
    int i0 = I0;
    int dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i == nSup_fullmodel)
	dim_sub = 1;
      int i1 = i0+dim_sub-1;

      mat Qu_sub = Qu.submat(i0, i0, i1, i1);
      d_Q_omega[ind] = (1/omega(k,i))*vectorize(Qu_sub);
      i0 += dim_sub;
      if (debug) d_Q_omega[ind].print("d_Q_omega[ind]=");
      ind++;
    }
  }

  mat d_Sig1_omega = zeros<mat>(nSup*(p+q), 4);
  mat d_Sig2_omega = zeros<mat>(nSup*(p+q), 4);
  d_Sig1_omega.submat(0,3,nSup-1,3) = trans(4*(lambda.row(0) - oneMinusExpMinusLambda.row(0))/lambda2.row(0));
  d_Sig2_omega.submat(nSup,3,2*nSup-1,3) = trans(4*(lambda.row(1) - oneMinusExpMinusLambda.row(1))/lambda2.row(1));
  //  d_Sig_omega.print("d_Sig_omega=");

  
  mat cA_ = zeros<mat>((p+q)*nSup, (p+q)*nSup);
  for (int k=0;k<p+q;k++) {
    int I0 = k*nSup;
    if (transf == 1 && nSup == 2) {
      cA_(I0,I0) = (lambda(k,0)-ParametersMulti::minlambda(k))*(1-(lambda(k,0)-ParametersMulti::minlambda(k))/(ParametersMulti::maxlambda(k)-ParametersMulti::minlambda(k)));
      cA_(I0+1,I0+1) = lambda(k,1)*(1-lambda(k,1)/ParametersMulti::minlambda(k));
    }
    else {
      vec lambdaStar(nSup);
      for (int i=0;i<nSup;i++) {
	lambdaStar(i) = (lambda(k, i)-ParametersMulti::minlambda(k))/(ParametersMulti::maxlambda(k)-ParametersMulti::minlambda(k));
      }
      for (int i=0;i<nSup;i++) {
	for (int j=i;j<nSup;j++) {
	  cA_(I0+i,I0+j) = (ParametersMulti::maxlambda(k)-ParametersMulti::minlambda(k))* lambdaStar(j) / (1 + exp(par(I0+i+q)));
	}
      }
    }
  }

  if (debug) cA_.print("cA_=");
  
  vec d_Sig1(2*q);
  vec d_Sig2(2*q);
  d_Sig1(0) = derSigma(0);
  d_Sig1(1) = derSigma(1);
  d_Sig1(2) = derSigma(2*q);
  d_Sig1(3) = derSigma(2*q+1);
  d_Sig2(0) = derSigma(5*q);
  d_Sig2(1) = derSigma(5*q+1);
  d_Sig2(2) = derSigma(7*q);
  d_Sig2(3) = derSigma(7*q+1);

  if (debug) d_Sig2.print("d_Sig2=");
  
  //d_F1_my=(0~2~zeros(1,6))|(zeros(1,3)~2*fi21~zeros(1,4));
  //d_F1_fi21=(0~0~1~2*my[2]~zeros(1,3)~2*fi21);
  mat d_F1_mu = zeros<mat>(q,8); // 2=q? 8?
  rowvec d_F1_phi21 = zeros<rowvec>(8); //6=?
  
  d_F1_mu(0,1) = 2;
  d_F1_mu(1,3) = 2*phi21;
  d_F1_phi21(2) = 1;
  d_F1_phi21(3) = 2*mu(1);
  d_F1_phi21(7) = 2*phi21;

  if (debug) d_F1_phi21.print("d_F1_phi21=");

  mat d_Omeg1_psi = zeros<mat>(p+q,2); //3=p+q? //zeros(2,2)|(1~4*psi[3]);
  d_Omeg1_psi(2,0) = 1;
  d_Omeg1_psi(2,1) = 4*psi(2);
  //d_Omeg1_lam=zeros(3*p__,1)~vec(zeros(2,p__)| (-8*omega[3,.].*(lamda[3,.].^(-3)).*(exp(-lamda[3,.])-1+lamda[3,.])+4*omega[3,.].*(lamda[3,.].^(-2)).*(-exp(-lamda[3,.])+1) ));  

  //d_Omeg1_omega=zeros(3*p__,1)~vec(zeros(2,p__) | (4*(lamda[3,.].^(-2)).*(exp(-lamda[3,.])-1+lamda[3,.])) );

  mat d_Omeg1_lam = zeros<mat>((p+q)*nSup, 2);
  ind = 2*nSup; // 2=q?
  d_Omeg1_lam.submat(ind,1,ind+nSup-1,1) = trans(-8*omega.row(2) % (lambda.row(2) - oneMinusExpMinusLambda.row(2))/lambda3.row(2) + 4*(omega.row(2)/lambda2.row(2))%(oneMinusExpMinusLambda.row(2)));
  mat d_Omeg1_omega = zeros<mat>((p+q)*nSup, 2);
  ind = 2*nSup; // 2=q?
  d_Omeg1_omega.submat(ind,1,ind+nSup-1,1) = trans(4*(lambda.row(2) - oneMinusExpMinusLambda.row(2))/lambda2.row(2));


  const int nEq2 = nEq*nEq;
  mat K_dd = zeros<mat>(nEq2, nEq2);
  for (int i=0;i<nEq2;i++) { // OK if row dim = col dim
    for (int j=0;j<nEq2;j++) {
      int j1 = j/nEq;
      int j2 = j - j1*nEq;
      int I = j1 + j2*nEq;
      K_dd(i,j) = (I == i);
    }
  }
  if (debug) K_dd.print("K_dd=");

  const int nSupq = (nSup+1)*q;
  mat K_ = zeros<mat>(nSupq, nSupq);
  for (int i=0;i<nSupq;i++) {
    for (int j=0;j<nSupq;j++) {
      int j1 = j/(nSup+1);
      int j2 = j - j1*(nSup+1);
      int I = j1 + j2*q;
      K_(i,j) = (I == i);
    }
  }
  if (debug) K_.print("K_=");
  
  //d_gama2_Fii_=(eye(q).*.vec(Gi)')*(K__'.*.eye(2));
  //d_gama2_fi21=submat(d_gama2_Fii_,2,0);
  mat d_gama2_Fii_ = kronecker(eye<mat>(q,q), trans(vectorize(Gi))) * kronecker(trans(K_), eye<mat>(2,2));
  rowvec d_gama2_phi21 = d_gama2_Fii_.row(1);
  if (debug) d_gama2_phi21.print("d_gama2_phi21=");

  //d_Sigma_F1= (Omeg1*F1'.*.eye(d))*(eye(d^2)+K__dd);
  mat d_Sigma_F1 = kronecker(Omeg1*trans(F1), eye<mat>(nEq,nEq)) * (eye<mat>(nEq*nEq, nEq*nEq) + K_dd);
  if (debug) d_Sigma_F1.print("d_Sigma_F1=");
  mat d_Sigma_Omeg1_tmp = kronecker(trans(F1), trans(F1));
  mat d_Sigma_Omeg1(2, d_Sigma_Omeg1_tmp.n_cols);
  d_Sigma_Omeg1.row(0) = d_Sigma_Omeg1_tmp.row(0); //submat(d_Sigma_Omeg1,1|4,0);
  d_Sigma_Omeg1.row(1) = d_Sigma_Omeg1_tmp.row(3); //submat(d_Sigma_Omeg1,1|4,0);


  vec gr = d_tau * dertau;
  //  d_Sig_psi.print("d_Sig_psi=");
  //  gr.print("gr=");
  //grmy=d_tau_my*dertau+d_Sig1_my*d_Sig1+d_Sig2_my*d_Sig2+d_F1_my*d_Sigma_F1*derSigma;
  vec gr_tmp = d_Sig1_mu*d_Sig1 + d_Sig2_mu*d_Sig2 + d_F1_mu*d_Sigma_F1*derSigma;
  gr.rows(0,q-1) = gr.rows(0,q-1) + gr_tmp;
  //  gr(0) = gr(0) + (d_Sig_mu * derSigma);   // mhp. mu

  ind = q + (p+q)*nSup;  // First psi index
  gr_tmp = d_Sig1_psi*d_Sig1 + d_Sig2_psi*d_Sig2 + d_Omeg1_psi*d_Sigma_Omeg1*derSigma; //+ d_Sig_psi * derSigma; // mhp. psi
  gr.rows(ind, ind+p+q-1) = gr.rows(ind, ind+p+q-1) + gr_tmp;

  //  gr.print("gr=");
  gr.rows(ind, ind+p+q-1) = psi % gr.rows(ind, ind+p+q-1);

  const vec sigma_prod_lambda = d_Sig1_lam*d_Sig1+d_Sig2_lam*d_Sig2 + d_Omeg1_lam*d_Sigma_Omeg1*derSigma; //d_Sig_lam * derSigma;
  const vec sigma_prod_omega = d_Sig1_omega*d_Sig1+d_Sig2_omega*d_Sig2 + d_Omeg1_omega*d_Sigma_Omeg1*derSigma; //d_Sig_omega * derSigma;
  ind = 0;
  for (int k=0;k<p+q;k++) {
    int I0 = q + k*nSup; // first lambda index
    int J0 = q + (p+q)*nSup + (p+q) + k*nSup; // first omega index
    int dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i==nSup_fullmodel)
	dim_sub = 1;
      const vec derfii_sub = derfii[ind].col(dim_sub-1);
      double tmp1 = accu(d_fii_lam[ind] % derfii_sub);
      const double tmp2 = accu(d_Q_lam[ind] % derQu[ind]);
      //      const vec tmp3 = d_Sig_lam.row(i)*derSigma;
      gr(I0+i) = gr(I0+i) + tmp1 + tmp2 + sigma_prod_lambda(ind); // wrt lambda
      tmp1 = accu(d_Q_omega[ind] % derQu[ind]);
      gr(J0+i) = gr(J0+i) + tmp1 + sigma_prod_omega(ind);  // wrt omega
      ind++;
    }
  }

  gr.rows(q, q + (p+q)*nSup-1) = cA_ * gr.rows(q, q + (p+q)*nSup-1);  // deriverte mhp lambda1, lambda2

  ind = q + (q+p)*nSup + p+q;
  gr.rows(ind,ind+(p+q)*nSup-1) = vectorize(trans(omega)) % gr.rows(ind,ind+(p+q)*nSup-1);

  //grfi21=d_tau_fi21*dertau+d_gama2_fi21*vec(derGama2)+d_F1_fi21*d_Sigma_F1*derSigma+d_Sig2_fi21*d_Sig2;
  mat derGama2 = derG.cols(nGamma-nSup-1, nGamma-1); // [.,p___-p__:p___];
  if (debug) derGama2.print("derGama2=");
  if (debug) gr.print_trans("gr (before phi21 assignment)=");
  if (0) {
    Rprintf("1) d_tau_fi21*dertau= %8.6f\n", gr(npar-1));
    double tmp =  as_scalar(d_gama2_phi21*vectorize(derGama2));
    Rprintf("2)d_gama2_phi21*vectorize(derGama2) =  %8.6f\n", tmp);
    tmp = as_scalar(d_F1_phi21*d_Sigma_F1*derSigma);
    Rprintf("3) d_F1_phi21*d_Sigma_F1*derSigma=%8.6f\n", tmp);
    tmp = as_scalar(d_Sig2_phi21*d_Sig2);
    Rprintf("4) d_Sig2_phi21*d_Sig2= %8.6f\n", tmp);
  }
  gr(npar-1) = gr(npar-1) + as_scalar(d_gama2_phi21*vectorize(derGama2) + d_F1_phi21*d_Sigma_F1*derSigma + d_Sig2_phi21*d_Sig2);


  vec df = -1 * gr;
  //  df.print("df=");
  
  //  Rprintf("Quit QL::gradient\n");

  if (addPenalty) {
    //    if (nSup > 1) {
    ind = q; // first lambda index
    for (int k=0;k<p+q;k++) {
      for (int i=0;i<nSup;i++) {
	double pen = 300*(MAX2(par(ind)-penaltyMax)-MAX2(penaltyMin-par(ind)));
	if (pen != 0) {
	  //	Rprintf("pen1 %6.4f pen2 %6.4f\n", pen1, pen2);
	  //	par.print_trans("par=");
	  //	df.print_trans("df=");
	  df(ind) += pen;
	  ind++;
	  //	df.print_trans("df=");
	}
      }
    }
  }

  if (0) {
    Rprintf("Quit QL::gradient_multivariat\n");
    df.print_trans("df=");
    exit(-1);
  }

  delete [] derQu;
  delete [] derfii;
  delete [] d_fii_lam;
  delete [] d_Q_lam;
  delete [] d_Q_omega;

  return df;
}


vec QL::gradient_multivariat_individual(const vec & par,
					const mat & b, const mat & W, const mat & BBt,
					const mat & Qu, const mat & sigma, const mat & fii,
					const mat & Gi, const mat & F1,
					const vec & tau, const mat & gama,
					mat & gr) {
  const int debug=0;
  if (debug) Rprintf("Enter QL::gradient_multivariat\n");

  ParametersMulti par_extract(par, transf);
  const vec mu = par_extract.mu;
  const mat lambda = par_extract.lambda;
  const vec psi = par_extract.psi;
  const mat omega = par_extract.omega;
  const double phi21 = par_extract.phi(2,1);

  const int npar = par.n_elem;
  const int p = ParametersMulti::p;
  const int q = ParametersMulti::q;
  

  const int nSup = par_extract.lambda.n_cols;
  const int ap = Z0.n_cols;
  const int nEq = Z0.n_rows;
  const int nGamma = gama.n_cols;
  int nSup_fullmodel; // number of terms with exact state space model representation
  if (nSup == 1)
    nSup_fullmodel = nSup;
  else
    nSup_fullmodel = nSup - 1;
  const int nSup_approx = nSup - nSup_fullmodel;
  const int nLatent = 2*nSup_fullmodel + nSup_approx;

  gr.set_size(npar, ap);

  const mat expMinusLambda = exp(-lambda);
  const mat expMinusTwoLambda = exp(-2.0*lambda);
  const mat oneMinusExpMinusLambda = 1 - expMinusLambda;
  const mat oneMinusExpMinusTwoLambda = 1 - expMinusTwoLambda;

  const mat lambda2 = lambda % lambda;
  const mat lambda3 = lambda2 % lambda;

  mat GG_ = gama;

  //diagrv(eye(2),psi[3]|(sumr(4*(omega[3,.]./lamda[3,.].^2).*(exp(-lamda[3,.])-1+lamda[3,.]))+2*psi[3]^2));
  mat Omeg1 = eye<mat>(2,2);
  Omeg1(0,0) = psi(2);
  const rowvec Omeg1_tmp1 = 4*(omega.row(2)/lambda2.row(2));
  const rowvec Omeg1_tmp2 =  lambda.row(2) -oneMinusExpMinusLambda.row(2);
  const rowvec Omeg1_tmp = Omeg1_tmp1 %  Omeg1_tmp2;
  //  double tmp = accu(4*(omega.row(2)/lambda2.row(2)) % (-oneMinusExpMinusLambda.row(2) + lambda.row(2)));
  Omeg1(1,1) = sum(Omeg1_tmp) + 2*psi(2)*psi(2);

  //  Qu.print("Qu=");
  
  mat iQu = zeros<mat>(nLatent*(q+p), nLatent*(q+p));
  for (int k=0;k<q+p;k++) {
     int I0 = k*nLatent;
     for (int i=0;i<nSup_fullmodel;i++) {
       int i0 = I0+2*i;
       int i1 = I0+2*i+1;
       mat Qu_sub = Qu.submat(i0,i0,i1,i1);
       //    Qu_sub.print("Qu_sub=");
       iQu.submat(i0,i0,i1,i1) = inv(Qu_sub);
     }
     for (int i=nSup_fullmodel;i<nSup;i++) {
       int i0 = I0+i+nSup_fullmodel;
       iQu(i0,i0) = 1/Qu(i0,i0);
     }
  }

  mat isigma = inv(sigma);
  
  if (debug) iQu.print("iQu=");

  //  isigma.print("isigma=");
  //  iQu.print("iQu=");
  //  tau.print("tau=");
  //  gama.print("gama=");
  //  ##derG = 0;
  
  vec dertau = zeros<vec>(nEq);
  vec derSigma = zeros<vec>(nEq*nEq);
  mat derG = zeros<mat>(2*q,nGamma);

  vec * derQu = new vec[nSup*(p+q)];
  mat * derfii = new mat[nSup*(p+q)];
  int ind=0;
  for (int k=0;k<q+p;k++) {
    int dim_sub = 2;
    for (int i=0;i<nSup;i++) {
      if (i == nSup_fullmodel)
	dim_sub = 1;
      derQu[ind] = zeros<vec>(dim_sub*dim_sub);
      derfii[ind++] = zeros<mat>(dim_sub,dim_sub);
    }
  }

  if (debug) Rprintf("Start loop\n");
  if (debug) Rprintf("ap=%d\n", ap);

  for (int t=0;t<ap;t++) {//do until t>ap;
    if (t==3 && debug)
      exit(-1);

    if (debug==2) Rprintf("t=%d\n", t);
    int t2 = nGamma*(t-1) - 1;
    int t1 = nGamma*t - 1;
    int t0 = nGamma*(t+1) - 1;
    
    
    vec Rt = Z0.col(t) - tau - GG_ * b.col(t);

    mat WW = GG_ * W.cols(t1+1,t0) * trans(GG_);

    mat Ht =  Rt * trans(Rt) + WW;

    dertau = isigma * Rt; // check the below line!
    derSigma = 0.5*(-vectorize(isigma) + kronecker(isigma, isigma) * vectorize(Ht));

    derG = isigma*(Rt*trans(b.col(t)) - GG_*W.cols(t1+1, t0));

    if (debug==2) dertau.print("dertau=");
    if (debug==2) derSigma.print("derSigma=");
    if (debug==2) derG.print("derG=");

    if (t>0) {
      mat VBt = W.cols((t1+1),t0) * trans(BBt.cols(t1+1,t0));
      mat Wt_ = W.cols(t2+1,t1);
      mat Wt = W.cols(t1+1,t0);

      Ht =  (b.col(t) - fii * b.col(t-1))  *  trans(b.col(t) - fii * b.col(t-1)) +
	Wt - fii * trans(VBt) - VBt * trans(fii) + fii * Wt_ * trans(fii);

      ind = 0;
      for (int k=0;k<q+p;k++) {
	int inc = 2; // dimension of sub model
	int I0 = k*nLatent;
	int i0 = I0;
	for (int i=0;i<nSup;i++) { // loop over full model + approx model super pos terms
	  if (i==nSup_fullmodel) // turn to approx model
	    inc = 1;
	  int i1 = i0+inc-1;

	  //	mat b_out01 = outer(b.col(t), b.col(t-1));
	  //	mat b_out11 = outer(b.col(t-1), b.col(t-1));
	  mat b_out01 = outer(b.submat(i0,t,i1, t), b.submat(i0,t-1,i1,t-1));
	  mat b_out11 = outer(b.submat(i0,t-1,i1, t-1), b.submat(i0,t-1,i1,t-1));

	  mat iQu_sub = iQu.submat(i0,i0, i1, i1);
	  mat VBt_sub = VBt.submat(i0,i0, i1, i1);
	  mat fii_sub = fii.submat(i0,i0, i1, i1);
	  mat Wt_sub = Wt_.submat(i0,i0, i1, i1);
	  //	fii_sub.print("fii_sub=");
	  derfii[ind] =  iQu_sub * (b_out01 + VBt_sub - fii_sub*(b_out11 + Wt_sub));
	  if (debug==2) Rprintf("(k,i)=(%d,%d)\n", k,i);
	  if (debug==2) derfii[ind].print("derfii[ind]=");
	  mat Ht_sub = Ht.submat(i0,i0,i1,i1);
	  derQu[ind] = 0.5*(-vectorize(iQu_sub) + kronecker(iQu_sub,iQu_sub)* vectorize(Ht_sub));
	  if (debug==2) derQu[ind].print("derQu[ind]=");
	  ind++;

	  i0 += inc;
	}
      }
    }
  
    if (debug) Rprintf("Loop finished\n");
    if (debug) dertau.print("dertau=");
    if (debug) derSigma.print("derSigma=");
    if (debug) derG.print("derG=");
    for (ind=0;ind<(p+q)*nSup;ind++) {
      if (debug) derfii[ind].print("derfii[ind]=");
      if (debug) derQu[ind].print("derQu[ind]=");
    }

    // diagonalelementene i deriverte mhp. Qu
    //  vec derfii_diag = derfii.diag();     // diagonalelementene i deriverte mhp. Fii 

    //  for (int i=0;i<nSup;i++) { // loop over full model + approx model super po
    //    derfii[i].print("derfii=");
    //    derQu[i].print("derQu=");
    //  }

    // d_tau_my=(1~2*my[1]~zeros(1,2))| (zeros(1,2)~1~2*my[2]);
    mat d_tau = zeros<mat>(npar, nEq);
    // wrt mu_1 and mu_2
    d_tau(0,0) = 1;
    d_tau(0,1) = 2*mu(0);
    d_tau(1,2) = 1;
    d_tau(1,3) = 2*mu(1);
    // wrt psi_1, psi_2, psi_3
    const int ind_psi = q + (p+q)*nSup;
    d_tau(ind_psi,1) = 1;
    d_tau(ind_psi+1,3) = 1;
    d_tau(ind_psi+2,1) = 1;
    d_tau(ind_psi+2,3) = phi21;
    // wrt phi21
    d_tau(npar-1, 3) = psi(2);

    if (debug) d_tau.print("d_tau=");

    vec twoSumPsi = 2*psi;

    mat d_Sig1_mu = zeros<mat>(2,nEq);
    mat d_Sig2_mu = zeros<mat>(2,nEq);
    d_Sig1_mu(0,1) = twoSumPsi(0);
    d_Sig1_mu(0,2) = twoSumPsi(0);
    d_Sig1_mu(0,3) = 4*mu(0)*twoSumPsi(0);

    d_Sig2_mu(1,1) = twoSumPsi(1);
    d_Sig2_mu(1,2) = twoSumPsi(1);
    d_Sig2_mu(1,3) = 4*mu(1)*twoSumPsi(1);

    rowvec d_Sig2_phi21 = zeros<rowvec>(nEq);
    d_Sig2_phi21(3) = 8*psi(1)*psi(2)*phi21;

    if (debug) d_Sig1_mu.print("d_Sig1_mu=");
    if (debug) d_Sig2_mu.print("d_Sig2_mu=");
    if (debug) d_Sig2_phi21.print("d_Sig2_phi21=");

    vec * d_fii_lam = new vec[(p+q)*nSup];
    ind = 0;
    for (int k=0;k<p+q;k++) {
      int dim_sub = 2;
      for (int i=0;i<nSup;i++) {
	if (i == nSup_fullmodel)
	  dim_sub = 1;
	d_fii_lam[ind] = zeros<vec>(dim_sub);

	if (dim_sub == 1) {
	  d_fii_lam[ind](0) = -expMinusLambda(k,i);
	}
	else {
	  d_fii_lam[ind](0) = (1/lambda(k,i))*expMinusLambda(k,i) -
	    (1/lambda2(k,i))*oneMinusExpMinusLambda(k,i);
	  d_fii_lam[ind](1) = -expMinusLambda(k,i);
	}
	if (debug) d_fii_lam[ind].print("d_fii_lam[ind]=");
	ind++;
      }
    }
  
    vec * d_Q_lam = new vec[(p+q)*nSup];
    ind = 0;
    for (int k=0;k<p+q;k++) {
      int dim_sub = 2;
      for (int i=0;i<nSup;i++) {
	if (i == nSup_fullmodel)
	  dim_sub = 1;

	d_Q_lam[ind] = zeros<vec>(dim_sub*dim_sub);
	if (dim_sub == 1) {
	  d_Q_lam[ind](0) = 2*omega(k,i);
	}
	else {
	  d_Q_lam[ind](0) = -(2/lambda3(k,i))*(-1.5 - 0.5*expMinusTwoLambda(k,i) + 2*expMinusLambda(k,i) + lambda(k,i)) + (1/lambda2(k,i))*(expMinusTwoLambda(k,i) - 2*expMinusLambda(k,i) + 1);
	  d_Q_lam[ind](1) = -(1/lambda2(k,i))*(oneMinusExpMinusLambda(k,i)-0.5*oneMinusExpMinusTwoLambda(k,i)) + (1/lambda(k,i))*(expMinusLambda(k,i) - expMinusTwoLambda(k,i));
	  d_Q_lam[ind](2) = d_Q_lam[ind](1);
	  d_Q_lam[ind](3) = expMinusTwoLambda(k,i);

	  d_Q_lam[ind] = 2*omega(k,i) * d_Q_lam[ind];
	}
	if (debug) d_Q_lam[ind].print("d_Q_lam[ind]=");
	ind++;
      }
    }

    mat d_Sig1_lam = zeros<mat>(3*nSup,4); // check what is 4?
    mat d_Sig2_lam = zeros<mat>(3*nSup,4);

    //d_Sig1_lam=zeros(3*p__,1)~zeros(3*p__,1)~zeros(3*p__,1)~vec((-8*omega[1,.].*(lamda[1,.].^(-3)).*(exp(-lamda[1,.])-1+lamda[1,.])+4*omega[1,.].*(lamda[1,.].^(-2)).*(-exp(-lamda[1,.])+1) )|zeros(2,p__));  

    d_Sig1_lam.submat(0,3,nSup-1,3) = trans(-8*(omega.row(0)/lambda3.row(0)) % (lambda.row(0) - oneMinusExpMinusLambda.row(0))
					    + 4*(omega.row(0)/lambda2.row(0)) % oneMinusExpMinusLambda.row(0));
    //  d_Sig_lam.print("d_Sig_lam=");

    d_Sig2_lam.submat(nSup,3,2*nSup-1,3) = trans(-8*(omega.row(1)/lambda3.row(1)) % (lambda.row(1) - oneMinusExpMinusLambda.row(1))
						 + 4*(omega.row(1)/lambda2.row(1)) % oneMinusExpMinusLambda.row(1));




    mat d_Sig1_psi = zeros<mat>(p+q, nEq);
    mat d_Sig2_psi = zeros<mat>(p+q, nEq);
    d_Sig1_psi(0,0) = 1;
    d_Sig1_psi(0,1) = 2*mu(0);
    d_Sig1_psi(0,2) = 2*mu(0);
    d_Sig1_psi(0,3) = 4*mu(0)*mu(0) + 4*psi(0) + 4*psi(2);
    d_Sig1_psi(2,3) = 4*psi(0);

    d_Sig2_psi(1,0) = 1;
    d_Sig2_psi(1,1) = 2*mu(1);
    d_Sig2_psi(1,2) = 2*mu(1);
    d_Sig2_psi(1,3) = 4*mu(1)*mu(1) + 4*psi(1) + 4*psi(2)*phi21*phi21;
    d_Sig2_psi(2,3) = 4*psi(1)*phi21*phi21;

    //  d_Sig_psi.print("d_Sig_psi=");

    //  d_Q_omega = diag(2*lambda, nrow=nSup, ncol=nSup);
    vec * d_Q_omega = new vec[(p+q)*nSup];
    ind =0;
    for (int k=0;k<p+q;k++) {
      int I0 = k*nLatent;
      int i0 = I0;
      int dim_sub = 2;
      for (int i=0;i<nSup;i++) {
	if (i == nSup_fullmodel)
	  dim_sub = 1;
	int i1 = i0+dim_sub-1;

	mat Qu_sub = Qu.submat(i0, i0, i1, i1);
	d_Q_omega[ind] = (1/omega(k,i))*vectorize(Qu_sub);
	i0 += dim_sub;
	if (debug) d_Q_omega[ind].print("d_Q_omega[ind]=");
	ind++;
      }
    }

    mat d_Sig1_omega = zeros<mat>(nSup*(p+q), 4);
    mat d_Sig2_omega = zeros<mat>(nSup*(p+q), 4);
    d_Sig1_omega.submat(0,3,nSup-1,3) = trans(4*(lambda.row(0) - oneMinusExpMinusLambda.row(0))/lambda2.row(0));
    d_Sig2_omega.submat(nSup,3,2*nSup-1,3) = trans(4*(lambda.row(1) - oneMinusExpMinusLambda.row(1))/lambda2.row(1));
    //  d_Sig_omega.print("d_Sig_omega=");

  
    mat cA_ = zeros<mat>((p+q)*nSup, (p+q)*nSup);
    for (int k=0;k<p+q;k++) {
      int I0 = k*nSup;
      if (transf == 1 && nSup == 2) {
	cA_(I0,I0) = (lambda(k,0)-ParametersMulti::minlambda(k))*(1-(lambda(k,0)-ParametersMulti::minlambda(k))/(ParametersMulti::maxlambda(k)-ParametersMulti::minlambda(k)));
	cA_(I0+1,I0+1) = lambda(k,1)*(1-lambda(k,1)/ParametersMulti::minlambda(k));
      }
      else {
	vec lambdaStar(nSup);
	for (int i=0;i<nSup;i++) {
	  lambdaStar(i) = (lambda(k, i)-ParametersMulti::minlambda(k))/(ParametersMulti::maxlambda(k)-ParametersMulti::minlambda(k));
	}
	for (int i=0;i<nSup;i++) {
	  for (int j=i;j<nSup;j++) {
	    cA_(I0+i,I0+j) = (ParametersMulti::maxlambda(k)-ParametersMulti::minlambda(k))* lambdaStar(j) / (1 + exp(par(I0+i+q)));
	  }
	}
      }
    }

    if (debug) cA_.print("cA_=");
  
    vec d_Sig1(2*q);
    vec d_Sig2(2*q);
    d_Sig1(0) = derSigma(0);
    d_Sig1(1) = derSigma(1);
    d_Sig1(2) = derSigma(2*q);
    d_Sig1(3) = derSigma(2*q+1);
    d_Sig2(0) = derSigma(5*q);
    d_Sig2(1) = derSigma(5*q+1);
    d_Sig2(2) = derSigma(7*q);
    d_Sig2(3) = derSigma(7*q+1);

    if (debug) d_Sig2.print("d_Sig2=");
  
    //d_F1_my=(0~2~zeros(1,6))|(zeros(1,3)~2*fi21~zeros(1,4));
    //d_F1_fi21=(0~0~1~2*my[2]~zeros(1,3)~2*fi21);
    mat d_F1_mu = zeros<mat>(q,8); // 2=q? 8?
    rowvec d_F1_phi21 = zeros<rowvec>(8); //6=?
  
    d_F1_mu(0,1) = 2;
    d_F1_mu(1,3) = 2*phi21;
    d_F1_phi21(2) = 1;
    d_F1_phi21(3) = 2*mu(1);
    d_F1_phi21(7) = 2*phi21;

    if (debug) d_F1_phi21.print("d_F1_phi21=");

    mat d_Omeg1_psi = zeros<mat>(p+q,2); //3=p+q? //zeros(2,2)|(1~4*psi[3]);
    d_Omeg1_psi(2,0) = 1;
    d_Omeg1_psi(2,1) = 4*psi(2);
    //d_Omeg1_lam=zeros(3*p__,1)~vec(zeros(2,p__)| (-8*omega[3,.].*(lamda[3,.].^(-3)).*(exp(-lamda[3,.])-1+lamda[3,.])+4*omega[3,.].*(lamda[3,.].^(-2)).*(-exp(-lamda[3,.])+1) ));  

    //d_Omeg1_omega=zeros(3*p__,1)~vec(zeros(2,p__) | (4*(lamda[3,.].^(-2)).*(exp(-lamda[3,.])-1+lamda[3,.])) );

    mat d_Omeg1_lam = zeros<mat>((p+q)*nSup, 2);
    ind = 2*nSup; // 2=q?
    d_Omeg1_lam.submat(ind,1,ind+nSup-1,1) = trans(-8*omega.row(2) % (lambda.row(2) - oneMinusExpMinusLambda.row(2))/lambda3.row(2) + 4*(omega.row(2)/lambda2.row(2))%(oneMinusExpMinusLambda.row(2)));
    mat d_Omeg1_omega = zeros<mat>((p+q)*nSup, 2);
    ind = 2*nSup; // 2=q?
    d_Omeg1_omega.submat(ind,1,ind+nSup-1,1) = trans(4*(lambda.row(2) - oneMinusExpMinusLambda.row(2))/lambda2.row(2));


    const int nEq2 = nEq*nEq;
    mat K_dd = zeros<mat>(nEq2, nEq2);
    for (int i=0;i<nEq2;i++) { // OK if row dim = col dim
      for (int j=0;j<nEq2;j++) {
	int j1 = j/nEq;
	int j2 = j - j1*nEq;
	int I = j1 + j2*nEq;
	K_dd(i,j) = (I == i);
      }
    }
    if (debug) K_dd.print("K_dd=");

    const int nSupq = (nSup+1)*q;
    mat K_ = zeros<mat>(nSupq, nSupq);
    for (int i=0;i<nSupq;i++) {
      for (int j=0;j<nSupq;j++) {
	int j1 = j/(nSup+1);
	int j2 = j - j1*(nSup+1);
	int I = j1 + j2*q;
	K_(i,j) = (I == i);
      }
    }
    if (debug) K_.print("K_=");
  
    //d_gama2_Fii_=(eye(q).*.vec(Gi)')*(K__'.*.eye(2));
    //d_gama2_fi21=submat(d_gama2_Fii_,2,0);
    mat d_gama2_Fii_ = kronecker(eye<mat>(q,q), trans(vectorize(Gi))) * kronecker(trans(K_), eye<mat>(2,2));
    rowvec d_gama2_phi21 = d_gama2_Fii_.row(1);
    if (debug) d_gama2_phi21.print("d_gama2_phi21=");

    //d_Sigma_F1= (Omeg1*F1'.*.eye(d))*(eye(d^2)+K__dd);
    mat d_Sigma_F1 = kronecker(Omeg1*trans(F1), eye<mat>(nEq,nEq)) * (eye<mat>(nEq*nEq, nEq*nEq) + K_dd);
    if (debug) d_Sigma_F1.print("d_Sigma_F1=");
    mat d_Sigma_Omeg1_tmp = kronecker(trans(F1), trans(F1));
    mat d_Sigma_Omeg1(2, d_Sigma_Omeg1_tmp.n_cols);
    d_Sigma_Omeg1.row(0) = d_Sigma_Omeg1_tmp.row(0); //submat(d_Sigma_Omeg1,1|4,0);
    d_Sigma_Omeg1.row(1) = d_Sigma_Omeg1_tmp.row(3); //submat(d_Sigma_Omeg1,1|4,0);


    gr.col(t) = d_tau * dertau;
    //  d_Sig_psi.print("d_Sig_psi=");
    //  gr.print("gr=");
    //grmy=d_tau_my*dertau+d_Sig1_my*d_Sig1+d_Sig2_my*d_Sig2+d_F1_my*d_Sigma_F1*derSigma;
    vec gr_tmp = d_Sig1_mu*d_Sig1 + d_Sig2_mu*d_Sig2 + d_F1_mu*d_Sigma_F1*derSigma;
    gr.submat(0,t,q-1,t) = gr.submat(0,t,q-1,t) + gr_tmp;
    //  gr(0) = gr(0) + (d_Sig_mu * derSigma);   // mhp. mu

    ind = q + (p+q)*nSup;  // First psi index
    gr_tmp = d_Sig1_psi*d_Sig1 + d_Sig2_psi*d_Sig2 + d_Omeg1_psi*d_Sigma_Omeg1*derSigma; //+ d_Sig_psi * derSigma; // mhp. psi
    gr.submat(ind, t,ind+p+q-1,t) = gr.submat(ind,t, ind+p+q-1,t) + gr_tmp;

    //  gr.print("gr=");
    gr.submat(ind,t, ind+p+q-1,t) = psi % gr.submat(ind,t, ind+p+q-1,t);

    const vec sigma_prod_lambda = d_Sig1_lam*d_Sig1+d_Sig2_lam*d_Sig2 + d_Omeg1_lam*d_Sigma_Omeg1*derSigma; //d_Sig_lam * derSigma;
    const vec sigma_prod_omega = d_Sig1_omega*d_Sig1+d_Sig2_omega*d_Sig2 + d_Omeg1_omega*d_Sigma_Omeg1*derSigma; //d_Sig_omega * derSigma;
    ind = 0;
    for (int k=0;k<p+q;k++) {
      int I0 = q + k*nSup; // first lambda index
      int J0 = q + (p+q)*nSup + (p+q) + k*nSup; // first omega index
      int dim_sub = 2;
      for (int i=0;i<nSup;i++) {
	if (i==nSup_fullmodel)
	  dim_sub = 1;
	const vec derfii_sub = derfii[ind].col(dim_sub-1);
	double tmp1 = accu(d_fii_lam[ind] % derfii_sub);
	const double tmp2 = accu(d_Q_lam[ind] % derQu[ind]);
	//      const vec tmp3 = d_Sig_lam.row(i)*derSigma;
	gr(I0+i,t) = gr(I0+i,t) + tmp1 + tmp2 + sigma_prod_lambda(ind); // wrt lambda
	tmp1 = accu(d_Q_omega[ind] % derQu[ind]);
	gr(J0+i,t) = gr(J0+i,t) + tmp1 + sigma_prod_omega(ind);  // wrt omega
	ind++;
      }
    }

    gr.submat(q,t, q + (p+q)*nSup-1,t) = cA_ * gr.submat(q,t, q + (p+q)*nSup-1,t);  // deriverte mhp lambda1, lambda2

    ind = q + (q+p)*nSup + p+q;
    gr.submat(ind,t,ind+(p+q)*nSup-1,t) = vectorize(trans(omega)) % gr.submat(ind,t,ind+(p+q)*nSup-1,t);

    //grfi21=d_tau_fi21*dertau+d_gama2_fi21*vec(derGama2)+d_F1_fi21*d_Sigma_F1*derSigma+d_Sig2_fi21*d_Sig2;
    mat derGama2 = derG.cols(nGamma-nSup-1, nGamma-1); // [.,p___-p__:p___];
    if (debug) derGama2.print("derGama2=");
    if (debug) gr.print_trans("gr (before phi21 assignment)=");
    if (0) {
      Rprintf("1) d_tau_fi21*dertau= %8.6f\n", gr(npar-1));
      double tmp =  as_scalar(d_gama2_phi21*vectorize(derGama2));
      Rprintf("2)d_gama2_phi21*vectorize(derGama2) =  %8.6f\n", tmp);
      tmp = as_scalar(d_F1_phi21*d_Sigma_F1*derSigma);
      Rprintf("3) d_F1_phi21*d_Sigma_F1*derSigma=%8.6f\n", tmp);
      tmp = as_scalar(d_Sig2_phi21*d_Sig2);
      Rprintf("4) d_Sig2_phi21*d_Sig2= %8.6f\n", tmp);
    }
    gr(npar-1,t) = gr(npar-1,t) + as_scalar(d_gama2_phi21*vectorize(derGama2) + d_F1_phi21*d_Sigma_F1*derSigma + d_Sig2_phi21*d_Sig2);

    if (addPenalty) {
      //      if (nSup > 1) {
      ind = q; // first lambda index
      for (int k=0;k<p+q;k++) {
	for (int i=0;i<nSup;i++) {
	  double pen = 300*(MAX2(par(ind)-penaltyMax)-MAX2(penaltyMin-par(ind)));
	  if (pen != 0) {
	    //	Rprintf("pen1 %6.4f pen2 %6.4f\n", pen1, pen2);
	    //	par.print_trans("par=");
	    //	df.print_trans("df=");
	    gr(ind,t) -= pen/ap;
	    ind++;
	    //	df.print_trans("df=");
	  }
	}
      }
    }
  }

  vec df = zeros<vec>(par.n_elem); // = -1 * gr;
  for (int t=0;t<ap;t++) {//do until t>ap;
    df = df - gr.col(t);
  }
  if (debug) {
    df.print_trans("df (should be zero)=");

    Rprintf("Quit QL::gradient_multivariat_individual\n");
  }
  return df;
}


mat FF(int t, mat fii) {
  return fii;
}



// QL:filter
void QL::filter(const vec & par, const mat & Z0, mat & A, mat & A_,
		mat & V, mat & V_, mat & Qu, mat & sigma, mat & fii, double & f) {
  //Rprintf("Enter QL::filter\n");
  Parameters par_extract(par, transf);

  const double mu = par_extract.mu;
  const vec lambda = par_extract.lambda;
  const double psi = par_extract.psi;
  const vec omega = par_extract.omega;
  
  const int nSup = (par.n_elem - 2)/2;
  const int ap = Z0.n_cols;
  const int nEq = Z0.n_rows;
  int nSup_fullmodel; // number of terms with exact state space model representation
  if (nSup == 1)
    nSup_fullmodel = nSup;
  else
    nSup_fullmodel = nSup - 1;
  const int nSup_approx = nSup - nSup_fullmodel;
  const int nLatent = 2*nSup_fullmodel + nSup_approx;
  const vec expMinusLambda = exp(-lambda);
  const vec expMinusTwoLambda = exp(-2.0*lambda);
  const vec oneMinusExpMinusLambda = 1 - expMinusLambda;
  const vec oneMinusExpMinusTwoLambda = 1 - expMinusTwoLambda;
  
  const vec lambda2 = lambda % lambda;


  //lambda.print("lambda=");
  //omega.print("omega=");
  //psi.print("psi=");
  //oneMinusExpMinusTwoLambda.print("oneMinusExpMinusTwoLambda=");

  vec tau(nEq); //= "mu accu(psi)";
  tau(0)= mu;
  tau(1) = psi + mu*mu;

  //tau.print("tau=");

  mat gama = zeros<mat>(2,nLatent); // nEq x nLatent
  for (int i=0;i<nSup_fullmodel;i++) {
    gama(1,2*i) = 1;
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    gama(1,i+nSup_fullmodel) = 1;
  }
  //gama.print("gama=");

  Qu = zeros<mat>(nLatent, nLatent);
  for (int i=0;i<nSup_fullmodel;i++) {
    Qu(2*i,2*i) = 2*(omega(i)/lambda2(i))*(-1.5-0.5*expMinusTwoLambda(i) + 2*expMinusLambda(i) + lambda(i));
    Qu(2*i,2*i+1) = 2*(omega(i)/lambda(i))*(oneMinusExpMinusLambda(i) -0.5*oneMinusExpMinusTwoLambda(i));
    Qu(2*i+1,2*i) = Qu(2*i,2*i+1);
    Qu(2*i+1,2*i+1) = omega(i)*oneMinusExpMinusTwoLambda(i);
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    Qu(nSup_fullmodel+i, nSup_fullmodel+i) = 2*omega(i)*lambda(i);
  }

  //Qu.print("Qu=");

  const double sumPsi = psi;

  sigma = zeros<mat>(2,2); // nEq x nEq
  sigma(0,0) = sumPsi;
  sigma(0,1) = 2*mu*sumPsi;
  sigma(1,0) = sigma(0,1);
  //  sigma(1,1) = 2*accu(omega + psi % psi);
  sigma(1,1) = accu(4*(omega/lambda2) % (expMinusLambda-1+lambda)) + 2*psi*psi + 4*psi*mu*mu;

  //sigma.print("sigma=");

  fii = zeros<mat>(nLatent, nLatent);
  for (int i=0;i<nSup_fullmodel;i++) {
    fii(2*i, 2*i+1) = oneMinusExpMinusLambda(i)/lambda(i);
    fii(2*i+1, 2*i+1) = expMinusLambda(i);
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    fii(nSup_fullmodel+i, nSup_fullmodel+i) = expMinusLambda(i);
  }

  mat GG_ = gama;
  V_ = zeros<mat>(nLatent,nLatent*(ap+1));
  V = V_;

  A_ = zeros<mat>(nLatent,(ap+1));
  A = A_;

  
  V_.cols(1, nLatent) = 0*Qu; // sjekk

  //mat Z0sub = Z0.cols(0,8);
  //Z0sub.print("Z0=");
//  tau.print("tau=");
//  Qu.print("Qu=");
//  gama.print("gama=");
//  sigma.print("sigma=");
//  fii.print("fii=");
 

  f = 0;

  //  const int debug=1;
  for (int t=0;t<ap;t++) {

    //    if (t==4 && debug)
    //      exit(-1);
    
    int tm1 = nLatent*t - 1; // limits in V
    int t0 = nLatent*(t+1) - 1;
    int tp1 = nLatent*(t+2) - 1;
    
    vec Et = Z0.col(t) - tau - GG_ * A_.col(t);
    //    mat tmp = V_.cols((tm1+1),t0);
    //    mat tmp2 = GG_ * tmp * trans(GG_);
    //    mat Dt = tmp2 + sigma;
    mat Dt = GG_*V_.cols((tm1+1),t0)*trans(GG_) + sigma;
    mat iDt = inv(Dt);

    mat Kt = V_.cols((tm1+1),t0) * trans(GG_) * iDt;
    
    mat tmpmat =  Kt * Et;
    A.col(t) = A_.col(t) + tmpmat;
    A_.col(t+1) = FF(t+1,fii)*A.col(t);
    
    V.cols((tm1+1),t0) = V_.cols((tm1+1),t0) - Kt*GG_*V_.cols((tm1+1),t0);
    V_.cols((t0+1),tp1) = FF(t+1,fii)*V.cols((tm1+1),t0)*trans(FF(t+1,fii)) + Qu;


//    Et.print("Et=");
//    Dt.print("Dt=");
//    iDt.print("iDt=");
//    Kt.print("Kt=");

    //    vec Dt.eigenvalues = eigen(Dt).values;
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, Dt);
//    eigval.print("eigval=");
    //    if (any(is.na(eigval))) {
      //      print(Dt)
      //      print(Dt.eigenvalues)
      //      stop("NA eigenvalues")
    //      f = Inf;
    //      break;
    //    }
    if (min(eigval) < 0) {
      //      print(Dt)
      //      print(Dt.eigenvalues)
      //      stop("Negative eigenvalues")
      f = Inf;
      break;
    }
    vec val =  0.5*accu(log(eigval)) + 0.5*trans(Et)*iDt*Et;
    f = f + val(0);

  }
  //mat Asub = A.cols(0,4);
  //  Asub.print("A=");
  //  mat Asub_ = A_.cols(0,4);
  //  Asub_.print("A_=");
  //  mat Vsub = V.cols(0,4);
  //  Vsub.print("V=");
  //  mat Vsub_ = V_.cols(0,4);
  //  Vsub_.print("V_=");
  //  Rprintf("f=%8.4f\n", f);

  // Add penalty function
  if (addPenalty) {
    const int indLambda = 1;
    for (int i=0;i<nSup;i++) {
      //    if (nSup > 1) {
      //    f += 100*max(0, (par(1)-3))^3 + 100*max(0,-(3+par(1)))^3 + 100*max(0,(par(2)-3))^3 + 100*max(0,-(3+par(2)]))^3;   /* straffer c[2:3] utenfor [-3,+4]; */
      
      //      double pen = 100*(MAX3(par(1)-penaltyMax) + MAX3(penaltyMin - par(1)) + MAX3(par(2)-penaltyMax) + MAX3(penaltyMin-par(2)));
      double pen = 100*(MAX3(par(indLambda+i)-penaltyMax) + MAX3(penaltyMin - par(indLambda+i)));
      if (pen < 0.0) {
	double pen1 = 100*MAX3(par(1)-4);
	double pen2 = 100*MAX3(-(4+par(1)));
	double pen3 = 100*MAX3(par(2)-4);
	double pen4 = 100*MAX3(-(4+par(2)));
	Rprintf("Negative penalty! %6.4f  %6.4f  %6.4f  %6.4f\n", pen1, pen2, pen3, pen4);
	
//	par.print("par=");
      }
      f += pen;
    }
  }

  return;
}


void QL::filter_multivariat(const vec & par, const mat & Z0, mat & A, mat & A_,
			    mat & V, mat & V_, mat & Qu, mat & sigma, mat & fii,
			    vec & tau, mat & gama, mat & Gi, mat & F1,
			    double & f) {
  //  Rprintf("Enter QL::filter_multivariat\n");
  ParametersMulti par_extract(par, transf);

  const int debug=0;

  const vec mu = par_extract.mu;
  const mat lambda = par_extract.lambda;
  const vec psi = par_extract.psi;
  const mat omega = par_extract.omega;
  const double phi21 = par_extract.phi(2,1);

  const int p = ParametersMulti::p;
  const int q = ParametersMulti::q;
  
  const int nSup = par_extract.lambda.n_cols;
  const int ap = Z0.n_cols;
  const int nEq = Z0.n_rows;
  int nSup_fullmodel; // number of terms with exact state space model representation
  if (nSup == 1)
    nSup_fullmodel = nSup;
  else
    nSup_fullmodel = nSup - 1;
  const int nSup_approx = nSup - nSup_fullmodel;
  const int nLatent = 2*nSup_fullmodel + nSup_approx;
  const mat expMinusLambda = exp(-lambda);
  const mat expMinusTwoLambda = exp(-2.0*lambda);
  const mat oneMinusExpMinusLambda = 1 - expMinusLambda;
  const mat oneMinusExpMinusTwoLambda = 1 - expMinusTwoLambda;
  
  const mat lambda2 = lambda % lambda;


  mat Fii_(q,p); // samme som fii paa side 16 i paperet
  Fii_(0,0) = 1;
  Fii_(1,0) = phi21;
  int ncols_F1;
  if (p==1)
    ncols_F1 = 2*p;
  else
    ncols_F1 = 2*p + p*(p-1)/2;
  if (debug) Fii_.print("Fii_=");

  F1 = zeros<mat>(2*q,ncols_F1); // = Phi_tilde side 16
  F1(0,0) = 1;
  F1(1,0) = 2*mu(0);
  F1(1,1) = 1;
  F1(2,0) = phi21;
  F1(3,0) = 2*mu(1)*phi21;
  F1(3,1) = phi21*phi21;

 if (debug)  F1.print("F1=");

  Gi = zeros<mat>(2,nLatent); // nEq x nLatent
  for (int i=0;i<nSup_fullmodel;i++) {
    Gi(1,2*i) = 1;
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    Gi(1,i+nSup_fullmodel) = 1;
  }
  if (debug) Gi.print("Gi=");

  mat Ident_Phi(q, p+q);
  Ident_Phi.submat(0,0,q-1,q-1) = eye<mat>(q,q);
  Ident_Phi.submat(0, q, q-1, q+p-1) = Fii_;
  if (debug) Ident_Phi.print("Ident_Phi=");
  gama = kronecker(Ident_Phi, Gi);
  if (debug) gama.print("gama=");

  //fii=(eye(q+p).*.((0~1~0)|(0~1~0)|(0~0~1))).*vec(( ((1-exp(-lamda1))./lamda1)~ exp(-lamda1)~exp(-lamda2)  )'); # med approx .*. kron .* punktvis

  fii = zeros<mat>(nLatent, nLatent); //samme som Fi paaside 16 i paperet
  for (int i=0;i<nSup_fullmodel;i++) {
    fii(2*i, 2*i+1) = 1;
    fii(2*i+1, 2*i+1) = 1;
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    fii(nSup_fullmodel+i, nSup_fullmodel+i) = 1;
  }
  if (debug) fii.print("fii=");

  //fii= (eye(q+p).*.((0~1)|(0~1))).*vec(( ((1-exp(-lamda))./lamda)~ exp(-lamda) )');
  //fii=(eye(q+p).*.((0~1~0)|(0~1~0)|(0~0~1))).*vec(( ((1-exp(-lamda1))./lamda1)~ exp(-lamda1)~exp(-lamda2))');

  vec fii_lambda = zeros<vec>(nLatent*(p+q));
  int ind=0;  
  for (int k=0;k<p+q;k++) {
    for (int j=0;j<nSup_fullmodel;j++) {
      fii_lambda(ind++) = oneMinusExpMinusLambda(k,j)/lambda(k,j);
      fii_lambda(ind++) = expMinusLambda(k,j);
    }
    for (int j=nSup_fullmodel;j<nSup;j++) {
      fii_lambda(ind++) = expMinusLambda(k,j);
    }
  }
  if (debug) fii_lambda.print("fii_lambda=");
  fii = kronecker(eye(q+p, q+p), fii);
  if (debug) fii.print("fii=");
  const int n_fii = fii.n_cols;
  for (int i=0;i<n_fii;i++) {
    fii.col(i) = fii.col(i) % fii_lambda;
  }
  if (debug) fii.print("fii=");




  //  tau=my[1]|( my[1]^2+psi[1]+psi[3])|my[2]|( my[2]^2+psi[2]+fi21*psi[3]);

  tau = zeros<vec>(nEq);
  tau(0)= mu(0);
  tau(1) = psi(0) + mu(0)*mu(0) + psi(2);
  tau(2) = mu(1);
  tau(3) = psi(1) + mu(1)*mu(1) + phi21*psi(2);

  if (debug) tau.print("tau=");

  Qu = zeros<mat>(nLatent*(q+p), nLatent*(q+p));
  for (int k=0;k<q+p;k++) {
    int I0 = k*nLatent;
    for (int i=0;i<nSup_fullmodel;i++) {
      Qu(I0+2*i,I0+2*i) = 2*(omega(k,i)/lambda2(k,i))*(-1.5-0.5*expMinusTwoLambda(k,i) + 2*expMinusLambda(k,i) + lambda(k,i));
      Qu(I0+2*i,I0+2*i+1) = 2*(omega(k,i)/lambda(k,i))*(oneMinusExpMinusLambda(k,i) -0.5*oneMinusExpMinusTwoLambda(k,i));
      Qu(I0+2*i+1,I0+2*i) = Qu(I0+2*i,I0+2*i+1);
      Qu(I0+2*i+1,I0+2*i+1) = omega(k,i)*oneMinusExpMinusTwoLambda(k,i);
    }
    for (int i=nSup_fullmodel;i<nSup;i++) {
      Qu(I0+nSup_fullmodel+i, I0+nSup_fullmodel+i) = 2*omega(k,i)*lambda(k,i);
    }
  }

  if (debug) Qu.print("Qu=");

  const vec sumPsi = psi;

  sigma = zeros<mat>(2*q,2*q); 
  ind=0;
  for (int k=0;k<q;k++) {
    mat sigmak = zeros<mat>(2,2); // nEq x nEq
    sigmak(0,0) = sumPsi(k);
    sigmak(0,1) = 2*mu(k)*sumPsi(k);
    sigmak(1,0) = sigmak(0,1);
    const rowvec tmp0 = omega.row(k)/lambda2.row(k);
    const rowvec tmp10 = expMinusLambda.row(k);
    const rowvec tmp11 = lambda.row(k);
    const rowvec tmp1 = tmp10 - 1 + tmp11; //expMinusLambda.row(k)-1+lambda.row(k);
    const rowvec tmp = 4*tmp0 % tmp1;
    sigmak(1,1) = accu(tmp) + 2*psi(k)*psi(k) + 4*psi(k)*mu(k)*mu(k);
    if (k==0)
      sigmak(1,1) = sigmak(1,1) + 4*psi(k)*psi(2);
    else
      sigmak(1,1) = sigmak(1,1) + 4*phi21*phi21*psi(k)*psi(2);

    if (debug) sigmak.print("sigmak=");
    
    sigma.submat(ind,ind,ind+1,ind+1) = sigmak;
    ind += 2;
  }
  if (debug) sigma.print("sigma=");


  //  Omeg1=diagrv(eye(2),psi[3]|(sumr(4*(omega[3,.]./lamda[3,.].^2).*(exp(-lamda[3,.])-1+lamda[3,.]))+2*psi[3]^2));
  mat Omeg1 = eye<mat>(2,2);
  Omeg1(0,0) = psi(2);
  //  const rowvec omega_tmp0 = expMinusLambda.row(2);
  //  const rowvec omega_tmp1 = -1 + lambda.row(2);
  const rowvec omega_tmp = lambda.row(2) - oneMinusExpMinusLambda.row(2); //omega_tmp0 + omega_tmp1;
  const rowvec omega22 = 4*(omega.row(2)/lambda2.row(2)) % omega_tmp;
  Omeg1(1,1) = sum(omega22) + 2*psi(2)*psi(2);
  if (debug) Omeg1.print("Omeg1=");

  sigma = sigma + F1*Omeg1*trans(F1);  // eq 15, page 9
  if (debug) sigma.print("sigma=");

  ///////////////////////////////////////////////////////

  mat GG_ = gama;
  int nGamma = gama.n_cols;
  V_ = zeros<mat>(nGamma,nGamma*(ap+1)); // Hvilket navn skal vi gi 'nGamma'?
  V = V_;

  A_ = zeros<mat>(nGamma,(ap+1));
  A = A_;

  
  V_.cols(1, nGamma) = 0*Qu; // sjekk
  if (debug) Rprintf("QL::filter_multivariat 1\n");


  //mat Z0sub = Z0.cols(0,8);
  //Z0sub.print("Z0=");
//  tau.print("tau=");
//  Qu.print("Qu=");
//  gama.print("gama=");
//  sigma.print("sigma=");
//  fii.print("fii=");
 

  f = 0;

  for (int t=0;t<ap;t++) {    
    int tm1 = nGamma*t - 1; // limits in V
    int t0 = nGamma*(t+1) - 1;
    int tp1 = nGamma*(t+2) - 1;
    
    vec Et = Z0.col(t) - tau - GG_ * A_.col(t);
    if (debug==2)    Et.print("Et=");
    mat Dt = GG_*V_.cols((tm1+1),t0)*trans(GG_) + sigma;
    if (debug==2)    Dt.print("Dt=");
    mat iDt = inv(Dt);
    if (debug==2)    iDt.print("iDt=");

    mat Kt = V_.cols((tm1+1),t0) * trans(GG_) * iDt;
    if (debug==2)    Kt.print("Kt=");
    
    mat tmpmat =  Kt * Et;
    A.col(t) = A_.col(t) + tmpmat;
    A_.col(t+1) = FF(t+1,fii)*A.col(t);
    
    V.cols((tm1+1),t0) = V_.cols((tm1+1),t0) - Kt*GG_*V_.cols((tm1+1),t0);
    V_.cols((t0+1),tp1) = FF(t+1,fii)*V.cols((tm1+1),t0)*trans(FF(t+1,fii)) + Qu;



    //    vec Dt.eigenvalues = eigen(Dt).values;
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, Dt);
    //    eigval.print("eigval=");
    if (min(eigval) < 0) {
      //      print(Dt)
      //      print(Dt.eigenvalues)
      //      stop("Negative eigenvalues")
      f = Inf;
      break;
    }
    vec val =  0.5*accu(log(eigval)) + 0.5*trans(Et)*iDt*Et;
    f = f + val(0);
    if (debug==2) Rprintf("f=%8.6f\n", f);

    if (t==3 && debug==2)
      exit(-1);
  }
  mat Asub = A.cols(0,4);
  if (debug) Asub.print("A=");
  mat Asub_ = A_.cols(0,4);
  if (debug) Asub_.print("A_=");
  mat Vsub = V.submat(0,nGamma, nGamma-1, 2*nGamma-1);
  if (debug) Vsub.print("V=");
  mat Vsub_ = V_.submat(0,nGamma, nGamma-1, 2*nGamma-1);
  if (debug) Vsub_.print("V_=");
  if (debug) Rprintf("(before penalty) f=%8.4f\n", f);

  // Add penalty function
  if (addPenalty) {
    //    if (nSup > 1) {
    //    f += 100*max(0, (par(1)-3))^3 + 100*max(0,-(3+par(1)))^3 + 100*max(0,(par(2)-3))^3 + 100*max(0,-(3+par(2)]))^3;   /* straffer c[2:3] utenfor [-3,+4]; */
    ind = q; // first lambda index
    for (int k=0;k<p+q;k++) {
      for (int i=0;i<nSup;i++) {
	double pen = 100*(MAX3(par(ind)-penaltyMax) + MAX3(penaltyMin - par(ind)));
	f += pen;
	ind++;
      }
    }
  }

  if (debug)  Rprintf("Quit QL::filter_multivariat  f=%8.6f\n", f);

  return;
}



void QL::smoother(const vec & par, const mat & a, const mat & a_, const mat & V, const mat & V_,
		  mat & b, mat & W, mat & BBt) {
  Parameters par_extract(par, transf);
  vec lambda = par_extract.lambda;

  int nSup = lambda.n_elem;
  int ap = a.n_cols - 1;

  int nSup_fullmodel; // number of terms with exact state space model representation
  if (nSup == 1)
    nSup_fullmodel = nSup;
  else
    nSup_fullmodel = nSup - 1;
  const int nSup_approx = nSup - nSup_fullmodel;
  const int nLatent = 2*nSup_fullmodel + nSup_approx;

  b = zeros<mat>(nLatent,ap);
  W = zeros<mat>(nLatent,nLatent*ap);
  BBt = zeros<mat>(nLatent,nLatent*ap);

  b.col(ap-1) = a.col(ap-1);
  W.cols((nLatent*(ap-1)),(nLatent*ap-1)) = V.cols((nLatent*(ap-1)),(nLatent*ap-1));

  mat fii = zeros<mat>(nLatent, nLatent);
  //  fii.diag() = exp(-lambda);
  vec expMinusLambda = exp(-lambda);
  vec oneMinusExpMinusLambda = 1 - expMinusLambda;
  for (int i=0;i<nSup_fullmodel;i++) {
    fii(2*i, 2*i+1) = oneMinusExpMinusLambda(i)/lambda(i);
    fii(2*i+1, 2*i+1) = expMinusLambda(i);
  }
  for (int i=nSup_fullmodel;i<nSup;i++) {
    fii(nSup_fullmodel+i, nSup_fullmodel+i) = expMinusLambda(i);
  }

  for (int i=0;i<=ap-2;i++) {    //  do until i>ap-2
    int t = ap-i;

    int t2 = nLatent*(t-2) - 1;
    int t1 = nLatent*(t-1) - 1;
    int t0 = nLatent*t - 1;
    t = t-1;

    mat y1 = inv(V_.cols((t1+1),t0)); //invswp;

    mat Bt = V.cols((t2+1),t1)*trans(FF(t,fii)) * y1;

    BBt.cols((t1+1),t0) = Bt;


    vec tmpvec = b.col(t)-a_.col(t);
    b.col(t-1) = a.col(t-1) + conv_to<vec>::from(Bt * tmpvec);
    mat diff = W.cols(t1+1,t0) - V_.cols(t1+1,t0);
    mat tmpmat =  Bt*diff*trans(Bt);
    W.cols(t2+1,t1) =  V.cols(t2+1,t1) + tmpmat;

    //    y1.print("y1=");
    //    Bt.print("Bt=");
    //    tmpmat.print("tmpmat=");
    //    mat Wsub = W.cols(t2+1,(nSup*ap-1));
    //    Wsub.print("W=");
    //    mat bt = b.cols(t-1,ap-1);
    //    bt.print("b[,t-1:ap-1]=");
  }

  //  mat bsub = b.cols(0,3);
  //  bsub.print("b=");
  //  mat Wsub = W.cols(0,3);
  //  Wsub.print("W=");
  //  mat BBtsub = BBt.cols(0,8);
  //  BBtsub.print("BBt=");

  //  Rprintf("Quit QL::smoother\n");

  return;
}

void QL::smoother_multivariate(const vec & par, const mat & a, const mat & a_, const mat & V, const mat & V_, const mat & fii,
			       mat & b, mat & W, mat & BBt) {
  ParametersMulti par_extract(par, transf);
  mat lambda = par_extract.lambda;

  int ap = a.n_cols - 1;

  const int nGamma = a.n_rows;

  b = zeros<mat>(nGamma,ap);
  W = zeros<mat>(nGamma,nGamma*ap);
  BBt = zeros<mat>(nGamma,nGamma*ap);

  b.col(ap-1) = a.col(ap-1);
  W.cols((nGamma*(ap-1)),(nGamma*ap-1)) = V.cols((nGamma*(ap-1)),(nGamma*ap-1));

  //  mat fii = zeros<mat>(nLatent, nLatent);
  mat expMinusLambda = exp(-lambda);
  mat oneMinusExpMinusLambda = 1 - expMinusLambda;
  //  for (int i=0;i<nSup_fullmodel;i++) {
  //    fii(2*i, 2*i+1) = oneMinusExpMinusLambda(i)/lambda(i);
  //    fii(2*i+1, 2*i+1) = expMinusLambda(i);
  //  }
  //  for (int i=nSup_fullmodel;i<nSup;i++) {
  //    fii(nSup_fullmodel+i, nSup_fullmodel+i) = expMinusLambda(i);
  //  }

  for (int i=0;i<=ap-2;i++) {    //  do until i>ap-2
    int t = ap-i;

    int t2 = nGamma*(t-2) - 1;
    int t1 = nGamma*(t-1) - 1;
    int t0 = nGamma*t - 1;
    t = t-1;

    mat y1 = inv(V_.cols((t1+1),t0)); //invswp;

    mat Bt = V.cols((t2+1),t1)*trans(FF(t,fii)) * y1;

    BBt.cols((t1+1),t0) = Bt;


    vec tmpvec = b.col(t)-a_.col(t);
    b.col(t-1) = a.col(t-1) + conv_to<vec>::from(Bt * tmpvec);
    mat diff = W.cols(t1+1,t0) - V_.cols(t1+1,t0);
    mat tmpmat =  Bt*diff*trans(Bt);
    W.cols(t2+1,t1) =  V.cols(t2+1,t1) + tmpmat;

    //    y1.print("y1=");
    //    Bt.print("Bt=");
    //    tmpmat.print("tmpmat=");
    //    mat Wsub = W.cols(t2+1,(nSup*ap-1));
    //    Wsub.print("W=");
    //    mat bt = b.cols(t-1,ap-1);
    //    bt.print("b[,t-1:ap-1]=");
  }

  //  mat bsub = b.cols(0,3);
  //  bsub.print("b=");
  //  mat Wsub = W.cols(0,3);
  //  Wsub.print("W=");
  //  mat BBtsub = BBt.cols(0,8);
  //  BBtsub.print("BBt=");

  //  Rprintf("Quit QL::smoother\n");

  return;
}

void QL::confidenceIntervals(const vec & estimate, const mat & H,
			     mat & Hi,
			     Parameters & sd,
			     Parameters & lower, Parameters & upper,
			     Parameters & lowerUn, Parameters & upperUn, 
			     const int sandwich,
			     const int deltaMethod) {
  const int debug=0;
  if (sandwich) {
    mat gr = qlExtern->quasiLikelihood_individual(estimate);
    if (debug) {
      H.print("H_ql=");
      vec gr_mean = mean(gr, 1);
      gr_mean.print_trans("gr_ql_mean=");
      mat gr_var = var(gr, 0, 1);
      gr_var.print("gr_ql_var=");
    }

    Hi = computeSandwichMatrix(H, gr, Z0.n_rows);
    if (debug) {
      Hi.print("Sandwich (QL)=");
    }
  }
  else {
    Hi = inv(H);
  }
  // Extract confidence intervals for parameters
  confidenceIntervals(estimate, Hi, sd, lower, upper, lowerUn, upperUn, deltaMethod);
}

void QL::confidenceIntervalsMulti(const vec & estimate, const mat & H,
				  mat & Hi,
				  ParametersMulti & sd,
				  ParametersMulti & lower, ParametersMulti & upper,
				  ParametersMulti & lowerUn, ParametersMulti & upperUn, 
				  const int sandwich) {
  if (sandwich) {
    mat gr = qlExtern->quasiLikelihoodMulti_individual(estimate);

    Hi = computeSandwichMatrix(H, gr, nObs);
  }
  else {
    Hi = inv(H);
  }
  // Extract confidence intervals for parameters
  confidenceIntervalsMulti(estimate, Hi, sd, lower, upper, lowerUn, upperUn);
}

void QL::confidenceIntervals(const vec & estimate, const mat & Hi,
			     Parameters & sd,
			     Parameters & lower, Parameters & upper,
			     Parameters & lowerUn, Parameters & upperUn, 
			     const int deltaMethod) {
  const int debug=0;
  const double q0_975 = 1.959964;
  //  mat Hi = inv(H);
  const vec var_transf = Hi.diag();
  const mat varmat_transf = Hi; //diagmat(var_transf);
  const vec sd_transf = sqrt(var_transf);

  mat Hi_sqrt = trans(robustCholesky(Hi));

  if (debug) {
    Hi.print("Hi=");
    Hi_sqrt.print("Hi_sqrt=");
  }
  const vec lowervec = estimate - q0_975*sd_transf;
  const vec uppervec = estimate + q0_975*sd_transf;

  lowerUn.setPars(lowervec, NOTRANSF);
  upperUn.setPars(uppervec, NOTRANSF);

  if (debug) {
    estimate.print_trans("estimate=");
    sd_transf.print_trans("sd_transf=");
    lowervec.print_trans("lowervec=");
    uppervec.print_trans("uppervec=");

    lower.setPars(lowervec, transf);
    upper.setPars(uppervec, transf);

    Rprintf("Transformed confidence interval (wrong for lambda2):\n");
    Rprintf("Lower:\n");
    lower.print();
    Rprintf("Upper:\n");
    upper.print();
  }

  const int npar = estimate.n_elem;
  const int nsup = Parameters::numberOfSuperPositions(estimate);

  vec varvec;

  Parameters x(nsup);
  const int nsim=40000;
  vec sumx = zeros<vec>(npar);
  vec sumx2 = zeros<vec>(npar);
  mat xmat = zeros<mat>(npar,nsim);
  int ind=0;
  //  GetRNGstate();
  for (int i=0;i<nsim/2;i++) {
    // Draw random normal numbers with mean=estimate, sd=sd_transf
    const vec eps = normal(npar);
    const vec Hi_sqrtTimesEps = Hi_sqrt * eps;
    //      const vec u2 = estimate + eps % sd_transf;
    for (int j=-1;j<2;j+=2) { // -1, 1
      const vec u = estimate + j * Hi_sqrtTimesEps;

      // Transform u
      x.setPars(u, transf);
      const vec xvec = x.asvector();
      xmat.col(ind) = xvec;
      sumx = sumx + xvec;
      sumx2 = sumx2 + xvec%xvec;
      ind++;
    }
  }
  //  PutRNGstate();
  varvec = (sumx2 - (1.0/nsim)*sumx%sumx)/(nsim-1.0);
  
  const double p = 0.025;
  const int iLower = nsim*p - 1;
  const int iUpper = nsim - iLower - 1;

  mat xsort = sort(xmat, 0, 1);
  vec xlower = xsort.col(iLower);
  vec xupper = xsort.col(iUpper);
  if (debug) {
    xlower.print_trans("xlower=");
    xupper.print_trans("xupper=");
  }
  lower.setPars(xlower, NOTRANSF);
  upper.setPars(xupper, NOTRANSF);

  if (deltaMethod) {
    Parameters p(estimate, transf);
    mat F = p.gradient(transf);
    //    F.print("F=");
    mat varmat = F * varmat_transf * trans(F);
    //    varmat.print("varmat=");
    varvec = varmat.diag();
    //    F.print("F=");
    //    varvec.print("varvec=");
  }
  vec sdvec = sqrt(varvec);
  sd.setPars(sdvec, NOTRANSF);

  if (debug) {
    sdvec.print_trans("sdvec=");
    sd.print();
  }
}

void QL::confidenceIntervalsMulti(const vec & estimate, const mat & Hi,
				  ParametersMulti & sd,
				  ParametersMulti & lower, ParametersMulti & upper,
				  ParametersMulti & lowerUn, ParametersMulti & upperUn) {
  const int debug=0;
  const double q0_975 = 1.959964;
  //  mat Hi = inv(H);
  const vec var_transf = Hi.diag();
  const mat varmat_transf = Hi; //diagmat(var_transf);
  const vec sd_transf = sqrt(var_transf);

  mat Hi_sqrt = trans(robustCholesky(Hi));

  if (debug) {
    Hi.print("Hi=");
    Hi_sqrt.print("Hi_sqrt=");
  }
  vec lowervec = estimate - q0_975*sd_transf;
  vec uppervec = estimate + q0_975*sd_transf;

  lowerUn.setPars(lowervec, NOTRANSF);
  upperUn.setPars(uppervec, NOTRANSF);

  if (debug) {
    estimate.print_trans("estimate=");
    sd_transf.print_trans("sd_transf=");
    lowervec.print_trans("lowervec=");
    uppervec.print_trans("uppervec=");

    lower.setPars(lowervec, transf);
    upper.setPars(uppervec, transf);

    Rprintf("Transformed confidence interval (wrong for lambda2):\n");
    Rprintf("Lower:\n");
    lower.print();
    Rprintf("Upper:\n");
    upper.print();
  }

  const int npar = estimate.n_elem;
  const int nsup = ParametersMulti::numberOfSuperPositions(estimate);

  ParametersMulti x(nsup);
  const int nsim=40000;
  vec sumx = zeros<vec>(npar);
  vec sumx2 = zeros<vec>(npar);
  mat xmat = zeros<mat>(npar,nsim);
  int ind=0;
  //  GetRNGstate();
  for (int i=0;i<nsim/2;i++) {
    // Draw random normal numbers with mean=estimate, sd=sd_transf
    const vec eps = normal(npar);
    const vec Hi_sqrtTimesEps = Hi_sqrt * eps;
    //    const vec u = estimate + eps % sd_transf;
    for (int j=-1;j<2;j+=2) { // -1, 1
      const vec u = estimate + j * Hi_sqrtTimesEps;
      
      // Transform u
      x.setPars(u, transf);
      const vec xvec = x.asvector();
      xmat.col(ind) = xvec;
      sumx = sumx + xvec;
      sumx2 = sumx2 + xvec%xvec;
      ind++;
    }
  }
  //  PutRNGstate();
  vec varvec = (sumx2 - (1.0/nsim)*sumx%sumx)/(nsim-1.0);

  const double p = 0.025;
  const int iLower = nsim*p - 1;
  const int iUpper = nsim - iLower - 1;

  mat xsort = sort(xmat, 0, 1);
  vec xlower = xsort.col(iLower);
  vec xupper = xsort.col(iUpper);
  if (debug) {
    xlower.print_trans("xlower=");
    xupper.print_trans("xupper=");
  }
  lower.setPars(xlower, NOTRANSF);
  upper.setPars(xupper, NOTRANSF);

  vec sdvec = sqrt(varvec);
  sd.setPars(sdvec, NOTRANSF);

  if (debug) {
    sdvec.print_trans("sdvec=");
    sd.print();
  }
}



void QL::constrainGradient(mat & gr) {
  //  Rprintf("gradMax=%8.4f\n", gradMax);
  rowvec grt = sum(gr % gr,0);
  int ap = gr.n_cols;
  //  Rprintf("ap=%d\n", ap);
  int ind=0;
  for (int t=0;t<ap;t++) {
    if (grt(t) < gradMax) {
      gr.col(ind++) = gr.col(t);
    }
    else if (0) {
      Rprintf("grt(%d)=%6.4f\n", t, grt(t));
    }
  }

  //  Rprintf("ind=%d, ap=%d\n", ind, ap);
  gr = gr.cols(0, ind-1);
  double mult = ((double) ap)/((double) ind);
  gr = mult * gr;
}

mat QL::computeSandwichMatrix(const mat & H, mat & gr, const int nObs) {
  const int debug=0;
  if (debug) {
    vec grmean= mean(gr, 1);
    grmean.print_trans("grmean=");
    vec grmax = max(gr, 1);
    double x = max(grmax);
    Rprintf("Max gr: %6.4f\n", x);
    vec grmin = min(gr, 1);
    x = min(grmin);
    Rprintf("Min gr: %6.4f\n", x);
  }

  constrainGradient(gr);

  if (debug) {
    vec grmax = max(gr, 1);
    double x = max(grmax);
    Rprintf("Max gr: %6.4f\n", x);
    vec grmin = min(gr, 1);
    x = min(grmin);
    Rprintf("Min gr: %6.4f\n", x);

    int npar = H.n_rows;
    
    mat dftsum = zeros<mat>(npar,npar);
    for (int t=0; t< 50;t++) {
      Rprintf("t=%d\n", t);
      const vec grt = gr.col(t);
      grt.print_trans("gr(t)=\n");
      const mat dft = grt * trans(grt);
      dftsum = dftsum + dft;
    dftsum.print("dft=");
    }

    dftsum = zeros<mat>(npar,npar);
    int ap = gr.n_cols;
    for (int t=0; t< ap;t++) {
      const vec grt = gr.col(t);
      const mat dft = grt * trans(grt);
      dftsum = dftsum + dft;
      if (t == 999 || t==1999 || t == ap-1) {
	dftsum.print("dft(999)=");
      }
    }
    dftsum.print("dftsum=");
  }

  mat I = gr * trans(gr) / nObs;
  mat Ji = inv(H) * nObs;

  if (debug) {
    mat df= I*nObs;
    df.print("df=");
    mat Hi = inv(H);
    Hi.print("Hi=");
  }

  mat S = Ji*I*Ji/nObs;


  if (debug) {
    Ji.print("Ji_ql=");
    I.print("I_ql=");
    const mat Sdiag = sqrt(S.diag());
    Sdiag.print_trans("sqrt(S.diag_ql())=");
  }

  return S;
}
