/*Include Files:*/
//#include <iostream>

#include <R.h>
#include <Rmath.h>

#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

#include "basic.h"
#include "math.h"
#include "optimise.h"
#include "conjugate.h"

// All these header files needed for the indirectExtern->resetPar() call...
#include "parameters.h"
#include "parametersMulti.h"
#include "bfgs.h"
#include "simulate.h"
#include "ql.h"
#include "indirect.h"


#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d)
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?	\
		   (maxarg1) : (maxarg2))
static double maxarg1, maxarg2;

int ncom;
vec pcom;
vec xicom;
double (*nrfunc)(const vec &, int &);

ConjugateDirection::ConjugateDirection(const int initialSteepestDescent_,
				       const int print_level_) : print_level(print_level_),
								 initialSteepestDescent(initialSteepestDescent_),
								 maxRandomStep(2)
{
}

ConjugateDirection::~ConjugateDirection() {
}

vec ConjugateDirection::findSteepestDescent(double (*func)(const vec &, int &), vec p, const int npar, const double h) {
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


void ConjugateDirection::setConjugateDirection(double (*func)(const vec &, int &), vec p, mat & xi, const int npar, const double h) {
  if (initialSteepestDescent) {
    if (print_level >= 2) {
      Rprintf("Computes inital steepest descent direction\n");
    }
    vec pder = findSteepestDescent(func, p, npar, h);
    if (print_level >= 3) {
      pder.print("Steepest descent direction=");
    }
    mat u;
    computeOrthogonalVectors(u, pder);
    xi = u;

    if (print_level >= 2) {
      xi.print("Initial conjugate direction matrix (xi)=");
    }
  }
  else {
    xi = eye<mat>(npar, npar);
  }
  int error = checkMat(xi, "(setConjugateDirection) xi"); // Check if nan
}


void ConjugateDirection::conjugateDirection(double (*func)(const vec &, int &),
					    vec & par,
					    double & fret,
					    double ftol, double ftol_weak, mat & xi,
					    int & restart, int & error) {
  if (print_level >= 3) {
    Rprintf("Enter conjugateDirection\n");

    par.print("par=");
  }
  int npar = par.n_elem;
  vec p(npar);
  for (int i=0;i<npar;i++) {
    p(i) = par(i);
  }

  
  const double h = 0.01; // NB: must be tuned
  setConjugateDirection(func, p, xi, npar, h);

  //  double ftol = 1e-1;
  int iter;
  powell(p, xi, npar, ftol, ftol_weak, iter, fret, func, restart, error);

  if (error) {
    return;
  }

  if (print_level >= 1) {
    Rprintf("Number of iterations in powell %d\n", iter);
    Rprintf("Function value %8.4f\n", fret);
  }

  for (int i=0;i<npar;i++) {
    par(i) = p(i);
  }
  //  delete [] p;

  return;
}
void ConjugateDirection::computeOrthogonalVectors(mat & u, const vec & pder) {
  const int n = pder.n_elem;

  int ind = 0;
  vec pder_abs = abs(pder);
  double pder_max = pder_abs(0);
  for (int i=1;i<n;i++) {
    if (pder_abs(i) > pder_max) {
      pder_max = pder_abs(i);
      ind = i;
    }
  }

  mat v = zeros<mat>(n,n);
  v.col(0) = pder/norm(pder,2);
  int index=1;
  for (int i=0;i<n;i++) {
    if (i != ind) {
      v(i, index++) = 1.0;
    }
  }

  u = zeros<mat>(n,n);
  gramSchmidt(u, v);

  //  cout.precision(11);

  //  pder.print("pder=");
  //  v.print("v=");
  //  u.raw_print("u=");
}

void ConjugateDirection::powell(vec & p, mat & xi, int n, double ftol, double ftol_weak, int & iter, double & fret,
				double (*func)(const vec &, int &), int & restart, int & error)
{
  int i,ibig,j;
  double del,fp,fptt;

  const double ftol_noImprovement = 1e-6;
  const double ftol_smallImprovement = 2e-1;
  const double deltaRandom = 0.01;
  const int ITMAX = 200; //#define ITMAX 200

  int tryRandomStep = 0;
  int iRandomStep = 0;
  int maxRandomStep = 3;
  restart = 0; /* default */
  error = 0;

  mat uRandomStep(maxRandomStep, n);
  for (int i=0;i<maxRandomStep;i++) {
    for (int j=0;j<n;j++) {
      uRandomStep(i,j) = norm_rand();
    }
    uRandomStep.row(i) = deltaRandom * uRandomStep.row(i) / norm(uRandomStep.row(i), 2);
  }
  //  uRandomStep.print("uRandomStep=");

  double f_step1;
  vec pr(n);
  vec gam(n);
  vec pt(n);
  vec p_step1(n);
  //  double * ptt = new double[n+1];
  vec xit(n);//  double * xit = new double[n+1];
  vec steplength = 0.01 * ones<vec>(n); // initial step length in bracketing

  int status;

  indirectExtern->resetDistMin();

  fret=(*func)(p, status);

  if (status == EXIT_FAILURE) {
    error = 1;
    restart = 1;

    //    delete  [] ptt;

    return;
  }

  for (j=0;j<n;j++) {
    pt(j) = p(j);
  }

  int iterSinceReset = 0;
  for (iter=1;iter<=ITMAX;++iter) {
    if (print_level >= 1) {
      Rprintf("powell iter %2d  fret=%7.3f  p=", iter, fret);
      for (int i=0;i<n;i++) {
	Rprintf("%6.3f ", p(i));
      }
      Rprintf("\n");
    }
    fp = fret;
    ibig = 0;
    del = 0.0;
    for (i=0;i<n;i++) { // n minimisations along the n conjugate directions

      for (j=0;j<n;j++)
	xit(j) = xi(j,i); // direction vector

      fptt = fret;

      if (print_level >= 3) {
	Rprintf("powell. Before linmin\n");
	for (int kk=0;kk<n;kk++) {
	  Rprintf("%6.4f ", p(kk));
	}
	Rprintf("\n");
      }

      linmin(p, xit, n, fret, func, steplength(i), error);

      if (error) {
	restart = 1;
	return;
      }

      if (steplength(i) > 1.0)
	steplength(i) = 1.0;
      else if (steplength(i) < 0.01)
	steplength(i) = 0.01;

      if (print_level >= 3) {
	Rprintf("powell. After linmin\n");
	for (int kk=0;kk<n;kk++) {
	  Rprintf("%6.4f ", p(kk));
	}
	Rprintf("\n");
      }
      if (fabs(fptt - fret) > del) {
	del = fabs(fptt-fret);
	ibig = i;
	//	Rprintf("powell: ibig=%d del=%8.6f\n", ibig, del);
      }
    }
    /* Original convergence criteria
       if (2.0*fabs(fp-(*fret)) <= ftol*(fabs(fp)+fabs(*fret))) {
       return;
       }
    */

    /* Special convergence criteria: f < ftol */
    //    int converged = fret < ftol;
    const double eps = 1e-3;
    int largerFunc = fret > fp + eps;
    int converged = fret < ftol;
    int noImprovement = fabs(fp-fret) < 0.5*ftol_noImprovement*(fabs(fp)+fabs(fret));
    int smallImprovement = fabs(fp-fret) < 0.5*ftol_smallImprovement*(fabs(fp)+fabs(fret));
    int weaklyConverged = fret < ftol_weak;
    int resetConjugateDirection = 0;

    if (tryRandomStep) {
      tryRandomStep = 2;
    }
    else if (converged || (noImprovement && weaklyConverged)) {
      if (noImprovement && weaklyConverged) {
	Rprintf("Weak convergence in powell\n");
      }

      return;
    }
    else if (iter >= ITMAX) {
      /*      nrerror("powell exceeding maximum iterations."); */
      Rprintf("powell exceeding maximum iterations\n");
      Rprintf("No convergence in powell\n");
      
      restart = 1;
      return;
    }
    else if (largerFunc) {
      Rprintf("Unexpected increase in function value during powell. fret=%10.6f  fprev=%10.6f\n", fret, fp);
      Rprintf("Parameter value\n");
      for (int kk=0;kk<n;kk++) {
	Rprintf("%6.4f ", p(kk));
      }
      Rprintf("\n");
      Rprintf("Reset parameter to previous best value\n");
      indirectExtern->resetToBestStartPar(p, fret);
      for (int kk=0;kk<n;kk++) {
	Rprintf("%6.4f ", p(kk));
      }
      Rprintf("\n");
      
      if (iterSinceReset > 1) {
	resetConjugateDirection = 1;
      }
      else {
	Rprintf("Try random step\n");
	tryRandomStep = 1;
	//	restart = 1;
	//	return;
      }
    }
    else if (noImprovement) { //  || (smallImprovement && iterSinceReset >= n)) {
      if (iterSinceReset == 1) {

	Rprintf("No improvement, try random step\n");

	tryRandomStep = 1;
	//	return;
      }
      else {
	resetConjugateDirection = 1;
      }
    }
    else if (smallImprovement && iterSinceReset >= n) {
      resetConjugateDirection = 1;
    }

    if (iter==-1) {
      tryRandomStep = 1;
      resetConjugateDirection = 0;
    }

    if (tryRandomStep==1) {
      if (iRandomStep < maxRandomStep) {
	//	Rprintf("random step not implemented\n");
	
	/* Pertubate old p --> pr */
	for (j=0;j<n;j++) {
	  gam(j) = uRandomStep(iRandomStep, j);
	}
	p_step1 = p;
	f_step1 = fret;
	for (j=0;j<n;j++) {
	  for (int k=0;k<n;k++) {
	    p(j) = pt(j) + xi(j,k)*gam(k);  /* random step */
	  }
	}
	iRandomStep++;
	//	double fr = func(pr);
      }
      else {
	Rprintf("iRandomStep=%d, maxRandomStep=%d\n", iRandomStep, maxRandomStep);
	Rprintf("No convergence in powell\n");
	restart = 1;

	return;
      }
    }
  
    int updateDirection = 0;
    if (tryRandomStep == 2) {
      updateDirection = 1;
      if (print_level >= 1) {
	Rprintf("Random step update\n");
	rowvec tmp = trans(p);
	tmp.print("  p=");
	tmp = trans(p_step1);
	tmp.print("  p_step1=");
	Rprintf("   fret=%8.6f f_step1=%8.6f fp=%8.6f xit=", fret, f_step1, fp);
      }
      if (fret > f_step1) { // Invariant: f(p) < f(p_step1)
	const vec tmp = p;
	p = p_step1;
	p_step1 = tmp;
	fret = f_step1;
      }
      indirectExtern->resetToBestStartPar(p, fret);
      for (j=0;j<n;j++) {
	//      ptt[j] = 2.0*p[j] - pt(j-1);
	xit(j) = p(j) - p_step1(j);
	pt(j) = p(j);
      }
      for (j=0;j<n;j++) {
	Rprintf(" %6.3f", xit(j));
      }
      
      Rprintf("\n");
    }
    else {
      for (j=0;j<n;j++) {
	//      ptt[j] = 2.0*p[j] - pt(j-1);
	xit(j) = p(j) - pt(j);
	pt(j) = p(j);
      }
    }
    int error= checkVec(xit, " (powell-1) xit");
    if (resetConjugateDirection) {
      iterSinceReset = 0;
      const double h = 0.001;
      setConjugateDirection(func, p, xi, n, h);
      steplength = 0.01 * ones<vec>(n); // initial step length in bracketing
    }
    else if (tryRandomStep == 0) {
      double normxit = norm(xit, 2);
      //      Rprintf("powell normxit=%10.8f\n", normxit);

      const double eps = 1e-6;
      if (normxit > eps) {
	//      xit.print("xit (1)=");
	//      Rprintf("normxit2=%8.6f\n", normxit);
	
	mat H = xi;
	for (j=0;j<n;j++) {
	  H(j,ibig) = H(j,n-1);
	  H(j,n-1) = xit(j)/normxit;
	}

	const double limit = 1e-3;
	vec s = svd(H);
	if (print_level >= 3) {
	  s.print("SVD values = ");
	}
	int fullrank = min(s) > limit;
	if (fullrank)
	  updateDirection = 1;
      }
    }
    if (updateDirection) {
      if (print_level >= 3) {
	Rprintf("powell: Update direction\n");
	Rprintf("powell: Before linmin 2: ");
	for (int kk=0;kk<n;kk++) {
	  Rprintf("%6.4f ", p(kk));
	}
	Rprintf("\n");
      }
      double steplength_new = 1.0;
      //      xit.print("xit (2)=");
      linmin(p, xit, n, fret, func, steplength_new, error);
      if (error) {
	restart = 1;
	return;
      }
      steplength(ibig) = steplength(n-1);
      steplength(n-1) = steplength_new;
      if (steplength(n-1) > 1.0)
	steplength(n-1) = 1.0;
      else if (steplength(n-1) < 0.01)
	steplength(n-1) = 0.01;
      steplength(n-1) = 0.01;
      if (print_level >= 3) {
	Rprintf("powell: After linmin 2: ");
	for (int kk=0;kk<n;kk++) {
	  Rprintf("%6.4f ", p(kk));
	}
	Rprintf("\n");
	//      xit.print("xit (3)=");
	Rprintf("powell: Normalise xit\n");
      }
      error = checkVec(xit, "(powell-2) xit");
      double normxit = norm(xit, 2); 
      //      Rprintf("powell normxit(1)=%10.8f\n", normxit);
      for (j=0;j<n;j++) {
	xit(j) /= normxit;
      }
      if (checkVec(xit, "(powell-3) xit")) {
	Rprintf("normxit=%6.4f\n", normxit);
      }
	

      for (j=0;j<n;j++) {
	xi(j, ibig) = xi(j, n-1);
	xi(j, n-1) = xit(j);
      }
      error = checkMat(xi, "(powell) xi");
      if (print_level >= 3) {
	xi.print("powell update xi:\n");
      }
    }
    if (tryRandomStep == 2) {
      tryRandomStep = 0;
    }
    iterSinceReset++;
  }
}

void ConjugateDirection::linmin(vec & p, const vec & xi, int n,
				double & fret, double (*func)(const vec &, int &), double & bx_start,
				int & error)
{
  int j;
  int status = EXIT_SUCCESS;
  double xx,xmin,fx,fb,fa,bx,ax;
  const double TOL = 1.0e-3;

  ncom=n;

  pcom = vec(n);

  xicom = vec(n);
  nrfunc=func;
  for (j=0;j<n;j++) {
    pcom(j) = p(j);
    xicom(j) = xi(j);
  }
  ax=0.0;
  //  xx=1.0;
  xx = bx_start;

  if (print_level >= 3) {
    Rprintf("powell-linmin: (before mnbrak) p: ");
    for (int kk=0;kk<n;kk++) {printf("%6.4f ", pcom(kk));}
    Rprintf("  f=%6.4f ax=%6.4f xx=%6.4f fa=%6.4f fx=%6.4f\n",
	    fret, ax, xx, fa, fx);
  }
  indirectExtern->resetDistMinBracket();
  mnbrak(ax, xx, bx, fa, fx, fb, status);
  if (status == EXIT_FAILURE) {
    Rprintf("powell-linmin: Terminate linmin\n");
    error = 1;
    return;
  }
  bx_start = bx;
  if (print_level >= 3) {
    Rprintf("powell-linmin: (after mnbrak)  p: ");
    for (int kk=0;kk<n;kk++) {printf("%6.4f ", pcom(kk));}
    Rprintf("  ax %6.4f xx %6.4f bx %6.4f fa %6.4f fx %6.4f fb %6.4f\n",
	    ax, xx, bx, fa, fx, fb);
  }

  indirectExtern->resetToBestBracketStartPar();
  
  error = 0;
  //  int iter=0;
  //  while (iter < 1 && error) {
  fret = brent(ax, xx, bx, TOL, xmin, status);
  if (status == EXIT_FAILURE) {
    Rprintf("powell-linmin: Terminate linmin\n");
    error = 1;
    //    break;
  }

  const double eps = 1e-3;
  if (fret > fx + eps) {
    Rprintf("powell-linmin: Error: f returned from brent %10.6f is larger than input f %10.6f\n", fret, fx);
    //      if (iter == 0) {
    //	Rprintf("powell-linmin: Tries a new start value (not implemented)\n");
    //	indirectExtern->resetToQLStartPar();
    //      }
  }
  //  else {
  //    error = 0;
  //  }
  //    iter++;
  //  }
  for (j=0;j<n;j++) {
    //    xi(j-1) *= xmin;
    p(j) += xi(j)*xmin;
  }
  if (print_level >= 3) {
    Rprintf("powell-linmin: (after brent)   p: ");
    for (int kk=0;kk<n;kk++) {
      Rprintf("%6.4f ", p(kk));
    }
    Rprintf("  f=%6.4f\n", fret);
  }
  bx_start = xmin;
  //  delete [] xicom; //free_vector(xicom,1,n);
  //  delete [] pcom; //free_vector(pcom,1,n);
}

double ConjugateDirection::f1dim(double x, int & status)
{
  //  xt=vector(1,ncom);
  vec xt(ncom); //double * xt = new double[ncom+1];

  for (int j=0;j<ncom;j++)
    xt(j) = pcom(j) + x*xicom(j);

  double f = (*nrfunc)(xt, status);

  //  delete [] xt;

  return f;
}



double ConjugateDirection::brent(double ax, double bx, double cx, double tol,
				 double & xmin, int & status)
{
  int iter;
  double a,b,d,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
  double e = 0.0;

  const int ITMAX = 100;
  const double CGOLD = 0.3819660;
  const double ZEPS = 1.0e-3;/*#define ZEPS 1.0e-10*/
  //  const double ZEPS = 1.0e-10;/*#define ZEPS 1.0e-10*/

  a = (ax < cx ? ax : cx);
  b = (ax > cx ? ax : cx);
  x = w = v = bx;
  fw = fv = fx = f1dim(x, status);
  if (status == EXIT_FAILURE) {
    return 0;
  }

  for (iter=1;iter<=ITMAX;iter++) {
    if (print_level >= 2) {
      Rprintf("powell-linmin-brent: iter %d x=%6.4f fx=%6.4f a=%6.4f b=%6.4f\n", iter, x, fx, a, b);
    }
    xm = 0.5*(a+b);
    tol1 = tol*fabs(x)+ZEPS;
    tol2 = 2.0*tol1;
    if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
      xmin = x;
      if (fabs(xmin) < 1e-6 && print_level >= 3)
	Rprintf("powell-linmin-brent(0): xmin %10.8f\n", xmin);

      return fx;
    }
    if (fabs(e) > tol1) {
      r = (x-w)*(fx-fv);
      q = (x-v)*(fx-fw);
      p = (x-v)*q-(x-w)*r;
      q = 2.0*(q-r);
      if (q > 0.0) p = -p;
      q = fabs(q);
      etemp = e;
      e = d; // NB: ‘d’ may be used uninitialized?!
      if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x)) {
	e = (x >= xm ? a-x : b-x);
	d = CGOLD*e;
      }
      else {
	d = p/q;
	u = x+d;
	if (u-a < tol2 || b-u < tol2)
	  d = SIGN(tol1,xm-x);
      }
    }
    else {
      e = (x >= xm ? a-x : b-x);
      d = CGOLD*e;
    }
    u = (fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
    fu = f1dim(u, status);
    if (status == EXIT_FAILURE) {
      return 0;
    }

    if (fu <= fx) {
      if (u >= x) a = x; else b = x;
      SHFT(v,w,x,u);
      SHFT(fv,fw,fx,fu);
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
	v = w;
	w = u;
	fv = fw;
	fw = fu;
      } else if (fu <= fv || v == x || v == w) {
	v = u;
	fv = fu;
      }
    }
  }
  if (print_level >= 1) {
    Rprintf("powell-linmin-brent: Too many iterations in brent");
  }
  xmin = x;
  if (fabs(xmin) < 1e-6 && print_level >= 3)
    Rprintf("powell-linmin-brent(1): xmin %10.8f\n", xmin);

  return fx;
}

void ConjugateDirection::mnbrak(double & ax, double & bx, double & cx, double & fa, double & fb, double & fc, int & status)
{
  double ulim,u,r,q,fu,dum;
  const int maxIter = 30;

  const double GOLD = 1.618034;
  const double GLIMIT = 100.0;
  const double TINY = 1.0e-20;

  fa = f1dim(ax, status);
  if (status == EXIT_FAILURE) {
    return;
  }
  fb = f1dim(bx, status);
  if (status == EXIT_FAILURE) {
    return;
  }
  if (fb > fa) {
    SHFT(dum,ax,bx,dum);
    SHFT(dum,fb,fa,dum);
  }
  cx = bx + GOLD*(bx-ax);
  fc = f1dim(cx, status);
  if (status == EXIT_FAILURE) {
    return;
  }
  if (print_level >= 3) {
    Rprintf("powell-linmin-mnbrak (init): ax=%8.4f bx=%8.4f cx=%8.4f fa%8.4f fb%8.4f fc%8.4f \n", ax, bx, cx, fa, fb, fc);
  }
  int index=0;
  int iter = 0;
  while (fb > fc && iter < maxIter) {
    if (print_level >= 3) {
      Rprintf("powell-linmin-mnbrak (%d): ax=%8.4f bx=%8.4f cx=%8.4f fa%8.4f fb%8.4f fc%8.4f \n", index++, ax, bx, cx, fa, fb, fc);
    }
    r = (bx-ax)*(fb-fc);
    q = (bx-cx)*(fb-fa);
    u = (bx)-((bx-cx)*q-(bx-ax)*r)/
      (2.0*SIGN(FMAX(fabs(q-r),TINY),q-r));
    ulim = (bx)+GLIMIT*(cx-bx);
    if (print_level >= 3) {
      Rprintf("powell-linmin-mnbrak (%d): u=%8.4f ulim=%8.4f\n", index-1, u, ulim);
    }
    if ((bx-u)*(u-cx) > 0.0) {
      fu = f1dim(u, status);
      if (status == EXIT_FAILURE) {
	return;
      }
      if (fu < fc) {
	ax = bx;
	bx = u;
	fa = fb;
	fb = fu;
	return;
      } else if (fu > fb) {
	cx = u;
	fc = fu;
	return;
      }
      u = cx+GOLD*(cx-bx);
      fu = f1dim(u, status);
      if (status == EXIT_FAILURE) {
	return;
      }
    } else if ((cx-u)*(u-ulim) > 0.0) {
      fu = f1dim(u, status);
      if (status == EXIT_FAILURE) {
	return;
      }
      if (fu < fc) {
	SHFT(bx,cx,u,cx+GOLD*(cx-bx));
	double fu_new = f1dim(u, status);
	if (status == EXIT_FAILURE) {
	  return;
	}
	SHFT(fb, fc, fu, fu_new);
      }
    } else if ((u-ulim)*(ulim-cx) > 0.0) { /* changed from >= */
      u = ulim;
      fu = f1dim(u, status);
      if (status == EXIT_FAILURE) {
	return;
      }
    } else {
      u = cx+GOLD*(cx-bx);
      fu = f1dim(u, status);
      if (status == EXIT_FAILURE) {
	return;
      }
    }
    SHFT(ax,bx,cx,u);
    SHFT(fa,fb,fc,fu);

    iter++;
  }

  if (iter == maxIter) {
    status = EXIT_FAILURE;
    if (print_level >= 1) {
      Rprintf("powell-linmin-mnbrak: Too many iterations in mnbrak\n");
    }
  }

  return;
}
#undef SHFT
#undef SIGN
