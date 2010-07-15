/*Include Files:*/
#include <iostream>

#include <R.h>

#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

#include "basic.h"
#include "optimise.h"
#include "bfgs.h"

static const int OK=0;
static const int NO_IMPROVEMENT=1;
static const int TOO_MANY_BRACKETS=2;
static const int TOO_MANY_SECTIONS=3;
//static double noImprovementValue=1e-10;

BFGS::BFGS(const double value) : rho(0.01), sigma(0.9), tau1(9), tau2(0.1), tau3(0.5) {
  noImprovementValue = value;
}

BFGS::BFGS() : rho(0.01), sigma(0.9), tau1(9), tau2(0.1), tau3(0.5) {
  noImprovementValue = 1e-10;
}

double cc(double z, double f0, double df0, double eta, double zeta) {
  return f0+(df0 + (eta + zeta*z)*z)*z;
}

//
//Kubisk interpolasjon
double BFGS::cinterpo(double a, double b, double b1, double b2,
		      double fb1, double fb2, double dfb1, double dfb2) {



  double f0 = fb1;
  double f1 = fb2;
  double bdiff = b2 - b1;
  double df0 = bdiff*dfb1;
  double df1 = bdiff*dfb2;
  double eta = 3*(f1-f0) - 2*df0-df1 ;
  double zeta = df0 + df1 - 2*(f1-f0);
  double h1 = (2*eta)/(3*zeta);
  double h2 = (4*df0)/(3*zeta);
  //  double s1 = min((a-b1)/bdiff, (b-b1)/bdiff);
  //  double s2 = max((a-b1)/bdiff, (b-b1)/bdiff);
  double s1 = (a-b1)/bdiff;
  double s2 = (b-b1)/bdiff;
  if (s1 > s2) {
    double tmp = s1;
    s1 = s2;
    s2 = tmp;
  }
  
  double z1;
  double z2;
  //  double h1_squared = h1*h1;
  double h12_diff = h1*h1 - h2;

  if (h12_diff >= 0) {
    z1 = (-h1-sqrt(h12_diff))/2;
    z2 = (-h1+sqrt(h12_diff))/2;
  }
  else {
    z1 = s1;
    z2 = s1;
  }
  if (z1<s1 || z1>s2) {
    z1 = s1;
  }
  if (z2<s1 || z2>s2) {
    z2 = s1;
  }

  vec cz(4);
  cz(0) = cc(s1, f0, df0, eta, zeta);
  cz(1) = cc(z1, f0, df0, eta, zeta);
  cz(2) = cc(z2, f0, df0, eta, zeta);
  cz(3) = cc(s2, f0, df0, eta, zeta);
  int alpha = argmin(cz);

  vec u(4);
  u(0) = s1;
  u(1) = z1;
  u(2) = z2;
  u(3) = s2;

  return b1+u(alpha)*bdiff;
}

double qv(double z, double b1, double b2, double fb1, double fb2, double dfb1) {
  return fb1 + dfb1 * (b2-b1) * z + (fb2 - fb1 - dfb1*(b2 - b1))*z*z;
}

//
//Kvadratisk interpolasjon
double BFGS::qinterpo(double a, double b, double b1, double b2,
		      double fb1, double fb2, double dfb1) {


  double bdiff = b2 - b1;
  double x = b1 - dfb1*bdiff*bdiff/(2*(fb2-fb1-dfb1*bdiff));

  if (x > max(a, b) || x < min(a, b)) {
    x = a;
  }
  vec u(3);
  u(0) = x;
  u(1) = a;
  u(2) = b;
  vec qz(3);
  qz(0) = qv((x-b1)/(b2-b1), b1, b2, fb1, fb2, dfb1);
  qz(1) = qv((a-b1)/(b2-b1), b1, b2, fb1, fb2, dfb1);
  qz(2) = qv((b-b1)/(b2-b1), b1, b2, fb1, fb2, dfb1);
  //  alfa = indcv(minc(qz),qz)
  int alfa = argmin(qz);
  
  return u(alfa);
}


//retningsprosedyre

void BFGS::retning(const vec & delta, const vec & gama, const vec & g, mat & Hi, vec & sk) {

  double x = sum(conv_to<vec>::from(delta % gama));
  mat Y = delta*trans(gama) * Hi;


  //  Hj = moment(trans(delta),0)/x
  mat Hj = (delta * trans(delta))/x; // A bit slower than GAUSS

  double tmp = arma::as_scalar(trans(gama)*Hi*gama);
  Hi = Hi + (1 + tmp/x)*Hj - (Y+trans(Y))/x;

  
  //  if (0) {
  //    Hi = Hi+(1+t(gama) %*% Hi%*%gama/x)%*%((delta%*%t(delta))/x) - (y+t(y))/(x)
  //  }

  vec Hig = Hi*g;
  double norm = as_scalar(sqrt(conv_to<rowvec>::from(trans(Hig)) * conv_to<vec>::from(Hig)));
  sk = -Hig/norm;
}





// generer lavere triangulaer matrise U fra GLm*vec(U)
mat BFGS::xpd(vec c, int d, int p) {
  //  local k,U,i,j

  mat U = zeros(d,p);
  int k = 0;

  int j = 1;
  while(j<= p) {//do until j>p
    int i = j;
    while (i<= d) { //do until i>d
      k++;
      U(i,j) = c(k);
      i++;
    }
    j++;
  }

  return U;
}
 
int BFGS::lineSearch(FunctionValue (*func)(const vec &, const int), double ai, double aj,
		     double fai, double faj, double dfai, double dfaj,
		     const vec & par0, const vec & sk, double myy, const vec & x0, double f0,
		     const vec & df0, int print_level, vec & g1, vec & x1, double & f1) {

  double b1, b2;
  double fb1, fb2;
  double dfb1, dfb2;


  double df00 = dfai;
  FunctionValue lik;
  
  vec df = df0;
  vec par = par0;

  // BRACKETING:
  // Find bracket interval [b1, b2]
  // For each iteration: ai, aj (j=i+1)

  // Initial values, ai=0, aj, fai, faj, dfai, dafj set in Optimal function before call to Linesearch

  int nBracket = 0;
  int terminateLoop = 0; // 1: terminate bracket, continue with sectioning, 2: terminate LineSearch
  if (print_level >= 2) {
    Rprintf("Bracketing:\n");
  }
  while(!terminateLoop) {    // BRACKETING

    if (print_level >= 2) {
      Rprintf("Bracket iter %3d ai %7.4f aj %7.4f fai %7.4f faj %7.4f dfai %7.4f dfaj %7.4f\n",
	      nBracket, ai, aj, fai, faj, dfai, dfaj);
    }
    nBracket = nBracket + 1;

    if (nBracket > 100) {
      Rprintf("Error(BFGS::lineSearch): Too many brackets\n");
      Rprintf("Error(BFGS::lineSearch): ai %7.4f aj %7.4f fai %7.4f faj %7.4f dfai %7.4f dfaj %7.4f\n",
	      ai, aj, fai, faj, dfai, dfaj);

      return TOO_MANY_BRACKETS;
      //      exit(-1);
    }

    if (0) { //if (!is.finite(faj)) {
      Rprintf("Not finite faj!");
      //ExtractPars(par).print();
      aj = ai + 0.5 * (aj-ai);
      par = x0+aj*sk;

      lik = func(par, 1);
      //      df = attr(faj, "gradient");
      faj = lik.f;
      df = lik.df;
      dfaj = sum(conv_to<vec>::from(df % sk));
    }
    else if (faj>f0+aj*rho*df00 || faj>fai) {
      b1 = ai;
      fb1 = fai;
      dfb1 = dfai;
      b2 = aj;
      fb2 = faj;
      dfb2 = dfaj;
      terminateLoop = 1; // Terminate BRACKET

      //        f = Likelihood(par, Z0=Z0)
      //        df = attr(f, "gradient")
      //    goto function
    }
    else {
      g1 = df;
      dfaj = sum(conv_to<vec>::from(g1 % sk));

      //      Rprintf("DEBUG: dfaj %6.4f sigma %6.4f df00 %6.4f\n", dfaj, sigma, df00);
      //      df.print("DEBUG df=");
      //      sk.print("DEBUG sk=");

      if (abs(dfaj) <= -sigma*df00) {
	x1 = par;
	f1 = faj;
	terminateLoop = 2; // Terminate Line search
      }
      else if (dfaj >= 0) {
	b1 = aj;
	fb1 = faj;
	dfb1 = dfaj;
	b2 = ai;
	fb2 = fai;
	dfb2 = dfai;

	terminateLoop = 1; // Terminate BRACKET
      }
      else {
	if (myy < 2*aj-ai) {
	  ai = aj;
	  fai = faj;
	  dfai = dfaj;

	  aj = myy;
	}
	else {
	  double ah = aj;

	  double min_value;
	  if (myy < aj)
	    min_value = myy + tau1*(aj-ai);
	  else
	    min_value = aj + tau1*(aj-ai);
	  //	  double min_value = apply(rbind(myy,aj)+tau1*(aj-ai), 2, min);
	  
	  aj = cinterpo(2*aj-ai,
			min_value, ai, aj, fai, faj, dfai, dfaj);
	  ai = ah;
	  fai = faj;
	  dfai = dfaj;
	}
        
	par = x0+aj*sk;

	lik = func(par, 1);
	faj = lik.f;
	df = lik.df;
	dfaj = sum(conv_to<vec>::from(df % sk));
      }
    }
  }
  //  if (verbose) {
  //    cout << "Number of brackets: " << nBracket << endl;
  //  }
  if (terminateLoop == 2) {
    if (print_level >= 2) {
      Rprintf("No sectioning\n");
    }
    return OK;
  }


  if (print_level >= 2) {
    Rprintf("Sectioning:\n");
  }

  int nSection = 0;
  terminateLoop = 0; // 1: terminate SECTION and LineSearch
  while (!terminateLoop) {
    nSection = nSection + 1;
    
    if (nSection > 30) {
      Rprintf("Error(BFGS::lineSearch): Too many sections\n");
      Rprintf("Error(BFGS::lineSearch): b1 %7.4f b2 %7.4f fb1 %7.4f fb2 %7.4f dfb1 %7.4f dfb2 %7.4f\n",
	      b1, b2, fb1, fb2, dfb1, dfb2);

      return TOO_MANY_SECTIONS;
    }
    
    double bi;
    int diffFromZero = !equal(dfb2, 0, 1e-6);
    if (diffFromZero) { //(all(dfb2 != 0)) {
      bi = cinterpo(b1+tau2*(b2-b1),b2-tau3*(b2-b1),b1,b2,fb1,fb2,dfb1,dfb2);
      //      Rprintf("CINTERPO: bi %6.4f, b1 %6.4f, b2 %6.4f, fb1 %6.4f, fb2 %6.4f, dfb1 %6.4f, dfb2 %6.4f\n",
      //	      bi, b1, b2, fb1, fb2, dfb1, dfb2);
      //      Rprintf("CINTERPO: cinterpo first two args %6.4f %6.4f \n", b1+tau2*(b2-b1),b2-tau3*(b2-b1));
    }
    else {
      bi = qinterpo(b1+tau2*(b2-b1),b2-tau3*(b2-b1),b1,b2,fb1,fb2,dfb1);
      //      Rprintf("QINTERPO: bi %6.4f, b1 %6.4f, b2 %6.4f, fb1 %6.4f, fb2 %6.4f, dfb1 %6.4f\n",
      //	      bi, b1, b2, fb1, fb2, dfb1);
      //      Rprintf("QINTERPO: qinterpo first two args %6.4f %6.4f \n", b1+tau2*(b2-b1),b2-tau3*(b2-b1));
    }

    par = x0+bi*sk;
    
    lik = func(par, 1);
    double fbi = lik.f;
    df = lik.df;

    double dfbi = sum(conv_to<vec>::from(df % sk));

    if (print_level >= 2) {
      Rprintf("Section iter %3d b1 %7.4f b2 %7.4f bi %7.4f fb1 %7.4f fb2 %7.4f dfb1 %7.4f dfb2 %7.4f\n",
	      nSection, b1, b2, bi, fb1, fb2, dfb1, dfb2);
      //      cout << "Section iter " << nSection << " b1 " << b1 << " b2 " << b2 << " bi " << bi << " fb1 " << fb1 << " fb2 " << fb2 << " dfb1 " << dfb1 << " dfb2 " << dfb2 << endl;
    }
    double prod = (b1-bi)*dfb1; // = Expected reduction in function value.
    //    double prod2 = fabs(b1-bi);
    if (prod < noImprovementValue) {// || prod2 < 1e-6) {
      if (print_level >= 2) {
	Rprintf("No improvement in line search. ");
	Rprintf("Debug information: b1=%7.4f bi=%7.4f dfb1=%7.4f (b1-bi)*dfb1=%7.4f\n",
		b1, bi, dfb1, prod);
      }
      g1 = df;
      x1 = par;
      f1 = fbi;
      //      g1 = df0;
      //      x1 = x0;
      //      f1 = f0;
      terminateLoop = 2;
    }
    else {
      if (fbi>f0 + rho*bi*df00 || fbi >= fb1) {
	b2 = bi;
	fb2 = fbi;
	dfb2 = dfbi;
      }
      else {
	g1 = df;
	dfbi = sum(conv_to<vec>::from(g1 % sk));

	if (abs(dfbi) < -sigma*df00) {
	  x1 = par;
	  f1 = fbi;
	  terminateLoop = 1; // TERMINATE SECTION
	}
	else if ((b2-b1)*dfbi>=0) {
	  // clear g1
	  b2 = b1;
	  fb2 = fb1;
	  dfb2 = dfb1;
	  b1 = bi;
	  fb1 = fbi;
	  dfb1 = dfbi;
	}
	else {
	  b1 = bi;
	  fb1 = fbi;
	  dfb1 = dfbi;
	}
      }
    }
  }

  //  if (print_level >= 2) {
  //    cout << "Number of sections: " << nSection << endl;
  //  }

  int status;
  if (terminateLoop == 2) {
    status = NO_IMPROVEMENT;
  }
  else {
    status = OK;
  }

  return status;
}

// Return 0 if OK
int BFGS::bfgs(FunctionValue (*func)(const vec &, const int), vec & par,
	       int print_level, double gradtol, mat & H, int & iter) {
  //  double f_min = -100000;        //f_min=nedre spesifisert grense funksjonsverdi
  double f_min = -10000000;        //f_min=nedre spesifisert grense funksjonsverdi
  double term = gradtol;        //term=konvergenskriterium
  
  int nMaxIter = 10000;

  int npar = par.n_elem;
  mat Hi = zeros(npar, npar);
  Hi.diag() = ones<vec>(npar);


  FunctionValue lik = func(par, 1);
  double f = lik.f;
  vec df = lik.df;

  vec df_abs = abs(df);
  double df_max = max(df_abs);
  int converged = (df_max < term);

  if (converged) {
    if (print_level >= 1) {
      Rprintf("Immediate convergence!\n");
    }
    return EXIT_SUCCESS;
  }

  vec x0 = par; //x0=variabel-vektor ved start av linjesoej
  vec g0 = df;  //g0=gradientvektor ved start av linjesoek

  vec sk = -g0/sqrt(accu(g0 % g0));  //sk= retningsvektor. Foerste retning=steepest descent

  //f0=funksjv. ved start av linjesoek
  double f0 = f;

  //df0= retn. deriverte ved start av linjesoek
  double df0 = sum(conv_to<vec>::from(sk % g0)); 
  
  double ai = 0;      // ai=foregaaende steglengde
  double aj = .01;    // aj=initial steglengde i bracketing-delen
  double fai = f0;    // fai=funksj.verdi i x0+ai*sk
  
  double dfai = df0;  //dfai=retningsderiverte i x0+ai*sk

  double myy = (f_min-f0)/(rho*df0);       //myy=max.steglengde
  if (print_level >= 2) {
    Rprintf("myy=%6.4f\n", myy);
  }

  par = x0+aj*sk;             //par = loepende variabel- vektor
  lik = func(par, 1);
  double faj = lik.f;
  df = lik.df;

  double dfaj = sum(conv_to<vec>::from(df % sk));
  
  iter = 1;
  vec g1;
  vec x1;
  double f1;

  while (1) {
    //    const int verbose = (print_level > 1);
    int status = lineSearch(func, ai, aj, fai, faj, dfai, dfaj, par, sk, myy, x0, f0, df, //g0,
			    print_level, g1, x1, f1);

    if (status == TOO_MANY_SECTIONS || status == TOO_MANY_BRACKETS) {
      Rprintf("\nTerminate optimisation\n");
      return EXIT_FAILURE;
    }
     

    // Check convergence and status from lineSearch
    int conv = (max(conv_to<vec>::from(abs(g1))) < term);
    if (conv || (iter>nMaxIter) || status == NO_IMPROVEMENT) {
      par = x1;
      H = inv(Hi);             //H=estimert Hessematrise
      
      if (print_level >= 1) {
	if (status == NO_IMPROVEMENT) {
	  Rprintf("\nNo improvement in line search. Terminate optimisation\n");
	}
	else if (conv) {
	  Rprintf("\nConvergence achieved.\n");
	}
	else if (iter > nMaxIter) {
	  Rprintf("\nMaximum number of iterations achieved.\n");
	}
	Rprintf(" Number of iterations: %d\n", iter);
	Rprintf("Final function value: %7.4f\n", f1);
	Rprintf("Final parameter: ");
	for (int i=0;i<npar;i++) {
	  Rprintf("%7.4f ", x1(i));
	}
	Rprintf("\n");

	Rprintf("Final gradient: ");
	for (int i=0;i<npar;i++) {
	  Rprintf("%7.4f ", g1(i));
	}
	Rprintf("\n");
      }

      return EXIT_SUCCESS;
    }
      
    // If optimisation is not terminated, new direction is choosed from function "retning"
    retning(x1-x0, g1-g0, g1, Hi, sk);


    g0 = g1;

    df0 = sum(conv_to<vec>::from(sk % g0));
    if (print_level >= 1) {
      Rprintf("Iteration: %4d  Function value: %7.4f\n", iter, f1);
      Rprintf("Parameter: ");
      for (int i=0;i<npar;i++) {
	Rprintf("%7.4f ", x1(i));
      }
      Rprintf("\n");

      Rprintf("Gradient:  ");
      for (int i=0;i<npar;i++) {
	Rprintf("%7.4f ", g1(i));
      }
      Rprintf("\n");
    }

    if (df0>0) {
      warning_own("Retningsder. positiv");
      Rprintf("df0= %7.4f\n", df0);
      mat H = inv(Hi);    //H = estimated Hessian

      par = x1;

      return EXIT_FAILURE;
    }

    x0 = x1;  //x0=variabelvektor ved start av nytt linjesoek

    aj = -2*(f0-f1)/df0; //aj= steglengde
    if (aj < .0000001)
      aj = 0.0000001;
    else if (aj > 1.0) // Changed 23.10.2009 0.01 --> 1.0
      aj = 1.0;
    //    aj = min(.01, max(-2*(f0-f1)/df0, .0000001));    //aj= steglengde

    f0 = f1;    //f0 funk.verdi ved start av nytt linjesoek
    ai = 0;
    fai = f0;
    dfai = df0;
    myy = (f_min-f0)/(rho*df0);
    par = x0+aj*sk;

    
    lik = func(par, 1);
    faj = lik.f;
    df = lik.df;
    dfaj = sum(conv_to<vec>::from(df % sk));

    iter++;
  }

  return EXIT_SUCCESS;
}
