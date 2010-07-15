/*Include Files:*/
//#include <iostream>
#include <math.h>

#include <R.h>
#include <Rmath.h>

#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

//#include "basic.h"
#include "math.h"
#include "parameters.h"
#include "simulate.h"

// Univariate
Simulate::Simulate(const int nSup_, const int nTimes_, const int print_level_, const int saveDraws_) : p(0), q(1), nSup(nSup_), nTimes(nTimes_), saveDraws(saveDraws_) {
  GetRNGstate(); // Read in .Random.seed

  multivariate = 0;
  print_level = print_level_;
}

// Multivariate
Simulate::Simulate(const int p_, const int q_,
		   const int nSup_, const int nTimes_, const int print_level_, const int saveDraws_) : p(p_), q(q_), nSup(nSup_), nTimes(nTimes_), saveDraws(saveDraws_) {
  GetRNGstate(); // Read in .Random.seed

  multivariate = 1;
  print_level = print_level_;
}

Simulate::~Simulate() {
  PutRNGstate(); // Write to .Random.seed
}

vec Simulate::normal(const int n) {
  vec u(n);
  for (int i=0;i<n;i++) {
    u(i) = rnorm(0, 1);
  }

  return u;
}

void Simulate::simulateInit() {
  //  const int resetSeed=0;
  // Observation noise
  //  if (resetSeed)
  //    GetRNGstate();// Read in .Random.seed

  if (!multivariate) {
    epsilon = normal(nTimes);
    if (print_level >= 2) {
      Rprintf("Check of draws for epsilon\n");
      Rprintf("Draw - mean(epsilon):%8.6f var(epsilon):%8.6f\n", mean(epsilon), var(epsilon));
      Rprintf("True - mean(epsilon):%8.6f var(epsilon):%8.6f\n", 0.0, 1.0);
    }
    //    saveEpsilonToFile();
  }
  else {
    epsilonMat = zeros<mat>(q, nTimes);
    for (int k=0;k<q;k++) {
      epsilonMat.row(k) = trans(normal(nTimes));
    }
    if (print_level >= 2) {
      Rprintf("Check of draws for epsilon\n");
      for (int k=0;k<q;k++) {
	Rprintf("Draw %d - mean(epsilon):%8.6f var(epsilon):%8.6f\n\n",
		k, mean(epsilonMat.row(k)), var(epsilonMat.row(k)));
      }
      Rprintf("True - mean(epsilon):%8.6f var(epsilon):%8.6f\n", 0.0, 1.0);
    }
  }

  nFirstDraw=2;
  nDrawLen=2;

  const int nSup_pq = nSup*(p+q);
  rfirst = new mat[nSup_pq];
  afirst = new mat[nSup_pq];
  draws(rfirst, afirst, nFirstDraw, nDrawLen);


  // Random vectors used in simulating stochastic volatility
  
  //  if (!resetSeed)
  PutRNGstate(); // Write to .Random.seed
}

void Simulate::cleanup() {
  delete [] rfirst;
  delete [] afirst;
}

void Simulate::draws(mat * rf, mat * af,
		     const int nDraw, const int nLen) {

  const int nSup_pq = nSup*(p+q);
  for (int isup=0;isup<nSup_pq;isup++) {
    rf[isup] = zeros<mat>(nLen, nTimes);
    for (int i=0;i<nTimes;i++) {
      for (int j=0;j<nDraw;j++) {
	rf[isup](j,i) = unif_rand();
      }
    }
  }

  for (int isup=0;isup<nSup_pq;isup++) {
    af[isup] = zeros<mat>(nLen, nTimes);
    for (int i=0;i<nTimes;i++) {
      af[isup](0,i) = rexp(1);
      for (int j=1;j<nDraw;j++) {
	af[isup](j,i) = af[isup](j-1,i) + rexp(1);
      }
    }
  }
  if (print_level >= 2) {
    checkDraws(rf, af, nDraw);
  }
}

void Simulate::checkDraws(mat * rf, mat * af, const int nDraw) {
  const double n = nSup * nTimes * nDraw;
  double sumr = 0.0;
  double sumr2 = 0.0;
  double sumu = 0.0;
  double sumu2 = 0.0;
  const int nSup_pq = nSup*(p+q);
  for (int isup=0;isup<nSup_pq;isup++) {
    sumr += accu(rf[isup]);
    sumr2 += accu(rf[isup] % rf[isup]);
    rowvec u = af[isup].row(0);
    sumu += accu(u);
    sumu2 += accu(u % u);
    for (int j=1;j<nDraw;j++) {
      u = af[isup].row(j) - af[isup].row(j-1);
      sumu += accu(u);
      sumu2 += accu(u % u);
    }
  }
  
  Rprintf("Check of draws for r\n");
  Rprintf("Draw - mean(r):%8.6f mean(r^2):%8.6f\n", sumr/n, sumr2/n);
  Rprintf("True - mean(r):%8.6f mean(r^2):%8.6f\n", 0.5, 1.0/3.0);
  Rprintf("Check of draws for u\n");
  Rprintf("Draw - mean(u):%8.6f mean(u^2):%8.6f\n", sumu/n, sumu2/n);
  Rprintf("True - mean(u):%8.6f mean(u^2):%8.6f\n", 1.0, 2.0);
}

void Simulate::newDraws() {
  const int nNewDraws = 1;
  if (print_level >= 2) {
    Rprintf("Draw %d new draws\n", nNewDraws);
  }

  const int nSup_pq = nSup*(p+q);

  mat * rnew = new mat[nSup_pq]; 
  mat * anew = new mat[nSup_pq];

  draws(rnew, anew, nNewDraws, nNewDraws);

  if (nFirstDraw + nNewDraws <= nDrawLen) {
    for (int isup=0;isup<nSup_pq;isup++) {
      rfirst[isup].submat(nFirstDraw, 0, nFirstDraw+nNewDraws-1, nTimes-1) = rnew[isup];
      for (int j=nFirstDraw;j<nFirstDraw+nNewDraws;j++) {
	afirst[isup].row(j) = afirst[isup].row(nFirstDraw-1) + anew[isup].row(j-nFirstDraw);
      }
    }
  }
  else {
    nDrawLen += 10 * nNewDraws;
    
    if (print_level >= 2) {
      Rprintf("Increase memory for afirst and rfirst by %d\n", 10*nNewDraws);
    }
    //  rfirst = new mat[nSup];
    for (int isup=0;isup<nSup_pq;isup++) {
      mat rold = rfirst[isup];
      mat aold = afirst[isup];
      rfirst[isup].set_size(nDrawLen, nTimes);
      afirst[isup].set_size(nDrawLen, nTimes);

      rfirst[isup].submat(0,0, nFirstDraw-1, nTimes-1) = rold;
      afirst[isup].submat(0,0, nFirstDraw-1, nTimes-1) = aold;

      rfirst[isup].submat(nFirstDraw, 0, nFirstDraw+nNewDraws-1, nTimes-1) = rnew[isup];

      for (int j=nFirstDraw;j<nFirstDraw+nNewDraws;j++) {
	afirst[isup].row(j) = afirst[isup].row(nFirstDraw-1) + anew[isup].row(j-nFirstDraw);
      }
    }

  }

  nFirstDraw += nNewDraws;
  if (print_level >= 2) {
    Rprintf("Updated number of draws: %d\n", nFirstDraw);
  }
  if (print_level >= 3) {
    checkDraws(rfirst, afirst, nFirstDraw);
  }
  //  saveDrawsToFile();

  delete [] rnew;
  delete [] anew;
}

void Simulate::saveEpsilonToFile() {
  ofstream ost("draws_epsilon_save.txt");
  ost.precision(10);
  
  ost << nTimes << endl;

  for (int i=0;i<nTimes;i++) {
    ost << epsilon(i) << endl;
  }
  ost.close();
}

void Simulate::saveDrawsToFile() {
  const int nSup_pq = nSup*(p+q);

  ofstream ost("draws_r_save.txt");
  ost.precision(10);
  
  ost << nSup << endl;
  ost << nTimes << endl;
  ost << nFirstDraw << endl;
  for (int isup=0;isup<nSup_pq;isup++) {
    for (int i=0;i<nTimes;i++) {
      for (int j=0;j<nFirstDraw;j++) {
	ost << rfirst[isup](j,i) << " ";
      }
      ost << endl;
    }
  }
  ost.close();

  ost.open("draws_a_save.txt");
  ost << nSup << endl;
  ost << nTimes << endl;
  ost << nFirstDraw << endl;
  for (int isup=0;isup<nSup_pq;isup++) {
    for (int i=0;i<nTimes;i++) {
      for (int j=0;j<nFirstDraw;j++) {
	ost << afirst[isup](j,i) << " ";
      }
      ost << endl;
    }
  }
  ost.close();
}

vec Simulate::simulate(double mu, const vec & lambda, const double psisum,
		       const vec & omega2, const int nObs, const double deltaT,
		       const int resetSeed, vec & s2) {
  const vec null;

  if (nObs > nTimes) {
    Rprintf("Error(Simulate::simulate): nObs > nTimes\n");
    exit(-1);
  }
  //  lambda.print("lambda=");
  //  omega2.print("omega2=");
  if (resetSeed) {
    GetRNGstate();
    
    if (print_level >= 3) {
      double test = rnorm(0,1);
      cout << "test rnorm " << test << endl;
    }
  }

  s2 = sigma2super(lambda, psisum, omega2, deltaT, nObs);
  if (s2.n_elem == 0) {
    return null;
  }

  vec y = mu * deltaT + sqrt(s2) % epsilon.rows(0, nObs-1);

  if (print_level >= 2) {
    Rprintf("simulate: Mean s2 = %6.4f  var s2 = %6.4f\n", mean(s2), var(s2));
    Rprintf("simulate: Mean y  = %6.4f  var y  = %6.4f\n", mean(y), var(y));
  }

  //  if (!resetSeed) {
  //    PutRNGstate();
  //  }

  //  Rprintf("Quit Simulate::simulate\n");
 
  return y;
}


mat Simulate::simulateMulti(const vec & mu, const mat & lambda, const vec & psisum,
			    const mat & omega2, const double phi21,
			    const int nObs, const double deltaT,
			    const int resetSeed, mat & s2) {
  const vec null;

  //  lambda.print("lambda=");
  //  omega2.print("omega2=");
  if (resetSeed) {
    GetRNGstate();
    
    if (print_level >= 3) {
      double test = rnorm(0,1);
      cout << "test rnorm " << test << endl;
    }
  }

  s2 = zeros<mat>(p+q, nObs);
  for (int k=0;k<p+q;k++) {
    const int indsup = k*nSup;
    const vec s2_uni = sigma2super(trans(lambda.row(k)), psisum(k), trans(omega2.row(k)), deltaT, nObs, indsup);
    if (s2.n_elem == 0) {
      return null;
    }
    s2.row(k) = trans(s2_uni);
  }


  mat y(q, nObs);

  const rowvec tmp = phi21 * s2.row(2);
  y.row(0) = mu(0) * deltaT + sqrt(s2.row(0) + s2.row(2)) % epsilonMat.row(0);
  //  y.row(1) = mu(1) * deltaT + sqrt(s2.row(1) + phi21*s2.row(2)) % epsilonMat.row(1);
  y.row(1) = mu(1) * deltaT + sqrt(s2.row(1) + tmp) % epsilonMat.row(1);

  if (print_level >= 2) {
    for (int k=0;k<p+q;k++) {
      Rprintf("simulateMulti: Latent s2 process %d: Mean s2 = %6.4f  var s2 = %6.4f\n",
	      k, mean(s2.row(k)), var(s2.row(k)));
    }
    for (int k=0;k<q;k++) {
      Rprintf("simulateMulti: Data vector %d: Mean y  = %6.4f  var y  = %6.4f\n",
	      k, mean(y.row(k)), var(y.row(k)));
    }
  }

  if (!resetSeed) {
    PutRNGstate();
  }

  //  Rprintf("Quit Simulate::simulateMulti\n");
 
  return y;
}

int Simulate::validParsForSimulation(const vec & parvec,
				     const int transf) {
  Parameters par(parvec, transf);
  vec nu = par.psi*par.psi/par.omega;
  const double deltaT = 1.0;
  return validParsForSimulation(par.lambda, nu, deltaT);
}

int Simulate::validParsForSimulation(const vec & lambda, const vec & nu,
				     const double deltaT) {
  int valid = 1;
  vec lambdaNuDeltaT = (lambda % nu) * deltaT;
  double maxLambdaNuDeltaT = max(lambdaNuDeltaT);

  const double limit = 100;
  if (maxLambdaNuDeltaT > limit) {
    Rprintf("max(lambda * nu * deltaT)= %10.2f > %6.2f\n", maxLambdaNuDeltaT, limit);
    valid = 0;
  }
  return valid;
}


vec Simulate::sigma2super(const vec & lambda, const double psisum, const vec & omega2,
			  const double deltaT, const int nObs, const int indsup) {
  const double psi = psisum/nSup;
  vec alpha = psi/omega2;
  vec nu = psi*psi/omega2;
  const vec null;

  if (!validParsForSimulation(lambda, nu, deltaT)) {
    return null;
  }

  vec sig2super = zeros<vec>(nObs);

  for (int i=0;i<nSup;i++) {
    vec sig2 = sigma2Volatility(alpha[i], nu[i], lambda[i], deltaT, indsup + i, nObs);
    if (sig2.n_elem == 0) {
      return null;
    }
    sig2super = sig2super + sig2;
  }

  return sig2super;
}

vec Simulate::sigma2Volatility(const double alpha, const double nu,
			       const double lambda, const double deltaT, const int isup, const int nObs) {
  // Using formula page 517 Griffin&Steel (see also page 177 Barndorff-Nielsen&Shephard);
  vec sigma2(nObs);

  // Set initial value = expected value of sigma2 to avoid drawing from a gamma distribution
  // which is an accept/reject algorithm and therefore may have different number of draws
  // from the random number generator. This could alter the subsequent random draws
  // and violate the continuity property of the simulations.
  sigma2(0) = nu/alpha;

  double lambdaNuDeltaT = lambda*nu*deltaT;

  for (int i=1;i<nObs;i++) {
    int index = 0; // Number of draws

    vec a = afirst[isup].col(i);
    vec r = rfirst[isup].col(i);

    while(a(index) < lambdaNuDeltaT) {
      index++;
      if (index >= nFirstDraw) {
	newDraws();
	a = afirst[isup].col(i);
	r = rfirst[isup].col(i);
      }
    }

    int ind = index - 1; // Number of draws fullfilling a < lambda * nu * deltaT

    double eta = 0;
    if (ind >= 0) {
      vec asub = a.rows(0,ind);
      vec rsub = r.rows(0,ind);
      eta = (1/alpha)* exp(-lambda*deltaT) *
	sum(conv_to<vec>::from(log(lambdaNuDeltaT/asub) % exp(lambda*rsub)));
    }

    sigma2(i) = exp(-lambda*deltaT) * sigma2(i-1) + eta;
    if (isnan(sigma2(i))) { // Check if nan
      Rprintf("Error(Simulate::sigma2Volatility): sigma2(i) is nan. lambda %10.8f eta %10.8f nu %10.8f alpha %10.8f\n",
	      lambda, eta, nu, alpha);
      const vec null;
      return null;
    }
  }

  return sigma2;
}
