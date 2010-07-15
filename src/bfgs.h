#ifndef BFGS__
#define BFGS__ 1

class BFGS : public Optimise {
 public:
  BFGS();
  BFGS(const double value);
  ~BFGS() {};

  int bfgs(FunctionValue (*func)(const vec &, const int), vec & par,
	   int print_level, double gradtol, mat & H, int & iter);

 private:
  double cinterpo(double a, double b, double b1, double b2,
		  double fb1, double fb2, double dfb1, double dfb2);
  double qinterpo(double a, double b, double b1, double b2,
		  double fb1, double fb2, double dfb1);
  void retning(const vec & delta, const vec & gama, const vec & g, mat & Hi, vec & sk);
  mat xpd(vec c, int d, int p);
  int lineSearch(FunctionValue (*func)(const vec &, const int), double ai, double aj,
		 double fai, double faj, double dfai, double dfaj,
		 const vec & par0, const vec & sk, double myy, const vec & x0, double f0,
		 const vec & df, int verbose, vec & g1, vec & x1, double & f1);
 //linjesoekparametre
  const double rho;
  const double sigma;
  const double tau1;
  const double tau2;
  const double tau3;
  double noImprovementValue;
};

#endif
