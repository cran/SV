#ifndef Simulate__
#define Simulate__ 1

class Simulate {
 public:
  Simulate(const int nSup_, const int nTimes_, const int print_level_);
  Simulate(const int p_, const int q_,
	   const int nSup_, const int nTimes_, const int print_level_);
  ~Simulate();

  void simulateInit();
  vec simulate(double mu, const vec & lambda, const double psisum, const vec & omega2,
	       const int nTimes, const double deltaT, const int resetSeed,
	       vec & s2);
  mat simulateMulti(const vec & mu, const mat & lambda, const vec & psisum, const mat & omega2, const double phi21,
		    const int nTimes, const double deltaT, const int resetSeed,
		    mat & s2);
  int validParsForSimulation(const vec & parvec,
			     const int transf);

  static vec normal(const int n);

 private:
  int print_level;
  int multivariate;

  vec epsilon; // univariate
  mat epsilonMat; // multivariate

  const int p;
  const int q;

  const int nSup;
  const int nTimes;

  int nFirstDraw;
  int nDrawLen;
  mat * rfirst;
  mat * afirst;

  int validParsForSimulation(const vec & lambda, const vec & nu,
			     const double deltaT);
  vec sigma2Volatility(const double alpha, const double nu,
		       const double lambda, const double deltaT, const int isup);
  vec sigma2super(const vec & lambda, const double psisum, const vec & omega2,
		  const double deltaT, const int indsup=0);

  void draws(mat * rf, mat * af,
	     const int nDraw, const int nLen);
  void newDraws();

  void checkDraws(mat * rf, mat * af, const int nDraw);
  void saveEpsilonToFile();
  void saveDrawsToFile();
};

#endif
