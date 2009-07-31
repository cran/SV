#ifndef INDIRECT__
#define INDIRECT__ 1

class Indirect : public Simulate, public QL {
 public:
  Indirect(const vec & y,
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
	   const int initialSteepestDescent_); // : QL(parObs_, minlambda_, maxlambda_), Simulate();
  ~Indirect();

  EstimationObject indirectInference(const vec & startpar);
  void test(vec & par);

  void checkContinuity(const vec & startpar, const int nEval, const double delta,
		       int * indpar, mat & xvals, mat & xvals_transf, mat & funcvals);

  double distanceMetric(const vec & par, int & status);

  int dimPar() {return parObs.n_elem;};

  void resetToQLStartPar() {
    startpar_inner = parObs;
  };
  void resetToBestBracketStartPar() {
    startpar_inner = startpar_inner_min_mnbrak;
  }
  void resetToBestStartPar(vec & par, double & f) {
    startpar_inner = startpar_inner_min;
    par = est_inner_min;
    f = distMin;
  }
  void resetDistMinBracket() {
    distMinBracket = DBL_MAX;
  }
  void resetDistMin() {
    distMin = DBL_MAX;
  }

 private:
  int nTimes;
  int nSim;
  mat W;

  int print_level_inner;

  int useStartPar;
  vec parObs;

  double distMin;
  double distMinBracket;
  vec startpar_inner;
  vec startpar_inner_min_mnbrak;
  vec startpar_inner_min;
  vec est_inner_min;

  // Used in inner optimisation
  const double gradtol;

  // Used in outer optimisation
  const double ftol;
  const double ftol_weak;
  const int print_level;
  const int initialSteepestDescent;
};

extern Indirect * indirectExtern;



#endif
