#ifndef INDIRECT__
#define INDIRECT__ 1

class Indirect : public QL {
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
	   const double gradMax_,
	   const int initialSteepestDescent_,
	   const int ITMAX_,
	   const int saveDraws_); // : QL(parObs_, minlambda_, maxlambda_), Simulate();
  ~Indirect();

  EstimationObject indirectInference(const vec & startpar, const int useQLAsStartPar=1);
  void test(vec & par);

  void checkContinuity(const vec & startpar, const int nEval, const double delta,
		       int * indpar, mat & xvals, mat & xvals_transf, mat & funcvals);

  double distanceMetric(const vec & par, int & status);

  mat computeBgradient(const vec & par, const vec & sdQL, const vec & startpar_QL, vec & res0, mat & H);

  vec simulateData(const Parameters & pex);
  void simulateAndSetData(const Parameters & pex);

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
  vec bestQLEstimate() {
    return est_inner_min;
  }

  void QLestimates(vec & est, vec & sd) {
    est = parObs;
    sd = sdObs;
  }

  void confidenceIntervalsIndirect(const EstimationObject & obj,
				   Parameters & sd,
				   Parameters & lower, Parameters & upper,
				   Parameters & lowerUn, Parameters & upperUn, 
				   const int sandwichIndirect);

 private:
  int nTimes;
  int nSim;
  mat W;

  int print_level_inner;

  int useStartPar;
  vec parObs; // QL estimate
  vec sdObs; // SD of QL estimate

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
  const int ITMAX;

  void computeQLSimEstimate(const Parameters & pex, const vec & startpar, vec & est, mat & H);
  void confidenceIntervalsIndirect(const EstimationObject & obj,
				   Parameters & sd,
				   Parameters & lower, Parameters & upper,
				   Parameters & lowerUn, Parameters & upperUn);
  mat computeSandwichMatrixIndirect(const mat & H, mat & gr, const mat & bder0, const int nObs, const int nTimes);
};

extern Indirect * indirectExtern;



#endif
