#ifndef QL__
#define QL__ 1

struct EstimationObject {
  //QL
  int nIter;

  // Common variables: QL, Indirect Inference
  int status;
  vec par;
  vec parStart;
  mat H;
  double gradtol;

 // Indirect inference only
  int nSimAll;
  mat parsim;
  ivec iter;
  ivec iterAll;
  vec funcval;
  vec funcvalAll; // also those not converging.
  
  EstimationObject() {
    status = EXIT_FAILURE;
  };

  // QL
  EstimationObject(const vec & par_, const vec & parStart_,
		   const mat & H_, const double gradtol_, const int status_, const int nIter_) {
    par = par_;
    parStart = parStart_;
    H = H_;
    gradtol = gradtol_;

    status = status_;
    nIter = nIter_;
  };

  //Indirect inference
  EstimationObject(const vec & par_, const vec & parStart_,
		   const mat & parsim_,
		   const mat & H_, const double gradtol_,
		   const vec & funcval_,
		   const vec & funcvalAll_,
		   const ivec & iter_,
		   const ivec & iterAll_,		   
		   const int nSimAll_, const int status_) {
    par = par_;
    parStart = parStart_;
    H = H_;
    gradtol = gradtol_;

    status = status_;

    nSimAll = nSimAll_;
    funcval = funcval_;
    funcvalAll = funcvalAll_;
    iter = iter_;
    iterAll = iterAll_;

    parsim = parsim_;
  };

  void print(int transf) {
    parStart.print_trans("Start parameter values (unrestricted):");
    Parameters parObj(parStart,transf);
    parObj.print("Start parameters (restricted)");
    par.print_trans("Estimated parameter values (unrestricted)");
    parObj.setPars(par, transf);
    parObj.print("Estimated parameter values (restricted)");
  };

  void printMulti(int transf) {
    parStart.print_trans("Start parameter values (unrestricted):");
    ParametersMulti parObj(parStart,transf);
    parObj.print("Start parameters (restricted)");
    par.print("Estimated parameter values (unrestricted)");
    parObj.setPars(par, transf);
    parObj.print("Estimated parameter values (restricted)");
  };
};

class QL : public MathOp, public Simulate {
 public:
  //  QL(const double minlambda_, const double maxlambda_, int useRoptimiser_);
  QL(const vec & y, const double minlambda_, const double maxlambda_,
     const int transf_, const int addPenalty_, int useRoptimiser,
     const double gradMax_);
  QL(const vec & y, const double minlambda_, const double maxlambda_,
     const int transf_, const int addPenalty_, int useRoptimiser_,
     const double gradMax_, const int nSup_, const int nTimes_, const int print_level_, const int saveDraws_);
  QL(const mat & y, const vec minlambda_, const vec maxlambda_,
     const int transf_, const int addPenalty_, int useRoptimiser_,
     const double gradMax_);
  QL(const mat & y, const vec minlambda_, const vec maxlambda_,
     const int transf_, const int addPenalty_, int useRoptimiser_,
     const double gradMax_, const int p_, const int q_,
     const int nSup_, const int nTimes_, const int print_level_, const int saveDraws_);
  ~QL();

  void setData(const vec & y);
  void setDataMulti(const mat & y);
  void setDataLogReturns(const vec & yret);

  EstimationObject optimise(const vec & startpar,
			    const double gradtol,
			    const int print_level,
			    const ivec & updatePars_,
			    FunctionValue (*func)(const vec &, const int));
  FunctionValue quasiLikelihood(const vec & par, const int evaluateGradient);
  mat quasiLikelihood_individual(const vec & parReduced);
  FunctionValue quasiLikelihoodMulti(const vec & par, const int evaluateGradient);
  mat quasiLikelihoodMulti_individual(const vec & par);
  void checkGradient(FunctionValue (*func)(const vec &, const int),
		     const vec & par, double eps, double tol, int verbose=0) {
    opt.checkGradient(func, par, eps, tol, verbose);
  };
  vec gradient(const vec & par,
	       const mat & b, const mat & W, const mat & BBt,
	       const mat & Qu, const mat & sigma, const mat & fii);
  vec gradient_multivariat(const vec & par,
			   const mat & b, const mat & W, const mat & BBt,
			   const mat & Qu, const mat & sigma, const mat & fii,
			   const mat & Gi, const mat & F1,
			   const vec & tau, const mat & gama);
  void setUpdates(const ivec updatePars_, const vec par_);
  vec setReducedPar(vec par0);
  vec setFullPar(vec parReduced);
  void confidenceIntervals(const vec & estimate, const mat & H,
			   mat & Hi,
			   Parameters & sd,
			   Parameters & lower, Parameters & upper,
			   Parameters & lowerUn, Parameters & upperUn, 
			   const int sandwich,
			   const int deltaMethod=0);
  void confidenceIntervalsMulti(const vec & estimate, const mat & H,
				mat & Hi,
				ParametersMulti & sd,
				ParametersMulti & lower, ParametersMulti & upper,
				ParametersMulti & lowerUn, ParametersMulti & upperUn, 
				const int sandwich);
  void setNObs(const int nObs_) {
    nObs = nObs_;
  };

  //  Parameters extractPars(const vec & par);
  //  vec extractParsInv(const Parameters & parObj);

 protected:
  BFGS opt;
  int nObs;
  mat Z0; // contains data
  int transf;
  int addPenalty;
  double penaltyMin;
  double penaltyMax;
  const double gradMax;

  int useRoptimiser;

  ivec updatePars;
  vec parFull;

  void filter(const vec & par, const mat & Z0, mat & a, mat & a_, mat & V, mat & V_, mat & Qu, mat & sigma, mat & fii, double & f);
  void filter_multivariat(const vec & par, const mat & Z0, mat & a, mat & a_, mat & V, mat & V_, mat & Qu, mat & sigma, mat & fii, vec & tau, mat & gama, mat & Gi, mat & F1, double & f);
  void smoother(const vec & par,  const mat & A, const mat & A_, const mat & V, const mat & V_,
		mat & b, mat & W, mat & BBt);
  void smoother_multivariate(const vec & par,  const mat & A, const mat & A_, const mat & V, const mat & V_, const mat & fii,
			     mat & b, mat & W, mat & BBt);
  void confidenceIntervals(const vec & estimate, const mat & Hi,
			   Parameters & sd,
			   Parameters & lower, Parameters & upper,
			   Parameters & lowerUn, Parameters & upperUn, 
			   const int deltaMethod=0);
  void confidenceIntervalsMulti(const vec & estimate, const mat & Hi,
				ParametersMulti & sd,
				ParametersMulti & lower, ParametersMulti & upper,
				ParametersMulti & lowerUn, ParametersMulti & upperUn);
  vec gradientIndividual(const vec & par,
			 const mat & b, const mat & W, const mat & BBt,
			 const mat & Qu, const mat & sigma, const mat & fii,
			 mat & gr);
  vec gradient_multivariat_individual(const vec & par,
				      const mat & b, const mat & W, const mat & BBt,
				      const mat & Qu, const mat & sigma, const mat & fii,
				      const mat & Gi, const mat & F1,
				      const vec & tau, const mat & gama,
				      mat & gr);
  void constrainGradient(mat & gr);
  mat computeSandwichMatrix(const mat & H, mat & gr, const int nObs);
};

FunctionValue func(const vec &, const int);
FunctionValue funcMulti(const vec &, const int);
extern QL * qlExtern;


#endif
