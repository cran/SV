#ifndef QL__
#define QL__ 1

struct EstimationObject {
  // Common variables: QL, Indirect Inference
  int status;
  vec par;
  vec parStart;
  mat H;
  double gradtol;

 // Indirect inference only
  int nSimAll;
  mat parsim;

  EstimationObject() {
    status = EXIT_FAILURE;
  };

  // QL
  EstimationObject(const vec & par_, const vec & parStart_,
		   const mat & H_, const double gradtol_, const int status_) {
    par = par_;
    parStart = parStart_;
    H = H_;
    gradtol = gradtol_;

    status = status_;
  };

  //Indirect inference
  EstimationObject(const vec & par_, const vec & parStart_,
		   const mat & parsim_,
		   const mat & H_, const double gradtol_,
		   const int nSimAll_, const int status_) {
    par = par_;
    parStart = parStart_;
    H = H_;
    gradtol = gradtol_;

    status = status_;

    nSimAll = nSimAll_;
    parsim = parsim_;
  };

  void print(int transf) {
    cout << "Start parameter values:     " << endl << parStart;
    Parameters parObj(parStart,transf);
    cout << "    back transformed      : " << endl;
    parObj.print();
    cout << "Estimated parameter values: " << endl;
    par.print();
    parObj.setPars(par, transf);
    cout << "    back transformed      : " << endl;
    parObj.print();
  };

  void printMulti(int transf) {
    cout << "Start parameter values:     " << endl << trans(parStart);
    ParametersMulti parObj(parStart,transf);
    cout << "Back transformed start parameter values: " << endl;
    parObj.print();
    cout << endl << "Estimated parameter values: " << trans(par);
    //par.print();
    parObj.setPars(par, transf);
    cout << "Back transformed estimated parameter values:" << endl;
    parObj.print();
  };
};

class QL : public MathOp {
 public:
  QL(const double minlambda_, const double maxlambda_, int useRoptimiser_);
  QL(const vec & y, const double minlambda_, const double maxlambda_,
     const int transf_, const int addPenalty_, int useRoptimiser);
  QL(const mat & y, const vec minlambda_, const vec maxlambda_,
     const int transf_, const int addPenalty_, int useRoptimiser_);
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
  vec gradientIndividual(const vec & par,
			 const mat & b, const mat & W, const mat & BBt,
			 const mat & Qu, const mat & sigma, const mat & fii,
			 mat & gr);
  vec gradient_multivariat(const vec & par,
			   const mat & b, const mat & W, const mat & BBt,
			   const mat & Qu, const mat & sigma, const mat & fii,
			   const mat & Gi, const mat & F1,
			   const vec & tau, const mat & gama);
  vec gradient_multivariat_individual(const vec & par,
				      const mat & b, const mat & W, const mat & BBt,
				      const mat & Qu, const mat & sigma, const mat & fii,
				      const mat & Gi, const mat & F1,
				      const vec & tau, const mat & gama,
				      mat & gr);
  void setUpdates(const ivec updatePars_, const vec par_);
  vec setReducedPar(vec par0);
  vec setFullPar(vec parReduced);

  //  Parameters extractPars(const vec & par);
  //  vec extractParsInv(const Parameters & parObj);

 protected:
  BFGS opt;
  mat Z0; // contains data
  int transf;
  int addPenalty;
  double penaltyMin;
  double penaltyMax;

  int useRoptimiser;

  ivec updatePars;
  vec parFull;

  void filter(const vec & par, const mat & Z0, mat & a, mat & a_, mat & V, mat & V_, mat & Qu, mat & sigma, mat & fii, double & f);
  void filter_multivariat(const vec & par, const mat & Z0, mat & a, mat & a_, mat & V, mat & V_, mat & Qu, mat & sigma, mat & fii, vec & tau, mat & gama, mat & Gi, mat & F1, double & f);
  void smoother(const vec & par,  const mat & A, const mat & A_, const mat & V, const mat & V_,
		mat & b, mat & W, mat & BBt);
  void smoother_multivariate(const vec & par,  const mat & A, const mat & A_, const mat & V, const mat & V_, const mat & fii,
			     mat & b, mat & W, mat & BBt);
};

FunctionValue func(const vec &, const int);
FunctionValue funcMulti(const vec &, const int);
extern QL * qlExtern;


#endif
