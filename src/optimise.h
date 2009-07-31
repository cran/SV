#ifndef OPTIMISE__
#define OPTIMISE__ 1

class Optimise {
 public:
  Optimise();
  Optimise(const double value);
  ~Optimise() {};
  void checkGradient(FunctionValue (*func)(const vec &, const int),
		     const vec & par, double eps, double tol, int verbose=0);

  static int nFuncEval;
  static int nGradEval;
  static int nFuncEvalOuter;

 private:
};

#endif
