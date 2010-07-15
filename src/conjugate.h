#ifndef CONJUGATEDIRECTION__
#define CONJUGATEDIRECTION__ 1

class ConjugateDirection : public Optimise, public MathOp {
 public:
  ConjugateDirection(const int initialSteepestDescent_, const int print_level_);
  ~ConjugateDirection();

  void conjugateDirection(double (*func)(const vec &, int &),
			  vec & par,
			  double & fret,
			  double ftol, double ftol_weak, mat & H,
			  int & restart, int & iter, const int ITMAX, int & error);
  void setConjugateDirection(double (*func)(const vec &, int &), vec p, mat & xi, const int npar, const double h);

 private:
  const int print_level;
  const int initialSteepestDescent;

  const int maxRandomStep;

  void computeOrthogonalVectors(mat & u, const vec & pder);

  void powell(vec & p, mat & xi, int n, double ftol, double ftol_weak, int & iter, double & fret,
	      double (*func)(const vec &, int &), int & restart, const int ITMAX, int & error);
  void linmin(vec & p, const vec & xi, int n, double & fret,
	      double (*func)(const vec &, int &), double & bx_start, int & error);
  void mnbrak(double & ax, double & bx, double & cx, double & fa, double & fb, double & fc, int & status);
  double brent(double ax, double bx, double cx, double tol,
	       double & xmin, int & status);
  double f1dim(double x, int & status);
};

#endif
