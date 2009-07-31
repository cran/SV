#ifndef PARAMETERS__
#define PARAMETERS__ 1

#ifndef NOTRANSF
#define NOTRANSF -1
#endif

class Parameters : public MathOp {
 public:
  Parameters(int nSup);

  Parameters(const vec & x, const int transf);
  void setPars(const vec & x, const int transf);
  double getPar(const int ind);

  static int numberOfSuperPositions(const vec & par);
  vec asvector();

  int checkPars(const int transf);

  //  Parameters extractPars(const vec & par);
  vec extractParsInv(const int transf); //const Parameters & parObj);

  Parameters(const double mu_, const double psi_,
	     const vec & omega_, const vec & lambda_, const int transf_);

  void print() const {
    Rprintf("mu: %8.5f\n", mu);
    Rprintf("psi: %8.5f ", psi);
    Rprintf("lambda: ");
    int n = lambda.n_elem;
    for (int i=0;i<n;i++)
      Rprintf("%8.5f ", lambda(i));
    Rprintf("omega: ");
    for (int i=0;i<n;i++)
      Rprintf("%8.5f ", omega(i));
    Rprintf("\n");
  };

  double psi;
  double mu;

  vec omega;
  vec lambda;

  static double minlambda;
  static double maxlambda;

 private:
  vec extractParsInv1();
  vec extractParsInv2();
  void setPars0(const vec & x);
  void setPars1(const vec & x);
  void setPars2(const vec & x);
};

#endif
