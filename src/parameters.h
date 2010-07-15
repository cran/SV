#ifndef PARAMETERS__
#define PARAMETERS__ 1

#ifndef NOTRANSF
#define NOTRANSF -1
#endif

class Parameters : public MathOp {
 public:
  Parameters(int nSup);

  Parameters(const vec & x, const int transf, const int check=1);
  void setPars(const vec & x, const int transf, const int check=1);
  double getPar(const int ind);

  static int numberOfSuperPositions(const vec & par);
  vec asvector();

  int checkPars(const int transf);

  mat gradient(const int transf);

  //  Parameters extractPars(const vec & par);
  vec extractParsInv(const int transf); //const Parameters & parObj);

  Parameters(const double mu_, const double psi_,
	     const vec & omega_, const vec & lambda_, const int transf_, const int check=1);

  void print() const;
  void print(const char * str) const;

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
