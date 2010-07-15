#ifndef PARAMETERSMULTI__
#define PARAMETERSMULTI__ 1

#ifndef NOTRANSF
#define NOTRANSF -1
#endif

class ParametersMulti : public MathOp {
 public:
  ParametersMulti(int nSup);

  ParametersMulti(const vec & x, const int transf, const int check=1);
  void setPars(const vec & x, const int transf, const int check=1);
  double getPar(const int ind);

  int checkPars(const int transf);

  //  Parametersmulti extractPars(const vec & par);
  vec extractParsInv(const int transf); //const Parametersmulti & parObj);

  static int numberOfSuperPositions(const vec & par);
    
  vec asvector();

  ParametersMulti(const vec mu_, const vec psi_,
		  const mat & omega_, const mat & lambda_,
		  const double phi21_, const int transf_, const int check=1);

  void print() const;
  void print(const char * str) const;

  static int q; // Number of observed currencies
  static int p; // default = q-1

  mat phi; // phi(2,1)

  vec psi; // p+q
  vec mu; // q

  mat omega; // (p+q):nSup
  mat lambda; // (p+q):nSup

  static vec minlambda;
  static vec maxlambda;

 private:
  vec extractParsInv1();
  vec extractParsInv2();
  void setPars0(const vec & x);
  void setPars1(const vec & x);
  void setPars2(const vec & x);
};

#endif
