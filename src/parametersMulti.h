#ifndef PARAMETERSMULTI__
#define PARAMETERSMULTI__ 1

#ifndef NOTRANSF
#define NOTRANSF -1
#endif

class ParametersMulti : public MathOp {
 public:
  ParametersMulti(int nSup);

  ParametersMulti(const vec & x, const int transf);
  void setPars(const vec & x, const int transf);
  double getPar(const int ind);

  int checkPars(const int transf);

  //  Parametersmulti extractPars(const vec & par);
  vec extractParsInv(const int transf); //const Parametersmulti & parObj);

  static int numberOfSuperPositions(const vec & par);
    
  vec asvector();

  ParametersMulti(const vec mu_, const vec psi_,
		  const mat & omega_, const mat & lambda_,
		  const double phi21_, const int transf_);

  void print() const {
    for (int k=0;k<q;k++)
      Rprintf("mu %d: %8.5f\n", k, mu(k));

    for (int k=0;k<q+p;k++)
      Rprintf("psi %d: %8.5f\n", k, psi(k));

    int nsup = lambda.n_cols;
    for (int k=0;k<q+p;k++) {
      Rprintf("lambda %1d: ", k);
      for (int i=0;i<nsup;i++)
	Rprintf("%8.5f ", lambda(k,i));
      Rprintf("\n");
    }
    for (int k=0;k<q+p;k++) {
      Rprintf("omega %d: ", k);
      for (int i=0;i<nsup;i++)
	Rprintf("%8.5f ", omega(k,i));
      Rprintf("\n");
    }
    Rprintf("phi21 %6.4f\n", phi(2,1));
  };

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
