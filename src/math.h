#ifndef MATHOP__
#define MATHOP__ 1

class MathOp {
 public:
  mat kronecker(const mat & A, const mat & B);
  vec vectorize(const mat & A);
  mat outer(const vec & x, const vec & y);
  vec proj(const vec & y, const vec & x);
  double Logit(double p);
  void gramSchmidt(mat & u, const mat & v);
  mat robustCholesky(const mat & S);
  vec findSteepestDescent(double (*func)(const vec &, int &), vec p, const int npar, const double h);
 private:
};



// projection of y on x
inline vec MathOp::proj(const vec & y, const vec & x) {
  return (dot(x,y)/dot(x,x)) * x;
}

#endif
