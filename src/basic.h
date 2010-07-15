#ifndef BASIC__
#define BASIC__ 1

int argmin(vec cz);
void stop_own(const char * str);
void warning_own(const char * str);
int equal(const double x, const double x0, const double tol);
void saveToFile(vec ysim, int index);
int checkVec(const vec & x, const char * str);
int checkMat(const mat & x, const char * str);
void writeData(const vec & yret, char * filename);

struct FunctionValue {
  double f;
  vec df;

  FunctionValue(double f_, vec df_) {
    f = f_;
    df = df_;
  };
  FunctionValue(double f_) {
    f = f_;
  };
  FunctionValue() {
    f = 0;
  };
  void print() {
    Rprintf("f = %6.4f\n", f);
    df.print("df=");
  }
};

#endif
