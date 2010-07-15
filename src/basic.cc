/*Include Files:*/
#include <iostream>


#include <R.h>

#include "armadillo"
using namespace std;
//using namespace Numeric_lib;

using namespace arma;

#include "basic.h"

int argmin(vec cz) {
  int n = cz.n_elem;

  int index = 0;
  double cz_min = cz(0);
  for (int i=1;i<n;i++) {
    if (cz(i) < cz_min) {
      cz_min = cz(i);
      index = i;
    }
  }
  return index;
}

void stop_own(const char * str) {
  Rprintf("Error: %s\n", str);
  exit(-1);
}

void warning_own(const char * str) {
  Rprintf("Warning: %s\n", str);
}

int equal(const double x, const double x0, const double tol) {
  return (fabs(x-x0) < tol);
}

void saveToFile(vec ysim, int index) {
  const int n = ysim.n_elem;
  vec y(n+1);
  y(0) = 1;
  for (int i=0;i<n;i++) {
    y(i+1) = y(i)*exp(ysim(i)/100.0);
  }

  string str;
  stringstream out;
  out << index;
  str = out.str();
  string filename = "simdat_output_" + str + ".txt";
  ofstream ost(filename.c_str());
  ost.precision(16);

  if (!ost) {
    //    error("Can't open input file", filename);
    cout << "Can't open input file " << filename << endl;
    exit(-1);
  }
  for (int i=0;i<n+1;i++) {
    ost << y(i) << endl;
  }
  Rprintf("saveToFile; y0 %6.4f y1 %6.4f\n", y(0), y(1)); 
  Rprintf("saveToFile; ysim0 %6.4f ysim1 %6.4f\n", ysim(0), ysim(1)); 
}

int checkVec(const vec & x, const char * str) {
  int n = x.n_elem;
  int error = 0;
  for (int i=0;i<n;i++) {
    if (isnan(x(i))) {
      error = 1;
      break;
    }
  }
  if (error) {
    Rprintf("Error(checkVec): %s is nan\n", str);
    x.print("Values =");
  }
  return error;
}

int checkMat(const mat & x, const char * str) {
  int n = x.n_rows;
  int m = x.n_cols;
  int error = 0;
  for (int i=0;i<n;i++) {
    for (int j=0;j<m;j++) {
      if (isnan(x(i,j))) {
	error = 1;
	break;
      }
    }
  }
  if (error) {
    Rprintf("Error(checkMat): %s is nan\n", str);
    x.print("Values =");
  }
  return error;
}


void writeData(const vec & yret, char * filename) {
  ofstream ost(filename);
  ost.precision(10);
  int ny = yret.n_elem;

  int nr = ny + 1;

  
  vec y(nr);
  y(0) = 1.0;
  ost << y(0) << endl;
  for (int j=1;j<nr;j++) {
    y(j) = y(j-1) * exp(yret(j-1)/100.0);
    ost << y(j) << endl;
  }
  ost.close();
}
