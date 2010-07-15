#include <iostream>
#include "armadillo"

using namespace arma;
using namespace std;


int main(int argc, char** argv) {
  mat A(5,3);
  mat B(3,2);
  vec x(3);

  for (int i=0;i<5;i++)
    for (int j=0; j<3;j++)
      A(i,j) = fabs(i-j);

  for (int i=0;i<3;i++)
    for (int j=0; j<2;j++)
      B(i,j) = i+j;

  for (int i=0;i<3;i++)
    x(i) = i*i;

  mat C = A * B;

  vec y = A * x;

  A.print("A =");
  cout << "B =" << endl << B << endl;
  C.print("C =");
  x.print("x =");
  y.print("y =");
  //  A.row(1).print("A(1,) =");
  vec a1 = A.col(1);
  a1.print("a1 =");
  A.col(0) = a1;
  A.print("A =");

  mat Asub = A.submat(0,0,1,1);
  Asub.print("Asub =");
  vec xsub = x.rows(0,1);
  xsub.print("xsub =");

  int n = x.n_elem;
  cout << n << endl;

  cout << accu(x) << endl;

  cout << "GIVE x(0): ";
  cin >> x(0);

  mat AA = "1 2 x(0); 4 5 6; 7 8 9;"; // does not work!
  AA.print("AA =");

  vec z = "1 2 3 4 5 6 7 8 9";
  mat Z = zeros(3,3);
  Z.print("Z=");

  int * a = new int[10];
  
  for (int i=0;i<10;i++)
    a[i] = (i-4)*(i-4);


  double tmp = sum(z);
  Z(1,1) = tmp;

  mat X = rand<mat>(3,3);
  mat Q;
  mat R;
  qr(Q,R,X);

  vec x2 = solve(AA, x);

  mat AA_sym = AA+trans(AA);
  mat U = AA_sym * AA_sym;
  U.print("U=");
  mat Uc = chol(U);
  Uc.print("Uc=");
  vec eigval;
  mat eigvec;
  eig_sym(eigval, eigvec, U);
  mat Ui = inv(U);

  X = rand<mat>(1,3);
  mat U2 = trans(X.row(0)) * X.cols(0,0) * X.row(0) + rand<mat>(3,3);
  U2.print("U2= ");
  return 0;
}
