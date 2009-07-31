# -*- mode: autoconf; autoconf-indentation: 4; -*-
#
# SV configure.in 
#
# $Id$

# require at least autoconf 2.65
AC_PREREQ(2.63)

# Process this file with autoconf to produce a configure script.
AC_INIT([SV], [1.2.3])
##AC_CHECK_LIB(lapack_atlas, clapack_dgetrf, [HAVE_LAPACK_ATLAS=TRUE], [HAVE_LAPACK_ATLAS=FALSE])

##if test x"${HAVE_LAPACK_ATLAS}" == x"FALSE" ; then
##    AC_MSG_ERROR([Library 'lapack_atlas' missing.])
##fi

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
   echo "could not determine R_HOME"
   exit 1
fi
CC=`"${R_HOME}/bin/R" CMD config CC`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`

# We are using C++
AC_LANG(C++)
AC_REQUIRE_CPP
#AC_PROG_CC
#AC_PROG_CXX

## look for Armadillo header

## check for header and ability to link
## first for headers Debian has in libarmadillo-dev
AC_CHECK_HEADER(armadillo,,
                [AC_MSG_ERROR([ERROR: Armadillo headers required; use '-Iincludedir' in CXXFLAGS for unusual locations.])])
## second check the lib itself
arma_savedlibs="$LIBS"
#LIBS="$LIBS -larmadillo -lpthread"
LIBS="$LIBS -larmadillo"
AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <armadillo>]], 
				[[arma::mat A = arma::eye<arma::mat>(5,5);]])],
				[],
				[AC_MSG_ERROR([ERROR: Armadillo test program did not link, maybe add '-Llibdir' to LDFLAGS.])])
LIBS="$arma_savedlibs"


## and check for minimum version: 0.9.0 had upstream changes
AC_MSG_CHECKING([if Armadillo version >= 0.9.0])
AC_RUN_IFELSE([AC_LANG_SOURCE([[
#include <armadillo>
int main() {
  struct arma::arma_version av;
  int arma = 10000*av.major + 100*av.minor + av.patch;
  if (arma >= 900) { /* 900 corresponds to 0.9.0 */
    exit(0);
  } else {
    exit(1);
  }
}
]])],
[arma_version090=1],
[arma_version090=0],
[arma_version090=1])
if test x"${arma_version090}" == x"0"; then
  AC_MSG_RESULT([no])
else
  AC_MSG_RESULT([yes])
fi

#AC_DEFUN([AC_PROG_R], [AC_CHECK_PROG(R,R,yes)])
#AC_PROG_R


## now use all these
AC_SUBST([PKG_LIBS],["${PKG_LIBS} -larmadillo"])

AC_OUTPUT
echo "Completed configuration and ready to build."
