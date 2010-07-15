\name{IndirectInference}
\alias{IndirectInference}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Indirect inference estimation}
\description{Estimation using indirect inference methods.
}
\usage{
IndirectInference(datfile, nTimes=NA, mu=0.015, xi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                  minlambda=0, maxlambda=2, transf=0, par=NULL, print.level=1, addPenalty=TRUE,
                  nObs=NA,
                  checkGradient=FALSE, ftol=0.001, ftol.weak=1, gradtol=1e-4, useRoptimiser=FALSE,
                  initialSteepestDescent=TRUE, ITMAX=200,
                  test.seed=-117,
                  useQLestimateAsStartPar=TRUE,
                  simfile="", gradMax=1000^2)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{datfile}{A character string giving the name and path of
    the ASCII data file to be read.}
%  \item{nSim}{Number of simulated log-return time series. The indirect
%    inference estimates are based on individual estimates that are
%    obtained for each simulation.}
  \item{nTimes}{Length of simulation log-return time series. If not
    specified, it defaults to
  the length of the observed log-return series.}
  \item{mu}{Start value for mu.}
  \item{xi}{Start value for xi.}
  \item{lambda}{Start values for lambda. Number of terms equal number of
  super position terms.}
  \item{omega}{Start values for omega. Number of terms equal number of
  super position terms.}
  \item{minlambda}{Minimum value for lambda. Defaults to 0.}
  \item{maxlambda}{Maximum value for lambda. Defaults to 2.}
  \item{transf}{Internal parameter. Values 0 or 1. 0 gives the default internal parmeterisaton of the unconstrained parameter vector.}
  \item{par}{Alternative for specifying start values. 'par' is the
    unconstrained parameter vector.}
  \item{print.level}{If 0, no information about estimation are output to
    screen. The higher value of print.level (1,2,3), the more
    information are output to screen.}
  \item{addPenalty}{Add a penalty term for lambda. Defaults to TRUE.}
  \item{nObs}{Limits the data to the last 'nObs' observations.}
  \item{checkGradient}{For debug purposes.}
  \item{ftol}{Convergence criteria. If function value is less than ftol,
  then optimisation is finished. Defaults to 1e-3.}
  \item{ftol.weak}{Convergence criteria. If function value does not
    achieve ftol.weak, the optimisation is restarted (with a different
    seed). Defaults to 1.}
  \item{gradtol}{Internal parameter. Convergence criteria for Quasi likelihood
    estimation. Defaults to 1e-4.}
  \item{useRoptimiser}{0: Use own implemented bfgs algorithm, 1: use R optimising routine}
  \item{initialSteepestDescent}{Use steepest descent as first initial
    direction in the conjugate direction algorithm.}
  \item{ITMAX}{Maximum number of iterations in the conjugate direction
    optimisation algorithm. Defaults to 200.}
  \item{test.seed}{Input seed}
  \item{useQLestimateAsStartPar}{If TRUE, use the QL estimate as
    starting value for the indirect inference optimisation. If FALSE,
    use the input starting values specified for the parameters. Defaults
  to TRUE}
  \item{simfile}{If different from empty string, the simulated data
    corresponding to
    the indirect inference estimate is written to this file.}
  \item{gradMax}{Internal parameter.}
}
\details{
Indirect inference methods:  First, an
auxiliary model based on an approximate linear state space
representation of the OU-based model is estimated.
Then simulations are
made from the underlying OU-model for given parameter values.
A method of moments version of indirect inference is implemented:
The indirect inference estimator is the parameter value in the OU-model which minimizes, in a
weighted mean squared error sense, the score vector of the quasi-likelihood function for
the simulated data, when this score vector is evaluated at the quasi-likelihood estimator
obtained from the real data.

The estimation may be cpu-demanding.
}
\value{
  An object of class \code{indirect} is returned.
}
\references{
  \url{http://folk.uio.no/skare/SV}

  Paper accepted for publication in Computational Statistics & Data Analysis:
  Arvid Raknerud and Øivind Skare
  Indirect inference methods for stochastic volatility models
  based on non-Gaussian Ornstein-Uhlenbeck processes.

  Available online on:
  \url{http://www.sciencedirect.com}
%  Discussion Papers No. 601, December 2009
%  Statistics Norway, Research Department
}
\author{Øivind Skare \email{oivind.skare@medisin.uio.no}}
%\note{ further notes 
%
% Make other sections like Warning with \section{Warning }{....} 
%}
\seealso{\code{\link{QL}}, \code{\link{SimulateVolatility}}}
\examples{
   help(IndirectInference) # R documentation of the IndirectInference function
   \dontrun{resind <- IndirectInference("example1.dat", nTimes=5000, mu=0, xi=0.1, lambda=0.05, omega=0.1)}
   \dontrun{summary(resind)} # estimates and information about the optimisation
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{iteration}
%\keyword{htest}% __ONLY ONE__ keyword per line
