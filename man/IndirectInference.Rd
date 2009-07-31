\name{IndirectInference}
\alias{IndirectInference}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Indirect inference estimation}
\description{Estimation using indirect inference methods.
}
\usage{
IndirectInference(datfile, nSim = 10, mu = 0.015, psi = 0.1, lambda = c(0.5, 
                  0.05), omega = c(0.1, 0.1), minlambda = 0, maxlambda = 2, 
                  transf = 0, par = NULL, print.level = 1, addPenalty = TRUE, 
                  nObs = NA, checkGradient = FALSE, ftol = 0.1, ftol.weak = 1, 
                  gradtol = 1e-04, useRoptimiser = FALSE, initialSteepestDescent = TRUE, 
                  test.seed = -117)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{datfile}{A character string giving the name and path of
    the ASCII data file to be read.}
  \item{nSim}{Number of simulated log-return time series. The indirect
    inference estimates are based on individual estimates that are
    obtained for each simulation.}
  \item{mu}{Start value for mu.}
  \item{psi}{Start value for psi.}
  \item{lambda}{Start values for lambda. Number of terms equal number of
  super position terms.}
  \item{omega}{Start values for omega. Number of terms equal number of
  super position terms.}
  \item{minlambda}{Minimum value for lambda.}
  \item{maxlambda}{Maximum value for lambda.}
  \item{transf}{0 or 1.}
  \item{par}{Alternative for specifying start values. 'par' is the
    unconstrained parameter vector.}
  \item{print.level}{If 0, no information about estimation are output to
    screen. The higher value of print.level (1,2,3), the more
    information are output to screen.}
  \item{addPenalty}{Add a penalty term for lambda.}
  \item{nObs}{Limits the data to the last 'nObs' observations.}
  \item{checkGradient}{For debug purposes.}
  \item{ftol}{Convergence criteria}
  \item{ftol.weak}{Convergence criteria}
  \item{gradtol}{Convergence criteria}
  \item{useRoptimiser}{0: Use own implemented bfgs algorithm, 1: use R optimising routine}
  \item{initialSteepestDescent}{Use steepest descent as first initial
    direction in the conjugate direction algorithm.}
  \item{test.seed}{Input seed}
}
\details{
Indirect inference methods: First, a quasi-likelihood for the actual data is estimated.
This quasi-likelihood is based on an approximative Gaussian state space
representation of the Ornstein-Uhlenbeck (OU) based
model. Next, simulations are made from the data generating OU-model for given parameter
values. The indirect inference estimator is the parameter value in the OU-model which gives the best
"match" between the quasi-likelihood estimator for the actual data and the quasi-likelihood estimator
for the simulated data.

The estimation may however be very cpu-demanding.
}
\value{
  An object of class \code{indirect} is returned.
}
%\references{
%  \url{http://folk.uio.no/thoree/FEST}
  
%  Øivind Skare, Nuala Sheehan, and Thore Egeland
%  Identification of distant family relationships
%  Bioinformatics Advance Access published on July 6, 2009.}
\author{Øivind Skare \email{oivind.skare@medisin.uio.no}}
%\note{ further notes 
%
% Make other sections like Warning with \section{Warning }{....} 
%}
\seealso{\code{\link{QL}}, \code{\link{SimulateVolatility}}}
\examples{
   help(IndirectInference) # R documentation of the IndirectInference function
   \dontrun{resind <- IndirectInference("example1.dat", nSim=10, lambda=0.05, omega=0.1)}
   \dontrun{print(resind)} # estimates and information about the optimisation
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{iteration}
%\keyword{htest}% __ONLY ONE__ keyword per line
