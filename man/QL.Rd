\name{QL}
\alias{QL}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Quasi likelihood estimation}
\description{
  Quasi likelihood estimation of stochastic volatility model for
  exchange rate data.
}
\usage{
QL(datfile, mu=0.015, xi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
   minlambda=0, maxlambda=2, transf=0, par=NULL, verbose=FALSE, addPenalty=TRUE, nObs=NA,
   checkGradient=FALSE, gradtol=0.01, useRoptimiser=FALSE, updatePars=NULL, sandwich=TRUE, gradMax=1000^2)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{datfile}{A character string giving the name and path of
    the ASCII data file to be read.}
  \item{mu}{Start value for mu.}
  \item{xi}{Start value for xi.}
  \item{lambda}{Start values for lambda.}
  \item{omega}{Start values for omega.}
  \item{minlambda}{Minimum value for lambda. Defaults to 0.}
  \item{maxlambda}{Maximum value for lambda. Defaults to 2.}
  \item{transf}{Internal parameter. Values 0 or 1. 0 gives the default internal parmeterisaton of the unconstrained parameter vector.}
  \item{par}{Alternative for specifying start values. 'par' is the
    unconstrained parameter vector.}
  \item{verbose}{If TRUE, information about estimation are
    output to screen. Default TRUE.}
  \item{addPenalty}{Add a penalty term for lambda. Defaults to TRUE.}
  \item{nObs}{Limits the data to the last 'nObs' observations.}
  \item{checkGradient}{For debug purposes.}
  \item{gradtol}{Convergence criteria. Defaults to 1e-4.}
  \item{useRoptimiser}{0: Use own implemented bfgs algorithm, 1: use R optimising routine}
  \item{updatePars}{Choose which parameters to optimise. Default: all parameters}
  \item{sandwich}{FALSE: Use the inverse hessian matrix as covariance
    matrix for the parameters, TRUE: use the sandwich
    approximation. Default TRUE.}
  \item{gradMax}{Internal parameter.}
}
%\details{
%   If necessary, more details than the description above 
%}
\value{
 An object of class \code{ql} is returned.
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
\seealso{\code{\link{IndirectInference}}}
\examples{
  help(QL) # R documentation of the QL function
  \dontrun{obj <- QL("example1.dat", mu=0.0, xi=0.1, lambda=0.05, omega=0.1)}
  \dontrun{summary(obj)} # estimates and information about the optimisation
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{iteration}
%\keyword{htest}% __ONLY ONE__ keyword per line
