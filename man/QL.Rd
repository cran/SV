\name{QL}
\alias{QL}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Quasi likelihood estimation}
\description{
  Quasi likelihood estimation of stochastic volatility model for
  exchange rate data.
}
\usage{
QL(datfile, mu=0.015, psi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
   minlambda=0, maxlambda=2, transf=0, par=NULL, verbose=FALSE, addPenalty=TRUE, nObs=NA,
   checkGradient=FALSE, gradtol=0.01, useRoptimiser=FALSE, updatePars=NULL, sandwich=TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{datfile}{A character string giving the name and path of
    the ASCII data file to be read.}
  \item{mu}{Start value for mu.}
  \item{psi}{Start value for psi.}
  \item{lambda}{Start values for lambda}
  \item{omega}{Start values for omega}
  \item{minlambda}{Minimum value for lambda.}
  \item{maxlambda}{Maximum value for lambda.}
  \item{transf}{0 or 1.}
  \item{par}{Alternative for specifying start values. 'par' is the
    unconstrained parameter vector.}
  \item{verbose}{If TRUE, information about estimation are
    output to screen. Default TRUE.}
  \item{addPenalty}{Add a penalty term for lambda.}
  \item{nObs}{Limits the data to the last 'nObs' observations.}
  \item{checkGradient}{For debug purposes.}
  \item{gradtol}{Convergence criteria}
  \item{useRoptimiser}{0: Use own implemented bfgs algorithm, 1: use R optimising routine}
  \item{updatePars}{Choose which parameters to optimise. Default: all parameters}
  \item{sandwich}{FALSE: Use the inverse hessian matrix as covariance
    matrix for the parameters, TRUE: use the sandwich
    approximation. Default TRUE.}
}
%\details{
%   If necessary, more details than the description above 
%}
\value{
 An object of class \code{ql} is returned.
}
\references{
    \url{http://folk.uio.no/skare/SV}

    Arvid Raknerud and Øivind Skare
    Indirect inference methods for stochastic volatility models
    based on non-Gaussian Ornstein-Uhlenbeck processes.
    Discussion Papers No. 601, December 2009
    Statistics Norway, Research Department
    }
%  Øivind Skare, Nuala Sheehan, and Thore Egeland
%  Identification of distant family relationships
%  Bioinformatics Advance Access published on July 6, 2009.}
\author{Øivind Skare \email{oivind.skare@medisin.uio.no}}
%\note{ further notes 
%
% Make other sections like Warning with \section{Warning }{....} 
%}
\seealso{\code{\link{IndirectInference}}}
\examples{
  help(QL) # R documentation of the QL function
  \dontrun{obj <- QL("example1.dat", mu=0.0, psi=0.1, lambda=0.05, omega=0.1)}
  \dontrun{print(obj)} # estimates and information about the optimisation
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{iteration}
%\keyword{htest}% __ONLY ONE__ keyword per line