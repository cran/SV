\name{QLmulti}
\alias{QLmulti}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Quasi likelihood estimation for bivariate data}
\description{
  Quasi likelihood estimation of stochastic volatility model for
  bivariate exchange rate data.
}
\usage{
QLmulti(datfile, mu=rep(0.015,2), psi=rep(0.1,3),
                    lambda=rep(c(0.5, 0.05),3),
                    omega=rep(c(0.1, 0.1),3), phi21=0.2,
                    minlambda=c(0,0,0), maxlambda=c(2,2,2), transf=0, par=NULL, verbose=FALSE, addPenalty=TRUE, nObs=NA,
                    checkGradient=FALSE, gradtol=0.01, useRoptimiser=FALSE, updatePars=NULL, sandwich=TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{datfile}{A character string giving the name and path of
    the ASCII data file to be read.}
  \item{mu}{A vector of length 2 containing start values for mu_1 and mu_2.}
  \item{psi}{A vector of length 3 with start values for psi.}
  \item{lambda}{Start values for lambda. A vector of length 3 times
    number of superposition terms (1 or 2). If number of superposition
    terms equal 2, then the order of lambda vector is  lambda_11,
    lambda_12, lambda21, lambda_22, lambda_31, lambda_32. Here lambda_ij
  is the superposition term j for volatility process i.}
  \item{omega}{Start values for omega^2. A vector of length 3 times
    number of superposition terms (1 or 2). If number of superposition
    terms equal 2, then the order of omega2 vector is  omega2_11,
    omega2_12, omega221, omega2_22, omega2_31, omega2_32. Here omega2_ij
  is the superposition term j for volatility process i.}
  \item{phi21}{Start value for phi21.}
  \item{minlambda}{A vector of length 3 containing minimum values for
    lambda for each of the three volatility processes.}
  \item{maxlambda}{A vector of length 3 containing maximum values for
    lambda for each of the three volatility processes.}
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
  \item{updatePars}{Choose which parameters to optimise. Default: all
    parameters}
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
    Multivariate stochastic volatility
               models based on non-Gaussian
               Ornstein-Uhlenbeck processes.
    Discussion Papers No. 614, December 2009
    Statistics Norway, Research Department
    }
\author{Øivind Skare \email{oivind.skare@medisin.uio.no}}
%\note{ further notes 
%
% Make other sections like Warning with \section{Warning }{....} 
%}
\seealso{\code{\link{QL}}}
\examples{
  help(QLmulti) # R documentation of the QLmulti function
  \dontrun{obj <- QLmulti("exampleMulti.dat", mu=c(0.0, 0.0), psi=rep(0.1, 3), lambda=rep(0.1, 3), omega=rep(0.1, 3), phi21=1)}
  \dontrun{print(obj)} # estimates and information about the optimisation
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{iteration}
%\keyword{htest}% __ONLY ONE__ keyword per line
