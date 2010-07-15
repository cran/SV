\name{SimulateVolatility}
\alias{SimulateVolatility}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Simulate and plot log-return data and latent volatility process}
\description{
  Simulate and plot log-return data and latent volatility process.
}
\usage{
SimulateVolatility(nSim=1, nTimes, par=NULL, mu=0, xi=0.5, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                   minlambda=0, maxlambda=2, transf=0, useParVec=FALSE,
verbose=FALSE, test.seed=NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{nSim}{Number of simulated data sets}
  \item{nTimes}{Length of time series}
  \item{par}{Alternative for specifying model parameters. 'par' is the
    unconstrained parameter vector.}
  \item{mu}{Value for mu}
  \item{xi}{Value for xi}
  \item{lambda}{Values for lambda}
  \item{omega}{Values for omega}
  \item{minlambda}{Minimum value for lambda}
  \item{maxlambda}{Maximum value for lambda}
  \item{transf}{0 or 1.}
  \item{useParVec}{If TRUE, use 'par' to specify model parameters,
    otherwise use the 'mu', 'xi', 'lambda' and 'omega'
    arguments. Default is FALSE.}
  \item{verbose}{If TRUE, information about estimation are
    output to screen. Default is TRUE.}
  \item{test.seed}{Input seed}
}
%\details{
%   If necessary, more details than the description above 
%}
\value{
  A list with elements:
  \item{logYRet}{Log-return data.}
  \item{sigma2}{The volatility series.}
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
%\seealso{\code{\link{IndirectInference}}}
\examples{
  help(SimulateVolatility) # R documentation of the SimulateVolatility function

  \dontrun{obj <- SimulateVolatility(1, 5000, mu=0.015, xi=0.5, lambda=0.1, omega=0.1)} #plots the sigma^2 and log-return data
  ## obj contains the sigma^2 and log-return data  (obj$sigma2, obj$logYRet)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{iteration}
%\keyword{htest}% __ONLY ONE__ keyword per line
