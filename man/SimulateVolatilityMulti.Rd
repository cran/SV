\name{SimulateVolatilityMulti}
\alias{SimulateVolatilityMulti}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Simulate and plot log-return data and latent volatility process}
\description{
  Simulate and plot log-return data and latent volatility process.
}
\usage{
SimulateVolatilityMulti(nSim=1, nTimes, par=NULL, mu=rep(0.015,2), xi=rep(0.1,3),
                        lambda=rep(c(0.5, 0.05),3),
                        omega=rep(c(0.1, 0.1),3), phi21=0.2,
                        minlambda=c(0,0,0), maxlambda=c(2,2,2), transf=0, useParVec=FALSE, verbose=FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{nSim}{Number of simulated data sets}
  \item{nTimes}{Length of time series}
  \item{par}{Alternative for specifying model parameters. 'par' is the
    unconstrained parameter vector.}
  \item{mu}{A vector of length 2 containing values for mu_1 and mu_2.}
  \item{xi}{A vector of length 3 with values for xi.}
  \item{lambda}{values for lambda. A vector of length 3 times
    number of superposition terms (1 or 2). If number of superposition
    terms equal 2, then the order of lambda vector is  lambda_11,
    lambda_12, lambda21, lambda_22, lambda_31, lambda_32. Here lambda_ij
  is the superposition term j for volatility process i.}
  \item{omega}{Values for omega^2. A vector of length 3 times
    number of superposition terms (1 or 2). If number of superposition
    terms equal 2, then the order of omega2 vector is  omega2_11,
    omega2_12, omega221, omega2_22, omega2_31, omega2_32. Here omega2_ij
  is the superposition term j for volatility process i.}
\item{phi21}{Value for phi21.}
  \item{minlambda}{Minimum value for lambda}
  \item{maxlambda}{Maximum value for lambda}
  \item{transf}{0 or 1.}
  \item{useParVec}{If TRUE, use 'par' to specify model parameters,
    otherwise use the 'mu', 'xi', 'lambda' and 'omega'
    arguments. Default is FALSE.}
  \item{verbose}{If TRUE, information about estimation are
    output to screen. Default is TRUE.}
}
%\details{
%   If necessary, more details than the description above 
%}
\value{
  A list with elements:
  \item{logYRet}{Matrix of log-return data of dimension 2 x
    nTimes. logYRet[i,] contains the i'th log-return data.}
  \item{sigma2}{Matrix of volatility series of dimension 3 x
    nTimes. sigma2[i,] contains the i'th volatility series.}
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
  help(SimulateVolatilityMulti) # R documentation of the SimulateVolatilityMulti function

  par(mfrow=c(3,2))
  \dontrun{obj <- SimulateVolatilityMulti(1, 5500, mu=rep(0, 2), xi=c(0.4,0.4,0.1), lambda=c(0.005, 0.005, 0.28), omega=c(0.1, 0.1, 0.06), phi21=1)}
  ## obj contains the sigma^2 and log-return data
  ##    obj$sigma2[k,]: k'th sigma^2 process (k=1,2,3)
  ##    obj$logYRet[k,]: k'th log-return data (k=1,2) 
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{iteration}
%\keyword{htest}% __ONLY ONE__ keyword per line
