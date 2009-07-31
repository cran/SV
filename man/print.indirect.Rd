\name{print.indirect}
\alias{print.indirect}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Print a indirect object}
\description{
 Print basic information about a indirect object
}
\usage{
\method{print}{indirect}(x, ...) 
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{A \code{indirect} object, i.e. the result of running \code{IndirectInference}.}
  \item{...}{Other arguments, passed on to \code{print}.}
}
%\details{}
%\references{}
\author{Oivind Skare\cr
  Division of Epidemiology\cr
  Norwegian Institute of Public Health\cr
  \email{oivind.skare@medisin.uio.no}}
\note{Further information is found on the web page
}
\seealso{\code{\link{IndirectInference}}}
\examples{

\dontrun{
obj <- IndirectInference("example1.dat", nSim=10, mu=0.0, psi=0.1, lambda=0.05, omega=0.1)
print(obj)
}
}