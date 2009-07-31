\name{print.ql}
\alias{print.ql}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Print a ql object}
\description{
 Print basic information about a ql object
}
\usage{
\method{print}{ql}(x, ...) 
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{A \code{ql} object, i.e. the result of running \code{QL}.}
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
\seealso{\code{\link{QL}}}
\examples{

\dontrun{
obj <- QL("example1.dat", mu=0.0, psi=0.1, lambda=0.05, omega=0.1)
print(obj)
}
}