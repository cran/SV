\name{summary.ql}
\alias{summary.ql}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Summary of a ql object}
\description{
  Provides detailed information about a
  \code{ql} object.
}
\usage{
\method{summary}{ql}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{A \code{ql} object, i.e. the result of running \code{QL}.}
  \item{...}{Further arguments to be passed on to the summary function.}
}
%\references{}

\author{Oivind Skare\cr
  Division of Epidemiology\cr
  Norwegian Institute of Public Health\cr
  \email{oivind.skare@medisin.uio.no}}

%\note{}

\seealso{\code{\link{QL}}}
\examples{

\dontrun{
obj <- QL("example1.dat", mu=0.0, psi=0.1, lambda=0.05, omega=0.1)
summary(obj)
}
}