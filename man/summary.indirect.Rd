\name{summary.indirect}
\alias{summary.indirect}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Summary of a indirect object}
\description{
  Provides detailed information about a
  \code{indirect} object.
}
\usage{
\method{summary}{indirect}(object, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{A \code{indirect} object, i.e. the result of running \code{IndirectInference}.}
  \item{...}{Further arguments to be passed on to the summary function.}
}
%\references{}

\author{Oivind Skare\cr
  Division of Epidemiology\cr
  Norwegian Institute of Public Health\cr
  \email{oivind.skare@medisin.uio.no}}

%\note{}

\seealso{\code{\link{IndirectInference}}}
\examples{

\dontrun{

# Standard run with permutation test:
res <- IndirectInference("data.dat", nsim.perm=1000)
summary(res)
}
}