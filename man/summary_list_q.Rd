\name{summary_list_q}
\alias{summary_list_q}
\title{Percentiles of elements of matrices in a list}
\description{For a list of matrices \eqn{\{ X^{(1)}, \dots, X^{(L)} \}}, all of the same dimension, this function computes the matrix \eqn{\hat{X}} with \eqn{i,j} entry \eqn{ \hat{X}_{i,j} = } \code{quantile(}\eqn{ \{ X_{ i,j }^{(l)} \}_{l=1}^L }\code{, q)}. This function does not run any check on the dimensions and uses OpenMP if available. This is only a convenience function that is supposed to speed up quantile computation for very large problems. The results may be slightly different from R's \code{quantile} which should be used for small problems.}
\usage{
summary_list_q(x, q)
}
\arguments{
\item{x}{A list of matrices of the same dimension}
\item{q}{A number between 0 and 1.}
}
\value{The matrix of mean values.}
\examples{
\dontrun{
# length of thinned chain
L <- 2000
# like 3 spatial factors on only 4000 spatial locations
x <- lapply(1:L, function(i) matrix(runif(12000), ncol=3)) 
quant_done1 <- summary_list_q(x, .9)
# compare with R
xarray <- array(as.numeric(unlist(x)), dim=c(dim(x[[1]]), L)) 
quant_done2 <- xarray \%>\% apply(1:2, quantile, .9)
mean(abs(quant_done2 - quant_done1)) # there is a small difference

# but this is a small problem and base R is 21 times slower
rbenchmark::benchmark(xarray \%>\% apply(1:2, quantile, .9),
                      summary_list_q(x, .9),
                      replications=10)
}
}