\name{summary_list_q}
\alias{summary_list_q}
\title{Quantiles of elements of matrices in a list}
\description{For a list of matrices \eqn{\{ X^{(1)}, \dots, X^{(L)} \}}, all of the same dimension, this function computes the matrix \eqn{\hat{X}} with \eqn{i,j} entry \eqn{ \hat{X}_{i,j} = } \code{quantile(}\eqn{ \{ X_{ i,j }^{(l)} \}_{l=1}^L }\code{, q)}. This function does not run any check on the dimensions and uses OpenMP if available. This is only a convenience function that is supposed to speed up quantile computation for very large problems. The results may be slightly different from R's \code{quantile} which should be used for small problems.}
\usage{
summary_list_q(x, q, n_threads=1)
}
\arguments{
\item{x}{A list of matrices of the same dimension.}
\item{q}{A number between 0 and 1.}
\item{n_threads}{integer number of OpenMP threads. This is ineffective if \code{meshed} was not compiled with OpenMP support.}
}
\value{The matrix of quantiles.}
\author{
  Michele Peruzzi \email{michele.peruzzi@duke.edu}
}
\examples{
# make some data into a list
set.seed(2021)
L <- 200
x <- lapply(1:L, function(i) matrix(runif(300), ncol=3)) 
quant_done1 <- summary_list_q(x, .9)
}