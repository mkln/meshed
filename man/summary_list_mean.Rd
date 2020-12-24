\name{summary_list_mean}
\alias{summary_list_mean}
\title{Arithmetic mean of matrices in a list}
\description{For a list of matrices \eqn{\{ X^{(1)}, \dots, X^{(L)} \}}, all of the same dimension, this function computes the matrix \eqn{\bar{X}} with \eqn{i,j} entry \eqn{ \bar{X}_{i,j} = \frac{1}{L}\sum_{ l=1 }^{L} X_{ i,j }^{(l)} }. This function does not run any check on the dimensions and uses OpenMP if available.}
\usage{
summary_list_mean(x, num_threads=1)
}
\arguments{
\item{x}{A list of matrices of the same dimension}
\item{num_threads}{Number of OpenMP threads to use}
}
\value{The matrix of mean values.}
\examples{
L <- 10
x <- lapply(1:L, function(i) matrix(runif(12), ncol=3)) 
summary_list_mean(x)
}