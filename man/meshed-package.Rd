\name{meshed-package}

\alias{meshed}
\alias{meshed-package}

\docType{package}

\title{
methods for fitting regression models based on Meshed Gaussian Processes
}

\description{
	\code{meshed} is a flexible package for Bayesian regression analysis on spatial or spatiotemporal datasets. The main function for fitting regression models is \code{\link{spmeshed}}, which outputs posterior samples obtained from Markov chain Monte Carlo which can be summarised using standard tools. The package also provides a function \code{\link{rmeshedgp}} for quickly simulating correlated spatial or spatiotemporal data at a very large number of locations.
}

\details{
  The functions \code{\link{rmeshedgp}} and \code{\link{spmeshed}} are provided for prior or posterior sampling (respectively) of Bayesian regression models based on Meshed Gaussian Processes as introduced by Peruzzi, Banerjee, and Finley (2020). Posterior sampling via \code{\link{spmeshed}} proceeds by default via GriPS as detailed in Peruzzi, Banerjee, Dunson, and Finley (2021). When at least one outcome is not modeled with Gaussian errors, sampling proceeds taking advantage of Metropolis-adjusted Langevin dynamics.
}

\author{
Michele Peruzzi <michele.peruzzi@duke.edu>
}

\references{
	Peruzzi, M., Banerjee, S., and Finley, A.O. (2020)
  Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains. \emph{Journal of the American Statistical Association}, in press. \doi{10.1080/01621459.2020.1833889}
  
  Peruzzi, M., Banerjee, S., Dunson, D.B., and Finley, A.O.  (2021)
  Grid-Parametrize-Split (GriPS) for Improved Scalable Inference in Spatial Big Data Analysis. \url{https://arxiv.org/abs/2101.03579}
}

\keyword{ package }

\seealso{
	\code{\link{spmeshed}}, \code{\link{rmeshedgp}}
}
