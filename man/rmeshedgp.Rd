\name{rmeshedgp}
\alias{rmeshedgp}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Prior sampling from a Meshed Gaussian Process
}
\description{
Generates samples from a (univariate) MGP assuming a cubic directed acyclic graph and axis-parallel domain partitioning.
}
\usage{
rmeshedgp(coords, theta, 
  axis_partition = NULL, block_size = 100, 
  n_threads=1, cache=TRUE, verbose=FALSE, debug=FALSE)
}

\arguments{
  \item{coords}{
  matrix of spatial or spatiotemporal coordinates with \eqn{d=2} or \eqn{d=3} columns for spatial or spatiotemporal data, respectively.
}
  \item{theta}{
  vector with covariance parameters. If \eqn{d=2} and \code{theta} is a 2-dimensional vector then \eqn{\theta = (\phi, \sigma^2)} where \eqn{\phi} is the spatial decay and \eqn{\sigma^2} is the spatial variance in the exponential covariance model. If \eqn{d=2} and \code{theta} is a 3-dimensional vector then \eqn{\theta = (\phi, \nu, \sigma^2)} and a Matern model with smoothness \eqn{\nu} is used instead. If \eqn{d=3}, \code{theta} must be a 4-dimensional vector and \eqn{\theta=(a, \phi, b, \sigma^2)} using Gneiting's non-separable spatiotemporal covariance detailed below.
}
  \item{axis_partition}{
  integer vector of length \eqn{d} with the number of intervals along which each axis should be partitioned. The domain will be partitioned into \code{prod(axis_partition)} blocks. This argument can be left blank when using \code{block_size}.
}
  \item{block_size}{
  integer specifying the (approximate) size of the blocks, i.e. how many spatial or spatiotemporal locations should be included in each block. Note: larger values correspond to an MGP that is closer to a full GP, but require more expensive computations. 
}
\item{n_threads}{
  integer number of OpenMP threads. This is ineffective if \code{meshed} was not compiled with OpenMP support.
}
\item{cache}{
  bool: whether to use cache. Some computational speedup is associated to \code{cache=TRUE} if \code{coords} are a grid. 
}
\item{verbose}{
  bool: print some messages.
}
\item{debug}{
  bool: print more messages.
}
}
\details{
  Gaussian processes (GPs) lack in scalability to big datasets due to the assumed unrestricted dependence across the spatial or spatiotemporal domain.
  \emph{Meshed} GPs instead use a directed acyclic graph (DAG) with patterns, called \emph{mesh}, to simplify the dependence structure across the domain. Each DAG node corresponds to a partition of the domain. MGPs can be interpreted as approximating the GP they originate from, or as standalone processes that can be sampled from. This function samples random MGPs and can thus be used to generate big spatial or spatiotemporal data. 
  The only requirement to sample from a MGP compared to a standard GP is the specification of the domain partitioning strategy. Here, either \code{axis_partition} or \code{block_size} can be used; the default \code{block_size=100} can be used to quickly sample smooth surfaces at millions of locations. 
  
  Just like in a standard GP, one needs a covariance function or kernel which can be set as follows. 
  For spatial data (\eqn{d=2}), the length of \code{theta} determines which model is used (see above). Letting \eqn{h = \| s-s' \|} where \eqn{s} and \eqn{s'} are locations in the spatial domain, the exponential covariance is defined as: 
  \deqn{C(h) = \sigma^2 \exp \{ - \phi h \},}
  whereas the Matern model is
  \deqn{C(h) = \sigma^2  \frac{2^{1-\nu}}{\Gamma(\nu)} \phi^{\nu} h^{\nu} K_{\nu} ( \phi h ),
  }
  where \eqn{K_{\nu}} is the modified Bessel function of the second kind of order \eqn{\nu}.
  For spatiotemporal data (\eqn{d=3}) the covariance function between locations \eqn{(s, t)} and \eqn{(s', t')} with distance \eqn{h = \| s-s' \|} and time lag \eqn{u = \| t-t' \|} is defined as
  \deqn{ C(h, u) = \sigma^2 / (a u + 1) \exp \{ -\phi h (a u + 1)^{-b/2} \}, }
  which is a special case of non-separable spacetime covariance as introduced by Gneiting (2002).
}
\value{
  data.frame with the (reordered) supplied coordinates in the first \code{d} columns, and the MGP sample in the last column, labeled \code{w}.
}
\references{
  Gneiting, T (2002) Nonseparable, Stationary Covariance Functions for Space-Time Data. \emph{Journal of the American Statistical Association}. \doi{10.1198/016214502760047113}
  
  Peruzzi, M., Banerjee, S., and Finley, A.O. (2020)
  Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains. \emph{Journal of the American Statistical Association}, in press. \doi{10.1080/01621459.2020.1833889}
}
\author{
Michele Peruzzi <michele.peruzzi@duke.edu>
}

\examples{
  
library(ggplot2)
library(magrittr)
library(meshed)

# spatial domain (we choose a grid to make a nice image later)
# this generates a dataset of size 6400
xx <- seq(0, 1, length.out=80)
coords <- expand.grid(xx, xx) \%>\%
  as.matrix()

raster_plot <- function(df){
  ggplot(df, aes(Var1, Var2, fill=w)) +
    geom_raster() +
    scale_fill_viridis_c() +
    theme_minimal() }

# spatial data, exponential covariance
# phi=14, sigma^2=2
simdata <- rmeshedgp(coords, c(14, 2))
raster_plot(simdata)

# spatial data, matern covariance
# phi=14, nu=1, sigma^2=2
simdata <- rmeshedgp(coords, c(14, 1, 2))
raster_plot(simdata)

# spacetime data, gneiting's covariance
# 64000 locations
stcoords <- expand.grid(xx, xx, seq(0, 1, length.out=10))

# it should take less than a couple of seconds
simdata <- rmeshedgp(stcoords, c(1, 14, .5, 2))
# plot data at 7th time period
raster_plot(simdata \%>\% dplyr::filter(Var3==unique(Var3)[7])) 

}

