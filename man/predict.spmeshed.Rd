\name{predict.spmeshed}
\alias{predict.spmeshed}
\title{Posterior predictive sampling for models based on MGPs}
\description{Sample from the posterior predictive distribution of the outcomes at new spatial or spatiotemporal locations after MCMC.}
\usage{
\method{predict}{spmeshed}(object, newx, newcoords, 
    n_threads=4, verbose=FALSE, ...)
        
}
\arguments{
\item{object}{Object output from \code{spmeshed} with option \code{settings$saving=TRUE}.}

\item{newx}{matrix of covariate values at the new coordinates.}

\item{newcoords}{matrix of new coordinates.}

\item{n_threads}{integer number of OpenMP threads. This is ineffective if \code{meshed} was not compiled with OpenMP support.}

\item{verbose}{boolean for progress messagging.}

\item{...}{other arguments (unused).}
}
\details{
While this function can always be used to make predictions, in most cases it is more efficient to just include the prediction locations in the main data as \code{NA} values; \code{\link{spmeshed}} will sample from the posterior predictive distribution at those locations while doing MCMC. The \code{predict} method is only recommended when all 4 of the following are true:

(1) \code{spmeshed} was run with \code{settings$forced_grid=FALSE} and

(2) the prediction locations are uniformly scattered on the domain (or rather, they are not clustered as a large empty area) and

(3) the number of prediction locations is a large portion of the number of observed data points and 

(4) the prediction locations are not on a grid. 

In all other cases the main \code{spmeshed} function is setup to be more efficient in automatically performing predictions during MCMC.
}
\value{
\item{coords_out}{matrix with the prediction location coordinates (order updated after predictions).}

\item{preds_out}{array of dimension (\eqn{n_{o}, q, m}) where \eqn{n_{o}} is the number of prediction locations, \eqn{q} is the output dimension, \eqn{m} is the number of MCMC samples.}
}

\references{
  Peruzzi, M., Banerjee, S., and Finley, A.O. (2020)
  Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains. \emph{Journal of the American Statistical Association}, in press. \doi{10.1080/01621459.2020.1833889}
  
  Peruzzi, M., Banerjee, S., Dunson, D.B., and Finley, A.O.  (2021)
  Grid-Parametrize-Split (GriPS) for Improved Scalable Inference in Spatial Big Data Analysis. \url{https://arxiv.org/abs/2101.03579}
}

\author{
  Michele Peruzzi \email{michele.peruzzi@duke.edu}
}

\examples{

# toy example with tiny dataset and short MCMC
# on a univariate outcome
library(magrittr)
library(dplyr)
library(meshed)

set.seed(2021)

SS <- 12
n <- SS^2 # total n. locations, including missing ones

coords <- data.frame(Var1=runif(n), Var2=runif(n)) \%>\%
  as.matrix()

# generate data
sigmasq <- 2.3
phi <- 6
tausq <- .1
B <- c(-1,.5,1)

CC <- sigmasq * exp(-phi * as.matrix(dist(coords)))
LC <- t(chol(CC))
w <- LC \%*\% rnorm(n)
p <- length(B)
X <- rnorm(n * p) \%>\% matrix(ncol=p)
y_full <- X \%*\% B + w + tausq^.5 * rnorm(n)

set_missing <- rbinom(n, 1, 0.1)

simdata <- data.frame(coords,
                      y_full = y_full,
                      w_latent = w) \%>\%
  mutate(y_observed = ifelse(set_missing==1, NA, y_full))

# MCMC setup
mcmc_keep <- 500
mcmc_burn <- 100
mcmc_thin <- 2

y <- simdata$y_observed
ybar <- mean(y, na.rm=TRUE)

# training set
y_in <- (y-ybar)[!is.na(y)]
X_in <- X[!is.na(y),]
coords_in <- coords[!is.na(y),]

# suppose we dont want to have gridded knots
# i.e. we are fixing the MGP reference set at the observed locations
# (this may be inefficient in big data settings)
meshout <- spmeshed(y_in, X_in, coords_in,
                    axis_partition=c(4,4),
                    n_samples = mcmc_keep, 
                    n_burn = mcmc_burn, 
                    n_thin = mcmc_thin, 
                    settings = list(forced_grid=FALSE, cache=FALSE),
                    prior=list(phi=c(1,15)),
                    verbose = 0,
                    n_threads = 1)

# test set
coords_out <- coords[is.na(y),]
X_out <- X[is.na(y),]

df_predict <- predict(meshout, newx=X_out, newcoords=coords_out)

y_posterior_predictive_mean <- df_predict$preds_out[,1,] \%>\% 
  apply(1, mean) \%>\% add(ybar)
df_predicted <- df_predict$coords_out \%>\% cbind(y_posterior_predictive_mean)

}
