\name{spmeshed}
\alias{spmeshed}
\title{Meshed GP multivariate spatial regression}
\description{Bayesian linear multivariate spatial regression using latent MGPs. This implements the following model:
\deqn{ y(s) = x(s)^\top \beta + \Lambda w(s) + \epsilon(s), }
where \eqn{y(s)} is a \eqn{q\times 1} vector of outcomes at spatial location \eqn{s}, \eqn{x(s)} is a \eqn{p\times 1} vector of covariates with static coefficients \eqn{\beta}, \eqn{\Lambda} is a \eqn{q \times k} matrix of factor loadings, \eqn{w(s)} is a \eqn{k \times 1} vector which collects the realization of independent Gaussian processes \eqn{w_j \sim spmeshed(0, C_j)} for \eqn{j=1, \dots, k} and where \eqn{C_j(s, s')} is a correlation function. \eqn{s} is a coordinate in space (\eqn{d=2}) or space plus time (\eqn{d=3}). The Meshed GP implemented here associates an axis-parallel tessellation of the domain to a cubic directed acyclic graph (mesh).}
\usage{
spmeshed(y, x, coords, 
  k = NULL,
  axis_partition = NULL, 
  block_size = 30,
  grid_size = NULL,
  grid_custom = NULL,
  n_samples = 1000,
  n_burnin = 100,
  n_thin = 1,
  n_threads = 4,
  print_every = NULL,
  predict_everywhere = F,
  settings    = list(adapting=T, forced_grid=NULL, saving=F),
  prior       = list(beta=NULL, tausq=NULL,
                    phi=NULL, nu = NULL,
                    toplim = NULL, btmlim = NULL, set_unif_bounds=NULL),
  starting    = list(beta=NULL, tausq=NULL, theta=NULL, lambda=NULL, w=NULL, 
                    nu = NULL,
                    mcmcsd=.05, 
                    mcmc_startfrom=0),
  debug       = list(sample_beta=T, sample_tausq=T, 
                    sample_theta=T, sample_w=T, sample_lambda=T,
                    verbose=F, debug=F)
)
}
\arguments{
\item{y}{matrix of multivariate outcomes with \eqn{n} rows and \eqn{q} columns. \code{NA} values are accepted in any combination and will be predicted.}

\item{x}{matrix of covariates with \eqn{n} rows and \eqn{p} columns.}

\item{coords}{matrix of coordinates with \eqn{n} rows and \eqn{d=2} or \eqn{d=3} columns for spatial or spacetime regression, respectively.}

\item{k}{integer \eqn{k\leq q}, number of latent processes to use for the linear model of coregionalization. If unspecified, this is set to \eqn{q}\code{=ncol(y)}.}

\item{axis_partition}{integer vector of size \eqn{d}: number of intervals each coordinate axis is split into}

\item{block_size}{integer approximate size of the blocks after domain partitioning. Only used if \code{axis_partition} is not specified.}

\item{grid_size}{integer vector of size \eqn{d}: number of 'knots' of the reference grid along each axis. 
This grid is then partitioned using either \code{axis_partition} or \code{block_size}.
If unspecified, this is set so that the eventual grid size is close to \eqn{n}.
This parameter is ignored if \code{settings$forced_grid = F} in which case the data are assumed to be on a grid.}

\item{grid_custom}{list with elements \code{grid} and \code{axis_interval_partition}. \code{grid} is a data.frame with the user supplied grid of knots. It is possible to include covariate values for the grid locations as additional columns, as long as their number matches \code{ncol(x)} - this is useful to make raster images of predictions. \code{axis_interval_partition} is the user supplied set of cuts for each coordinate axis. No checks are made on the validity of this grid. This parameter is ignored if \code{settings$forced_grid = F} in which case the data are assumed to be on a grid.}

\item{n_samples}{integer number of MCMC samples at which all the unknowns are stored (including the latent effects).}

\item{n_burnin}{integer number of MCMC samples to discard at the beginning of the chain.}

\item{n_thin}{integer thinning parameter for the MCMC chain. Only the chain of latent effects is thinned to save memory in big data problems. Chains for other unknowns are not thinned and thus will be longer.}

\item{n_threads}{integer number of OpenMP threads. This is ineffective if \code{spmeshed} was not compiled with OpenMP support.}

\item{print_every}{integer number of MCMC iterations separating each info message. The info message includes percentage completion, timing, estimated time remaining, MCMC acceptance, and current values of some unknowns.}

\item{predict_everywhere}{bool used if settings$forced_grid=T. Should predictions be made at the reference grid locations? If not, predictions will be made only at the supplied NA values of Y.}

\item{settings}{list: \code{settings$adapting} turns the adaptation of MCMC on/off, \code{settings$forced_grid} determines whether or not to use the data grid or a forced grid; if unspecified, the function will try to see what the data look like. Note: if \code{forced_grid=F} and \eqn{n} is very large and \eqn{coords} are irregularly spaced, then expect slowdowns in preprocessing and consider using \code{forced_grid=T} instead. \code{settings$saving} will save model data if set to \code{T} and MCMC can be automatically restarted using \code{spmeshed_restart}.}

\item{prior}{list: setup for priors of unknown parameters. There is currently limited functionality here and some parameters are currently ignored. Defaults are: a vague Gaussian for \eqn{\beta}, \eqn{\Lambda_{i,j} \sim N(0,1)}, \eqn{\tau^2_i \sim IG(2,1)}, \eqn{\theta_j \sim IG(2,2)}, all subject to change.}

\item{starting}{list: setup for starting values of unknown parameters. \code{starting$mcmcsd} is the initial standard deviation of proposals. \code{starting$mcmc_startfrom} is input to the adaptive MCMC and can be used to manually restart MCMC without \code{spmeshed_restart}. There is currently limited functionality here and some parameters may be ignored.}

\item{debug}{list: setup for debugging things. Some parts of MCMC can be turned off here.}
}
\value{
\item{coordsdata}{A \code{data.frame} including the original \eqn{n} coordinates plus the \eqn{n_g} knot coordinates if the model was run on a forced grid. The additional column \code{forced_grid} has value 1 if the corresponding coordinate is a knot in the forced grid.}
\item{savedata}{If \code{settings$saving==T}, this stores all model data for use in \code{spmeshed_restart}.}
\item{yhat_mcmc}{A list of length \code{n_samples} whose elements are matrices of size \eqn{(n + n_g) \times q}. Each matrix in the list is a posterior sample of the latent spatial process. \eqn{n_g = 0} if the data grid is being used. Given the possibly large \eqn{n}, only the thinned chain is output for \eqn{w}.}
\item{w_mcmc}{A list of length \code{n_samples} whose elements are matrices of size \eqn{(n + n_g) \times k}. Each matrix in the list is a posterior sample of the latent spatial process. \eqn{n_g = 0} if the data grid is being used. Given the possibly large \eqn{n}, only the thinned chain is output for \eqn{w}.}
\item{beta_mcmc}{An array of size \code{(p, q, n_thin*n_samples)} with the posterior sample for the static regression coefficients \eqn{\beta}. The \eqn{j}th column of each \eqn{p \times q} matrix corresponds to the \eqn{p} linear effects on the \eqn{j}th outcome. The full chain minus burn-in is returned NOT thinned since \code{p} and \code{q} are relatively small.}
\item{tausq_mcmc}{A matrix of size \code{(q, n_thin*n_samples)}. Each row corresponds to the full MCMC chain for the nugget \eqn{\tau^2_j} of the \eqn{j}th outcome in the coregionalization/factor model. The full chain minus burn-in is returned NOT thinned since \code{q} is relatively small.}
\item{theta_mcmc}{An array of size \code{(h, k, n_thin*n_samples)} with the posterior sample for the correlation function parameters \eqn{\theta}. \code{h} is 1 for spatial data (corresponding to the spatial decay of the exponential covariance), 3 for spacetime data (corresponding to temporal decay, spatial decay, and separability -- these are referred to as \eqn{a}, \eqn{c}, and \eqn{\beta} in Gneiting (2002), see \url{https://doi.org/10.1198/016214502760047113}). The full chain minus burn-in is returned NOT thinned since \code{h} and \code{k} are relatively small.}
\item{lambda_mcmc}{An array of size \code{(q, k, n_thin*n_samples)}. Each \eqn{q\times k} matrix is a posterior sample for \eqn{\Lambda} in the coregionalization/factor model. In univariate models, this is usually called \eqn{\sigma}. The full chain minus burn-in is returned NOT thinned since \code{q} and \code{k} are relatively small.}
\item{paramsd}{Cholesky factorization of the proposal covariance for adaptive MCMC, after adaptation.}
\item{mcmc}{Total number of MCMC iterations performed.}
\item{logpost}{Vector of size \code{n_thin*n_samples} storing \eqn{p(y | x, \beta, w, \tau^2)}.}
\item{w_logdens}{Vector of size \code{n_thin*n_samples} storing \eqn{p(w | \theta)}.}
\item{mcmc_time}{Time in seconds taken for MCMC (not including preprocessing).}
\item{proposal_failures}{Number of proposals for \eqn{\theta} that resulted in numerical failures and were automatically rejected. This number should be small.}
}

\references{
  Peruzzi, M., Banerjee, S., and Finley, A.O. (2020)
  Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains. \emph{Journal of the American Statistical Association}, in press. \url{https://doi.org/10.1080/01621459.2020.1833889}
  
  Peruzzi, M., Banerjee, S., Finley, A.O., and Dunson, D.B. (2021)
  Grid-Parametrize-Split (GriPS) for boosting sampling efficiency of spatial regression models. \url{arXiv:TBD}
}

\author{
  Michele Peruzzi \email{michele.peruzzi@duke.edu}, \cr
  Sudipto Banerjee \email{sudipto@ucla.edu}, \cr
  Andrew O. Finley \email{finleya@msu.edu}
}

\examples{
\dontrun{
rm(list=ls())
library(spmeshed)
library(magrittr)
library(dplyr)

set.seed(2020)

SS <- 80 # coord values for jth dimension 
dd <- 2 # spatial dimension
n <- SS^2 # number of locations
p <- 3 # number of covariates

xlocs <- seq(0, 1, length.out=SS)
coords <- expand.grid(list(xlocs, xlocs)) \%>\% 
  as.data.frame() 

phi <- 5
sigmasq <- 2

# cholesky decomp of covariance matrix
LC <- t(chol(exp(- phi * as.matrix(dist(coords)))))
# spatial latent effect factor
WW <- sigmasq^.5 * LC \%*\% rnorm(n)
# nugget
tausq <- .05
# measurement errors
EE <- tausq^.5 * rnorm(n)

# covariates and coefficients
XX <- 1:p \%>\% lapply(function(i) rnorm(n)) \%>\% do.call(cbind, .)
Beta <- matrix(rnorm(p), ncol=1)

# univariate outcome, fully observed
YY_full <- XX \%*\% Beta + WW + EE

# .. introduce some NA values in the outcomes
YY <- YY_full
# leave out everything at some area
YY[(coords$Var1 > .3) & (coords$Var1 < .5) & (coords$Var2 < .85) & (coords$Var2 > .7),] <- NA
# pick some other random locs
YY[sample(1:n, n/5, replace=FALSE), ] <- NA

simdata <- coords \%>\%
  cbind(data.frame(Outcome_full=YY_full, 
                   Outcome_obs = YY)) 

mcmc_keep <- 1000
mcmc_burn <- 200
mcmc_thin <- 5

axis_partition <- c(20, 10) # resulting in blocks of approx size 32

mesh_total_time <- system.time({
  meshout <- spmeshed(YY, XX, coords,
                      axis_partition=axis_partition,
                      n_samples = mcmc_keep, n_burn = mcmc_burn, n_thin = mcmc_thin, 
                      n_threads = 10
  )})

plot_cube <- function(cube_mcmc, q, k, name="Parameter"){
  par(mar=c(2.5,2,1,1), mfrow=c(q,k))
  for(i in 1:q){
    for(j in 1:k){
      cube_mcmc[i, j,] \%>\% plot(type='l', main="{name} {i}, {j}" \%>\% glue::glue())
    }
  }
}
# chain plots
plot_cube(meshout$theta_mcmc, 1, 1, "theta")
plot_cube(meshout$lambda_mcmc, 1, 1, "Lambda") # = sigma
plot_cube(meshout$beta_mcmc, p, 1, "Beta")

# posterior means
meshout$tausq_mcmc \%>\% apply(1, mean)
meshout$lambda_mcmc \%>\% apply(1:2, mean) # sigma
meshout$beta_mcmc \%>\% apply(1:2, mean)
meshout$theta_mcmc \%>\% apply(1:2, mean)

# process means
wmesh <- data.frame(wmesh_1 = meshout$w_mcmc \%>\% summary_list_mean())
# predictions
ymesh <- data.frame(ymesh_1 = meshout$yhat_mcmc \%>\% summary_list_mean())

mesh_df <- 
  meshout$data \%>\% 
  cbind(ymesh) \%>\%
  # in this case all the coords are at forced_grid==0
  # see examples with forced grid to see how this is used
  dplyr::filter(forced_grid==0) \%>\% 
  dplyr::select(-forced_grid)

results <- simdata \%>\% left_join(mesh_df)

# prediction rmse, out of sample
results \%>\% filter(!complete.cases(Outcome_obs)) \%>\% 
  with((Outcome_full - ymesh_1)^2) \%>\% mean() \%>\% sqrt()

# mapping predictions
library(ggplot2)

results \%>\% 
    tidyr::gather(Variable, Value, -Var1, -Var2) \%>\%
    ggplot(aes(Var1, Var2, fill=Value)) +
    geom_raster() +
    facet_wrap(Variable ~ ., ncol= 2) +
    scale_fill_viridis_c()
}
}