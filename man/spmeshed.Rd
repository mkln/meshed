\name{spmeshed}
\alias{spmeshed}
\title{Posterior sampling for models based on MGPs}
\description{Fits Bayesian multivariate spatial or spatiotemporal regression models with latent MGPs via Markov chain Monte Carlo.}
\usage{
spmeshed(y, x, coords, k=NULL,
       family = "gaussian",
       axis_partition = NULL, 
       block_size = 30,
       grid_size = NULL,
       grid_custom = NULL,
       n_samples = 1000,
       n_burnin = 100,
       n_thin = 1,
       n_threads = 4,
       verbose = 0,
       predict_everywhere = FALSE,
       settings = list(adapting=TRUE, forced_grid=NULL, 
                          cache=NULL, ps=TRUE, saving=TRUE, low_mem=FALSE, hmc=4),
       prior = list(beta=NULL, tausq=NULL, sigmasq = NULL,
                          phi=NULL, a=NULL, nu = NULL,
                          toplim = NULL, btmlim = NULL, set_unif_bounds=NULL),
       starting = list(beta=NULL, tausq=NULL, theta=NULL, lambda=NULL, v=NULL, 
                       a=NULL, nu = NULL,
                       mcmcsd=.05, 
                       mcmc_startfrom=0),
       debug = list(sample_beta=TRUE, sample_tausq=TRUE, 
                    sample_theta=TRUE, sample_w=TRUE, sample_lambda=TRUE,
                    verbose=FALSE, debug=FALSE),
       indpart=FALSE
)
}

\arguments{
\item{y}{matrix of multivariate outcomes with \eqn{n} rows and \eqn{q} columns. Each row of \code{y} corresponds to a row of \code{coords}. \code{NA} values are accepted in any combination and will be predicted via MCMC.}

\item{x}{matrix of covariates with \eqn{n} rows and \eqn{p} columns.}

\item{coords}{matrix of coordinates with \eqn{n} rows and \eqn{d=2} or \eqn{d=3} columns for spatial or spacetime regression, respectively.}

\item{k}{integer \eqn{k\leq q}, number of latent processes to use for the linear model of coregionalization. If unspecified, this is set to \eqn{q}\code{=ncol(y)}.}

\item{family}{a vector with length \eqn{1} or \eqn{q} whose elements corresponds to the data types of columns of \code{y}. Available choices are \code{gaussian}, \code{poisson}, \code{binomial}, \code{beta} for outcomes that are continuous, count, binary, or \eqn{(0,1)} proportions.}

\item{axis_partition}{integer vector of size \eqn{d}: number of intervals each coordinate axis is split into}

\item{block_size}{integer approximate size of the blocks after domain partitioning. Only used if \code{axis_partition} is not specified.}

\item{grid_size}{integer vector of size \eqn{d}: number of 'knots' of the reference grid along each axis. 
This grid is then partitioned using either \code{axis_partition} or \code{block_size}.
If unspecified, this is set so that the eventual grid size is close to \eqn{n}.
This parameter is ignored if \code{settings$forced_grid=FALSE} in which case the data are assumed to be on a grid.}

\item{grid_custom}{list with elements \code{grid} and \code{axis_interval_partition}. \code{grid} is a data.frame with the user supplied grid of knots. It is possible to include covariate values for the grid locations as additional columns, as long as their number matches \code{ncol(x)} - this is useful to make raster images of predictions. \code{axis_interval_partition} is the user supplied set of cuts for each coordinate axis (Note: these are the actual cutpoints along the axes, not the number of cuts). If left empty, \code{axis_partition} will be used to partition the custom grid. No checks are made on the validity of this grid. This parameter is ignored if \code{settings$forced_grid=FALSE} in which case the data are assumed to be on a grid.}

\item{n_samples}{integer number of MCMC samples at which all the unknowns are stored (including the latent effects).}

\item{n_burnin}{integer number of MCMC samples to discard at the beginning of the chain.}

\item{n_thin}{integer thinning parameter for the MCMC chain. Only the chain of latent effects (\eqn{w}) is thinned to save memory in big data problems. Chains for other unknowns are not thinned and thus will be of length \code{n_thin * n_samples}.}

\item{n_threads}{integer number of OpenMP threads. This is ineffective if \code{meshed} was not compiled with OpenMP support.}

\item{verbose}{integer. If \code{verbose<=20}, then this is the number of times a message is displayed during MCMC. If \code{verbose>20}, then this is the number of MCMC iterations to wait until the next message update. If \code{verbose=Inf}, then a message will be printed at each MCMC iteration.}

\item{predict_everywhere}{bool used if settings$forced_grid=T. Should predictions be made at the reference grid locations? If not, predictions will be made only at the supplied NA values of Y.}

\item{settings}{list: \code{settings$adapting} turns the adaptation of MCMC on/off, \code{settings$forced_grid} determines whether or not to use the data grid or a forced grid; if unspecified, the function will try to see what the data look like. Note: if \code{forced_grid=FALSE} and \eqn{n} is very large and \eqn{coords} are irregularly spaced, then expect slowdowns in preprocessing and consider using \code{forced_grid=TRUE} instead. \code{settings$saving} will save model data if set to \code{TRUE}. \code{settings$low_mem} will only save \code{beta_mcmc}, \code{lambda_mcmc}, \code{v_mcmc}, \code{tausq_mcmc} (and not \code{w_mcmc} and \code{lp_mcmc}, which can be recovered from the others), thereby using less memory. All fitted predictions remain available in \code{yhat_mcmc} for convenience. \code{settings$ps} (default \code{TRUE}) determines whether to use the PS parametrization (Peruzzi et al 2021). \code{settings$hmc}, used if any outcome is not Gaussian, (1: MALA, 2: NUTS, 3: RM-MALA, 4: Simplified manifold preconditioning (default))}

\item{prior}{list: setup for priors of unknown parameters. \code{prior$phi} needs to be specified as the support of the Uniform prior for \eqn{\phi}. There is currently limited functionality here and some inputs are currently ignored. Defaults are: a vague Gaussian for \eqn{\beta}, \eqn{\tau^2_i \sim IG(2,1)}, \eqn{\theta_j \sim IG(2,2)}, all subject to change.}

\item{starting}{list: setup for starting values of unknown parameters. \code{starting$mcmcsd} is the initial standard deviation of proposals. \code{starting$mcmc_startfrom} is input to the adaptive MCMC and can be used to manually restart MCMC. There is currently limited functionality here and some parameters may be ignored.}

\item{debug}{list: setup for debugging things. Some parts of MCMC can be turned off here.}
\item{indpart}{bool defaults to \code{FALSE}. If \code{TRUE}, this computes an independent partition model.}
}

\details{
This function targets the following model:
\deqn{ y(s) = x(s)^\top \beta + \Lambda v(s) + \epsilon(s), }
where \eqn{y(s)} is a \eqn{q}-dimensional vector of outcomes at spatial location \eqn{s}, \eqn{x(s)} is a \eqn{p}-dimensional vector of covariates with static coefficients \eqn{\beta}, \eqn{\Lambda} is a matrix of factor loadings of size \eqn{(q, k)}, \eqn{v(s)} is a \eqn{k}-dimensional vector which collects the realization of independent Gaussian processes \eqn{v_j \sim spmeshed(0, C_j)} for \eqn{j=1, \dots, k} and where \eqn{C_j(s, s')} is a correlation function. \eqn{s} is a coordinate in space (\eqn{d=2}) or space plus time (\eqn{d=3}). The Meshed GP implemented here associates an axis-parallel tessellation of the domain to a cubic directed acyclic graph (mesh).
}

\value{
\item{coordsdata}{data.frame including the original \eqn{n} coordinates plus the \eqn{n_g} knot coordinates if the model was run on a forced grid. The additional column \code{forced_grid} has value 1 if the corresponding coordinate is a knot in the forced grid. See examples.}
\item{savedata}{Available if \code{settings$saving==TRUE}. Needed for making predictions using \code{predict()} after MCMC. Note: \code{NA} values of the output are automatically and more efficiently predicted when running \code{spmeshed}.}
\item{yhat_mcmc}{list of length \code{n_samples} whose elements are matrices with \eqn{n + n_g} rows and \eqn{q} columns. Each matrix in the list is a posterior predictive sample of the latent spatial process. \eqn{n_g = 0} if the data grid is being used. Given the possibly large \eqn{n}, only the thinned chain is output for \eqn{y}.}
\item{v_mcmc}{list of length \code{n_samples} whose elements are matrices with \eqn{n + n_g} rows and \eqn{k} columns. Each matrix in the list is a posterior sample of the \eqn{k} latent spatial process. \eqn{n_g = 0} if the data grid is being used. Given the possibly large \eqn{n}, only the thinned chain is output for \eqn{v}.}
\item{w_mcmc}{list of length \code{n_samples} whose elements are matrices with \eqn{n + n_g} rows and \eqn{q} columns. Each matrix in the list is a posterior sample of \eqn{w = \Lambda v}. \eqn{n_g = 0} if the data grid is being used. Given the possibly large \eqn{n}, only the thinned chain is output for \eqn{w}.}
\item{lp_mcmc}{list of length \code{n_samples} whose elements are matrices with \eqn{n + n_g} rows and \eqn{q} columns. Each matrix in the list is a posterior sample of the linear predictor \eqn{X\beta + \Lambda v}. \eqn{n_g = 0} if the data grid is being used. Given the possibly large \eqn{n}, only the thinned chain is output for \eqn{w}.}
\item{beta_mcmc}{array of size \code{(p, q, n_thin*n_samples)} with the posterior sample for the static regression coefficients \eqn{\beta}. The \eqn{j}th column of each matrix (\eqn{p} rows and \eqn{q} columns) corresponds to the \eqn{p} linear effects on the \eqn{j}th outcome. The full chain minus burn-in is returned NOT thinned since \code{p} and \code{q} are relatively small.}
\item{tausq_mcmc}{matrix of size \code{(q, n_thin*n_samples)}. Each row corresponds to the full MCMC chain for the nugget \eqn{\tau^2_j} of the \eqn{j}th outcome in the coregionalization/factor model. The full chain minus burn-in is returned NOT thinned since \code{q} is relatively small.}
\item{theta_mcmc}{array of size \code{(h, k, n_thin*n_samples)} with the posterior sample for the correlation function parameters \eqn{\theta}. \code{h} is 2 for spatial data (corresponding to the spatial decay of the exponential covariance (\eqn{\phi_i, i=1, \dots, k}), and the variance \eqn{\sigma^2_i, i=1, \dots, k}), 4 for spacetime data (corresponding to temporal decay, spatial decay, and separability -- these are referred to as \eqn{a_i}, \eqn{\phi_i}, and \eqn{\beta_i, i=1, \dots, k}, in Gneiting (2002), see \doi{10.1198/016214502760047113}, plus the variance \eqn{\sigma^2, i=1, \dots, k}). The full chain minus burn-in is returned NOT thinned since \code{h} and \code{k} are relatively small. If \code{settings$ps=TRUE}, the MCMC output for \eqn{\sigma^2_i} (last row of \code{theta_mcmc}) should be discarded and \eqn{\Lambda} used instead.}
\item{lambda_mcmc}{array of size \code{(q, k, n_thin*n_samples)}. Each matrix (of size \eqn{(q,k)}) is a posterior sample for \eqn{\Lambda} in the coregionalization/factor model. In univariate models, this is usually called \eqn{\sigma}. The full chain minus burn-in is returned NOT thinned since \code{q} and \code{k} are relatively small.}
\item{paramsd}{Cholesky factorization of the proposal covariance for adaptive MCMC, after adaptation.}
\item{mcmc}{Total number of MCMC iterations performed.}
\item{mcmc_time}{Time in seconds taken for MCMC (not including preprocessing).}
}

\references{
  Peruzzi, M., Banerjee, S., and Finley, A.O. (2022)
  Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains. \emph{Journal of the American Statistical Association}, 117(538):969-982. \doi{10.1080/01621459.2020.1833889}
  
  Peruzzi, M., Banerjee, S., Dunson, D.B., and Finley, A.O.  (2021)
  Grid-Parametrize-Split (GriPS) for Improved Scalable Inference in Spatial Big Data Analysis. \url{https://arxiv.org/abs/2101.03579}
  
  Peruzzi, M. and Dunson, D.B. (2022)
  Spatial meshing for general Bayesian multivariate models. \url{https://arxiv.org/abs/2201.10080}
}

\author{
  Michele Peruzzi \email{michele.peruzzi@duke.edu}
}

\examples{

# toy example with tiny dataset and short MCMC
# on a univariate outcome
library(magrittr)
library(dplyr)
library(ggplot2)
library(meshed)

set.seed(2021)

SS <- 12
n <- SS^2 # total n. locations, including missing ones

coords <- expand.grid(xx <- seq(0,1,length.out=SS), xx) \%>\% 
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

meshout <- spmeshed(y-ybar, X, coords,
                    axis_partition=c(4,4),
                    n_samples = mcmc_keep, 
                    n_burn = mcmc_burn, 
                    n_thin = mcmc_thin, 
                    prior=list(phi=c(1,15)),
                    verbose = 0,
                    n_threads = 1)

# posterior means
best_post_mean <- meshout$beta_mcmc \%>\% apply(1:2, mean)
  
# process means
wmesh <- data.frame(w_mgp = meshout$w_mcmc \%>\% summary_list_mean())
# predictions
ymesh <- data.frame(y_mgp = meshout$yhat_mcmc \%>\% summary_list_mean())

outdf <- 
  meshout$coordsdata \%>\% 
  cbind(ymesh, wmesh) 

# plot predictions
pred_plot <- outdf \%>\% 
  ggplot(aes(Var1, Var2, color=y_mgp)) +
  geom_point() +
  scale_color_viridis_c()

# plot latent process
latent_plot <- outdf \%>\% 
  ggplot(aes(Var1, Var2, color=w_mgp)) +
  geom_point() + 
  scale_color_viridis_c()

# estimation of regression coefficients
plot(density(meshout$beta_mcmc[1,1,]))
abline(v=B[1], col="red")


}