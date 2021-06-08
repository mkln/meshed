\name{spmeshed.map}
\alias{spmeshed.map}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Meshed GP spatial regression and predictions of univariate outcomes via Maximum a Posteriori.
}
\description{
This function fits Bayesian univariate spatial regressions with latent MGPs via MAP.
}
\usage{
spmeshed.map(y, x, coords, 
       family = "gaussian",
       axis_partition = NULL, 
       block_size = 30,
       grid_size = NULL,
       grid_custom = NULL,
       pars = list(sigmasq=NULL, phi=NULL, tausq=NULL),
       maxit = 1000,
       n_threads = 4,
       print_every = NULL,
       predict_everywhere = FALSE,
       settings = list(forced_grid=NULL, cache=NULL),
       debug = list(map_beta=TRUE, map_w=TRUE,
                    verbose=FALSE, debug=FALSE))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{y}{
matrix of outcomes with \eqn{n} rows and \eqn{1} columns. Each row of \code{y} corresponds to a row of \code{coords}. \code{NA} values are accepted in any combination and will be predicted via MAP
}
  \item{x}{
  matrix of covariates with \eqn{n} rows and \eqn{p} columns.
}
  \item{coords}{
  matrix of coordinates with \eqn{n} rows and \eqn{d=2} columns for spatial regression.
}
\item{family}{
  same as in \code{\link{spmeshed}}
}
  \item{axis_partition}{
  same as in \code{\link{spmeshed}}.
}
  \item{block_size}{
  same as in \code{\link{spmeshed}}.
}
  \item{grid_size}{
  same as in \code{\link{spmeshed}}.
}
  \item{grid_custom}{
  same as in \code{\link{spmeshed}}.
}
  \item{pars}{
  list with elements named after the fixed values of the hyperparameters \eqn{\sigma^2, \phi} and \eqn{\tau^2}. Each of the list elements is a vector enumerating all values to be used. The function will then then compute the MAP for \eqn{w} and/or \eqn{\beta} for all combinations of the supplied values.
}
  \item{maxit}{
  integer maximum number of iterations to attempt finding the MAP estimates for each combination of values in \code{pars}.
}
  \item{n_threads}{
  integer number of OpenMP threads. This is ineffective if \code{meshed} was not compiled with OpenMP support.
}
  \item{print_every}{
  same as in \code{\link{spmeshed}}
}
  \item{predict_everywhere}{
  same as in \code{\link{spmeshed}}
}
  
  \item{settings}{
  same as in \code{\link{spmeshed}}; here, only \code{settings$forced_grid} and \code{settings$cache} can be used.
}
  \item{debug}{
  list: setup for debugging things. Some parts of the algorithm can be turned off here, plus more options for intermediate messages.
}

}
\details{
This function targets the following model:
\deqn{ y(s) \sim F( x(s)^\top \beta + w(s), \tau^2 ) }
where \eqn{y(s)} is the scalar outcome at spatial location \eqn{s}, \eqn{x(s)} is a \eqn{p}-dimensional vector of covariates with static coefficients \eqn{\beta}, \eqn{w(s)} collects the realization of a Meshed Gaussian processes \eqn{w \sim MGP(0, \sigma^2 C)} for \eqn{j=1, \dots, k} and where \eqn{C(s, s')} is an exponential correlation function. \eqn{\tau^2} is a disperion parameter for distribution \eqn{F}. \eqn{s} is a coordinate in space (\eqn{d=2}). The Meshed GP implemented here associates an axis-parallel tessellation of the domain to a cubic directed acyclic graph (mesh). As a special case, if \eqn{F} is a Gaussian distribution (\code{family="gaussian"}) then the model becomes
\deqn{ y(s) = x(s)^\top \beta + w(s) + \epsilon(s) }
where \eqn{\epsilon(s) \sim N(0, \tau^2)}. 
}
\value{
\item{coordsdata}{same as in \code{\link{spmeshed}}.}
\item{pardf}{data.frame with \eqn{M} rows, storing all combinations of hyperparameter values as input in \code{pars}.}
\item{yhat_map}{matrix with \eqn{n} rows and \eqn{M} columns storing the predicted values of the outcome at the coordinates in \code{coordsdata}.}
\item{beta_map}{matrix with \eqn{p} rows and \eqn{M} columns storing the MAP estimates for \eqn{\beta} for each combination of hyperparameters.}
\item{w_map}{matrix with \eqn{n} rows and \eqn{M} columns storing the MAP estimates for the latent spatial effects \eqn{w} for each combination of hyperparameter values.}
\item{lp_map}{matrix with \eqn{n} rows and \eqn{M} columns storing the MAP estimates for the linear predictor \eqn{X\beta + w} for each combination of hyperparameter values.}
}
\references{
Peruzzi, M., Banerjee, S., and Finley, A.O. (2020)
  Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains. \emph{Journal of the American Statistical Association}, in press. \doi{10.1080/01621459.2020.1833889}
}
\author{
Michele Peruzzi <michele.peruzzi@duke.edu>
}

\examples{


}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
