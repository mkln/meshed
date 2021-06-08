\name{predict.spmeshed}
\alias{predict.spmeshed}
\title{Predictions for spatial MGP models after running MCMC}
\description{This function performs predictions after having run MCMC using \code{spmeshed}.}
\usage{
\method{predict}{spmeshed}(object,
               newx, newcoords, n_threads=4,
               verbose=FALSE, ...)
        
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
While this function can always be used to make predictions, in most cases it is more efficient to just include the prediction locations in the main data as \code{NA} values. The \code{predict} function is only recommended when all 4 of the following are true:

(1) \code{spmeshed} was run with \code{settings$forced_grid=FALSE} and

(2) the prediction locations are uniformly scattered on the domain and

(3) the number of prediction locations is a large portion of the number of observed data points and 

(4) the prediction locations are not on a grid. 

In all other cases the main \code{spmeshed} function is setup to be more efficient in automatically performing predictions during MCMC.
}
\value{
\item{coords_out}{matrix with the prediction location coordinates (order updated after predictions).}

\item{preds_out}{\code{array} of dimension (\eqn{n_{o}, q, m}) where \eqn{n_{o}} is the number of prediction locations, \eqn{q} is the output dimension, \eqn{m} is the number of MCMC samples.}
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
\dontrun{

    new_X <- matrix(rnorm(10), ncol=1)
    new_coords <- cbind(runif(10), runif(10))
    meshed_preds <- predict(meshout, 
                          newx = new_X, 
                          newcoords = new_coords, 
                          verbose=F, num_threads=16)
                          
    y.mesh.out <- mesh_preds$coords_out \%>\% 
      as.data.frame() 
      cbind(data.frame(ymesh_1=mesh_preds$preds_out[,1,] \%>\% apply(1, mean),
                       yhi_1=mesh_preds$preds_out[,1,] \%>\% apply(1, quantile, .975),
                       ylo_1=mesh_preds$preds_out[,1,] \%>\% apply(1, quantile, .025)))
    
}
}
