\name{meshedgp_restart}
\alias{meshedgp_restart}
\title{Restarting MCMC from previous chain.}
\description{This function restarts MCMC from the last state of a previously computed chain as output from \code{meshedgp}. Output from this function can be input again to continue running MCMC. This function will not run MCMC for a burn-in period.}
\usage{
meshedgp_restart(meshedgp_output, 
             n_samples = 1000,
             n_thin = NULL,
             print_every = NULL)
)
}
\arguments{
\item{meshedgp_output}{A list object as output from \code{meshedgp}. No checks are performed on the contents of the list.}

\item{n_samples}{integer number of MCMC samples at which all the unknowns are stored (including the latent effects).}

\item{n_thin}{integer thinning parameter for the MCMC chain. Only the chain of latent effects is thinned to save memory in big data problems. Chains for other unknowns are not thinned and thus will be longer.}

\item{print_every}{integer number of MCMC iterations separating each info message. The info message includes percentage completion, timing, estimated time remaining, MCMC acceptance, and current values of some unknowns.}
}
\value{
The output is the same as the \code{meshedgp} function.
}

\examples{
\dontrun{
mgp_out2 <- meshedgp_restart(mgp_out1, 1000, 5, 100)
}
}