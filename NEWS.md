#  - `meshed` version 0.2.0

 * `spmeshed` can now use less memory by only storing essentials. 
 * Added Negative binomial as option for outcome family
 * Gneiting spacetime covariance now works with Matern smoothness 0.5, 1.5, or 2.5 
 * Prior for the temporal decay in the Gneiting spacetime covariance can be specified separately from phi
 * [to do] Added postprocessing for dealing with low-memory situations
 * [to do] Improved preprocessing for sampling from rmeshedgp
 * Beta and Lambda are updated as a block in most cases
 * Fixes in posterior sampling with non-Gaussian outcomes
 * Minor changes and cleanup

#  6 Oct 21 - `meshed` version 0.1.4

 * version bump after fixing vignette bugs for CRAN

# 23 Sep 21 - `meshed` version 0.1.3

 * better memory management when exiting MCMC
 * fixed `M_PI` for upcoming Rcpp updates
 * other minor bugfixes

# 21 Jun 21 - `meshed` version 0.1.2

 * fixed a couple of typos in docs
 * more OMP options in prior sampling
 * fix compile error on Solaris

# 10 Jun 21 - `meshed` version 0.1.1

Addressing comments from CRAN:

 * fixed `par()` setting on vignette
 * set `verbose=0` in all vignettes and examples

# 9 Jun 21 - `meshed` version 0.1.0

First version submitted to CRAN.