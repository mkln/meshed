#ifndef UTILS_DENS_GRAD 
#define UTILS_DENS_GRAD

#include <RcppArmadillo.h>

using namespace std;

inline double gaussian_logdensity(const double& x, const double& sigsq){
  return -0.5*log(2.0 * M_PI * sigsq) -0.5/sigsq * x*x;
}

inline double gaussian_loggradient(const double& x, const double& sigsq){
  // derivative wrt mean parameter
  return 1.0/sigsq * x;
}

inline double poisson_logpmf(const double& x, const double& lambda){
  return x * log(lambda) - lambda - lgamma(x+1);
}

inline double poisson_loggradient(const double& y, const double& offset, const double& w){
  // llik: y * log(lambda) - lambda - lgamma(y+1);
  // lambda = exp(o + w);
  // llik: y * (o + w) - exp(o+w);
  // grad: y - exp(o+w)
  return y - exp(offset + w);
}

inline double bernoulli_logpmf(const double& x, const double& p){
  return x * log(p) + (1-x) * log(1-p);
}

inline double bernoulli_loggradient(const double& y, const double& offset, const double& w){
  // llik: (y-1) * (o+w) - log{1+exp(-o-w)}
  // grad: y-1 + exp(-o-w)/(1+exp(-o-w))
  return y-1 + 1.0/(1.0+exp(offset+w));
}

#endif