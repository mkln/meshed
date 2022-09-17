#ifndef UTILS_DENS_GRAD 
#define UTILS_DENS_GRAD

#include <RcppArmadillo.h>

using namespace std;

const double TOL_LOG_LOW=exp(-10);
const double TOL_HIGH=exp(10);
const double TOL_LOG_HIGH=10;

inline double gaussian_logdensity(const double& x, const double& sigsq){
  return -0.5*log(2.0 * M_PI * sigsq) -0.5/sigsq * x*x;
}

inline double gaussian_loggradient(const double& x, const double& sigsq){
  // derivative wrt mean parameter
  return x/sigsq;
}

inline double poisson_logpmf(const double& x, double lambda){
  if(lambda < TOL_LOG_LOW){
    lambda = TOL_LOG_LOW;
  } else {
    if(lambda > TOL_HIGH){
      lambda = TOL_HIGH;
    }
  }
  return x * log(lambda) - lambda - lgamma(x+1);
}

inline double poisson_loggradient(const double& y, const double& offset, const double& w){
  // llik: y * log(lambda) - lambda - lgamma(y+1);
  // lambda = exp(o + w);
  // llik: y * (o + w) - exp(o+w);
  // grad: y - exp(o+w)
  if(offset + w > TOL_LOG_HIGH){
    return y - TOL_HIGH;
  }
  return y - exp(offset + w);
}

inline double poisson_neghess_mult_sqrt(const double& mu){
  return pow(mu, 0.5);
}

inline double bernoulli_logpmf(const double& x, double p){
  if(p > 1-TOL_LOG_LOW){
    p = 1-TOL_LOG_LOW;
  } else {
    if(p < TOL_LOG_LOW){
      p = TOL_LOG_LOW;
    }
  }
  return x * log(p) + (1-x) * log(1-p);
}

inline double bernoulli_loggradient(const double& y, const double& offset, const double& w){
  // llik: (y-1) * (o+w) - log{1+exp(-o-w)}
  // grad: y-1 + exp(-o-w)/(1+exp(-o-w))
  return y-1 + 1.0/(1.0+exp(offset+w));
}

inline double bernoulli_neghess_mult_sqrt(const double& exij){
  double opexij = (1.0 + exij);
  return pow(exij / (opexij*opexij), 0.5);
}

inline double betareg_logdens(const double& y, const double& mu, double phi){
  // ferrari & cribari-neto A3
  // using logistic link

  double muphi = mu*phi;
  return R::lgammafn(phi) - R::lgammafn(muphi) - R::lgammafn(phi - muphi) +
    (muphi - 1.0) * log(y) + 
    (phi - muphi - 1.0) * log(1.0-y);
  
}

inline double betareg_loggradient(const double& ystar, const double& mu, const double& phi){
  // ferrari & cribari-neto A3
  // using logistic link
  double muphi = mu*phi;
  double oneminusmu = 1.0-mu;
  //double ystar = log(y/(1.0-y));
  double mustar = R::digamma(muphi) - R::digamma(phi - muphi);
  
  return phi * (ystar - mustar) * mu * oneminusmu;
}

inline double betareg_neghess_mult_sqrt(const double& sigmoid, const double& tausq){
  double tausq2 = tausq * tausq;
  double multout = pow(
    //- 
      1.0/tausq2 * (R::trigamma( sigmoid / tausq ) + 
      R::trigamma( (1.0-sigmoid) / tausq ) ) * pow(sigmoid * (1.0 - sigmoid), 2.0), 
                     .5);  // notation of 
      //Rcpp::Rcout << sigmoid << " " << multout << " " << tausq << endl;
  return multout;
}

inline double negbin_logdens(const double& y, double mu, double logmu, double alpha){
  // Cameron & Trivedi 2013 p. 81
  if(mu > TOL_HIGH){
    mu = TOL_HIGH;
    logmu = TOL_LOG_HIGH;
  }
  if(alpha < TOL_LOG_LOW){
    // reverts to poisson
    return y * logmu - mu - lgamma(y+1);
  } 
  double sumj = 0;
  for(int j=0; j<y; j++){
    sumj += log(j + 1.0/alpha);
  }
  double p = 1.0 + alpha * mu;
  return sumj - lgamma(y+1) - (y+1.0/alpha) * log(p) + y * (log(alpha) + logmu);
}

inline double negbin_loggradient(const double& y, double mu, const double& alpha){
  if(mu > TOL_HIGH){
    mu = TOL_HIGH;
  }
  return ((y-mu) / (1.0 + alpha * mu));
}

inline double negbin_neghess_mult_sqrt(const double& y, double logmu, const double& alpha){
  double mu = exp(logmu);
  if(mu > TOL_HIGH){
    mu = TOL_HIGH;
    logmu = TOL_LOG_HIGH;
  }
  double onealphamu = (1.0 + alpha*mu);
  //double result = pow(mu * (1 + alpha * y)/(onealphamu * onealphamu), .5);
  double result = pow(mu/onealphamu, .5);
  return result;
}


inline double get_mult(const double& y, const double& tausq, const double& offset, 
                       const double& xij, const int& family){
  // if m is the output from this function, then
  // m^2 X'X is the negative hessian of a glm model in which X*x is the linear term
  double mult=1;
  if(family == 0){  // family=="gaussian"
    mult = pow(tausq, -0.5);
  } else if (family == 1){
    double mu = exp(offset + xij);
    mult = poisson_neghess_mult_sqrt(mu);
  } else if (family == 2){
    double exij = exp(- offset - xij);
    mult = bernoulli_neghess_mult_sqrt(exij);
  } else if (family == 3){
    double sigmoid = 1.0/(1.0 + exp(-offset - xij));
    mult = betareg_neghess_mult_sqrt(sigmoid, tausq);
  } else if(family == 4){
    double logmu = offset + xij;
    double alpha = tausq;
    mult = negbin_neghess_mult_sqrt(y, logmu, alpha);
  }
  return mult;
}

inline arma::vec get_likdens_likgrad(double& loglike,
                              const double& y, const double& ystar, const double& tausq, 
                              const double& offset, const double& xij, const int& family,
                              bool do_grad=true){
  
  arma::vec gradloc;
  if(family == 0){  // family=="gaussian"
    double y_minus_mean = y - offset - xij;
    loglike += gaussian_logdensity(y_minus_mean, tausq);
    if(do_grad){ gradloc = gaussian_loggradient(y_minus_mean, tausq); }
  } else if(family == 1){ //if(family == "poisson"){
    double lambda = exp(offset + xij);//xz);//x(i));
    loglike += poisson_logpmf(y, lambda);
    if(do_grad){ gradloc = poisson_loggradient(y, offset, xij); } //xz) * z(i);
  } else if(family == 2){ //if(family == "binomial"){
    double exij = exp(-offset - xij);
    double opexij = (1.0 + exij);
    double sigmoid = 1.0/opexij;//xz ));
    loglike += bernoulli_logpmf(y, sigmoid);
    if(do_grad){ gradloc = bernoulli_loggradient(y, offset, xij); } //xz) * z(i);
  } else if(family == 3){
    double sigmoid = 1.0/(1.0 + exp(-offset - xij));
    loglike += betareg_logdens(y, sigmoid, 1.0/tausq);
    if(do_grad){ gradloc = betareg_loggradient(ystar, sigmoid, 1.0/tausq); }
  } else if(family == 4){
    double logmu = offset + xij;
    double mu = exp(logmu);
    double alpha = tausq;
    double addloglike = negbin_logdens(y, mu, logmu, alpha);
    loglike += addloglike;
    if(do_grad){ gradloc = negbin_loggradient(y, mu, alpha); }
  }
  return gradloc;
}

#endif
