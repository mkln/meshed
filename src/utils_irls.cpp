#include "utils_irls.h"


using namespace std;

arma::vec gaussian_linkinv(const arma::vec& eta){
  return eta;
}

arma::vec gaussian_mueta(const arma::vec& eta){
  return arma::ones(eta.n_elem);
}

arma::vec gaussian_variance(const arma::vec& mu){
  return arma::ones(mu.n_elem);
}

arma::vec binomial_linkinv(const arma::vec& eta){
  return 1.0/(1.0 + exp(-eta));
}

arma::vec binomial_mueta(const arma::vec& eta){
  arma::vec expeta   = exp(eta);
  arma::vec opexpeta = 1.0 + expeta;
  return expeta/(opexpeta%opexpeta);
}

arma::vec binomial_variance(const arma::vec& mu){
  return mu%(1.0-mu);
}

arma::vec negbinomial_linkinv(const arma::vec& eta){
  return exp(eta);
}

arma::vec negbinomial_mueta(const arma::vec& eta){
  return exp(eta);
}

arma::vec negbinomial_variance(const arma::vec& mu, const double& tausq){
  return mu + pow(mu, 2) * tausq;
}


arma::vec poisson_linkinv(const arma::vec& eta){
  return exp(eta);
}

arma::vec poisson_mueta(const arma::vec& eta){
  return exp(eta);
}

arma::vec poisson_variance(const arma::vec& mu){
  return mu;
}

arma::vec irls_bayes_cpp(const arma::vec& y, 
                         const arma::mat& X, 
                         const arma::vec& offset,
                         const arma::mat& Vi,
                         double scaling,
                         std::string family, 
                         int maxit,
                         double tol){
  
  arma::vec x = arma::zeros(X.n_cols);
  
  int fam_id = family=="gaussian" ? 1 : (family=="binomial" ? 2 : (family=="poisson" ? 3 : 0));
  if(fam_id==0){
    x.fill(arma::datum::nan);
    return x;
  }
  
  auto linkinv = [&](const arma::vec& arg){
    switch (fam_id) 
    {
    case 1:
      return gaussian_linkinv(arg);
    case 2:
      return binomial_linkinv(arg);
    case 3:
      return poisson_linkinv(arg);
    }
    return arg;
  };
  
  auto mueta = [&](const arma::vec& arg){
    switch (fam_id) 
    {
    case 1:
      return gaussian_mueta(arg);
    case 2:
      return binomial_mueta(arg);
    case 3:
      return poisson_mueta(arg);
    }
    return arg;
  };
  
  auto variance = [&](const arma::vec& arg){
    switch (fam_id) 
    {
    case 1:
      return gaussian_variance(arg);
    case 2:
      return binomial_variance(arg);
    case 3:
      return poisson_variance(arg);
    }
    return arg;
  };
  
  for(int j=0; j<maxit; j++){
    arma::vec eta    = X * x + offset;
    arma::vec g      = linkinv(eta);
    arma::vec gprime = mueta(eta);
    arma::vec z      = eta - offset + (y - g)/gprime;
    arma::vec ww     = (gprime % gprime) / variance(g);
    arma::vec xold   = x;
    
    arma::mat Xstar = X.t() * arma::diagmat(ww);
    x = arma::solve(Xstar * X/scaling + Vi, Xstar * z/scaling, arma::solve_opts::likely_sympd);
    
    
    if(arma::norm(x - xold) < tol){
      break;
    }
  }
  
  return x;
}


arma::vec irls_step(const arma::vec& start_x,
                    const arma::vec& y, 
                    const arma::mat& X, 
                    const arma::vec& offset,
                    const arma::mat& Vi,
                    int fam_id,
                    double scaling){
  
  auto linkinv = [&](const arma::vec& arg){
    switch (fam_id) 
    {
    case 0:
      return gaussian_linkinv(arg);
    case 2:
      return binomial_linkinv(arg);
    case 1:
      return poisson_linkinv(arg);
    case 4:
      return negbinomial_linkinv(arg);
    }
    return arg;
  };
  
  auto mueta = [&](const arma::vec& arg){
    switch (fam_id) 
    {
    case 0:
      return gaussian_mueta(arg);
    case 2:
      return binomial_mueta(arg);
    case 1:
      return poisson_mueta(arg);
    case 4:
      return negbinomial_mueta(arg);
    }
    return arg;
  };
  
  auto variance = [&](const arma::vec& arg){
    switch (fam_id) 
    {
    case 0:
      return gaussian_variance(arg);
    case 2:
      return binomial_variance(arg);
    case 1:
      return poisson_variance(arg);
    case 4:
      return negbinomial_variance(arg, scaling);
    }
    return arg;
  };
  
  arma::vec eta    = X * start_x + offset;
  arma::vec g      = linkinv(eta);
  arma::vec gprime = mueta(eta);
  arma::vec z      = eta - offset + (y - g)/gprime;
  arma::vec ww     = (gprime % gprime) / variance(g);
  
  arma::mat Xstar = X.t() * arma::diagmat(ww);
  arma::vec stepped = arma::solve(Xstar * X + Vi, Xstar * z, arma::solve_opts::likely_sympd);
  if(stepped.has_nan()){
    return start_x;
  } else {
    return stepped;
  }
}

