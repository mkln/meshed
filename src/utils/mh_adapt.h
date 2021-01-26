#include <RcppArmadillo.h>


const double EPS = 0.1; // 0.01
const double tau_accept = 0.234; // target
const double g_exp = .01;
const int g0 = 500; // iterations before starting adaptation
const double rho_max = 2;
const double rho_min = 5;


inline bool do_I_accept(double logaccept){ //, string name_accept, string name_count, List mcmc_pars){
  double acceptj = 1.0;
  if(!arma::is_finite(logaccept)){
    acceptj = 0.0;
  } else {
    if(logaccept < 0){
      acceptj = exp(logaccept);
    }
  }
  Rcpp::RNGScope scope;
  double u = arma::randu();
  if(u < acceptj){
    return true;
  } else {
    return false;
  }
}

inline void adapt(const arma::vec& param, 
           arma::vec& sumparam, 
           arma::mat& prodparam,
           arma::mat& paramsd, 
           arma::vec& sd_param, // mcmc sd
           int mc, 
           double accept_ratio){
  // adaptive update for theta = [sigmasq, phi, tau]
  
  int siz = param.n_rows;
  sumparam += param;
  prodparam += param * param.t();
  
  sd_param(mc+1) = sd_param(mc) + pow(mc+1.0, -g_exp) * (accept_ratio - tau_accept);
  //clog << sd_param.col(j).row(mc) << endl;
  if(sd_param(mc+1) > exp(rho_max)){
    sd_param(mc+1) = exp(rho_max);
  } else {
    if(sd_param(mc+1) < exp(-rho_min)){
      sd_param(mc+1) = exp(-rho_min);
    }
  }

  
  if(mc > g0){
    paramsd = sd_param(mc+1) / (mc+1.0) * 
      ( prodparam - (sumparam*sumparam.t())/(mc+0.0) ) +
      sd_param(mc+1) * EPS;
    
  }
}


class MHAdapter {
public:
  int npars;
  
  double EPS;
  double tau_accept;
  double g_exp;
  int g0;
  double rho_max;
  double rho_min;
  
  double propos_count;
  double accept_count;
  double accept_ratio;
  double propos_count_local;
  double accept_count_local;
  double accept_ratio_local;
  
  arma::vec sumparam;
  arma::mat prodparam;
  arma::mat paramsd;
  arma::vec sd_param;
  
  void count_proposal();
  void count_accepted();
  void update_ratios();
  void adapt(const arma::vec&, int);
  void print(int itertime, int mc);
  void print_summary(int time_tick, int time_mcmc, int m, int mcmc);
  
  MHAdapter(){};
  MHAdapter(int npars, int mcmc, const arma::mat& metropolis_sd);
};

inline MHAdapter::MHAdapter(int npars, int mcmc, const arma::mat& metropolis_sd){
  
  EPS = 0.1;
  tau_accept = 0.234;
  g_exp = .01;
  g0 = 500;
  rho_max = 2;
  rho_min = 5;
  
  propos_count = 0;
  accept_count = 0;
  accept_ratio = 0;
  propos_count_local = 0;
  accept_count_local = 0;
  accept_ratio_local = 0;
  
  sumparam = arma::zeros(npars); // include sigmasq
  prodparam = arma::zeros(npars,npars);
  paramsd = metropolis_sd; // proposal sd
  sd_param = arma::zeros(mcmc +1); // mcmc sd
}

inline void MHAdapter::count_proposal(){
  propos_count++;
  propos_count_local++;
}

inline void MHAdapter::count_accepted(){
  accept_count++;
  accept_count_local++;
}

inline void MHAdapter::update_ratios(){
  accept_ratio = accept_count/propos_count;
  accept_ratio_local = accept_count_local/propos_count_local;
}

inline void MHAdapter::adapt(const arma::vec& param, int mc){
  sumparam += param;
  prodparam += param * param.t();
  
  sd_param(mc+1) = sd_param(mc) + pow(mc+1.0, -g_exp) * (accept_ratio - tau_accept);
  //clog << sd_param.col(j).row(mc) << endl;
  if(sd_param(mc+1) > exp(rho_max)){
    sd_param(mc+1) = exp(rho_max);
  } else {
    if(sd_param(mc+1) < exp(-rho_min)){
      sd_param(mc+1) = exp(-rho_min);
    }
  }
  
  if(mc > g0){
    paramsd = sd_param(mc+1) / (mc+1.0) * 
      ( prodparam - (sumparam*sumparam.t())/(mc+0.0) ) +
      sd_param(mc+1) * EPS;
    
  }
}

inline void MHAdapter::print(int itertime, int mc){
  printf("%5d-th iteration [ %dms ] ~ MCMC acceptance %.2f%% (total: %.2f%%)\n", 
         mc+1, itertime, accept_ratio_local*100, accept_ratio*100);
}
inline void MHAdapter::print_summary(int time_tick, int time_mcmc, int m, int mcmc){
  
  accept_count_local = 0;
  propos_count_local = 0;
  
  printf("%.1f%% %dms (total: %dms) ~ MCMC acceptance %.2f%% (total: %.2f%%) \n",
         floor(100.0*(m+0.0)/mcmc),
         time_tick,
         time_mcmc,
         accept_ratio_local*100, accept_ratio*100);
}
inline double logistic(double x, double l=0, double u=1){
  return l + (u-l)/(1.0+exp(-x));
}

inline double logit(double x, double l=0, double u=1){
  return -log( (u-l)/(x-l) -1.0 );
}

inline arma::vec par_transf_fwd(arma::vec par){
  if(par.n_elem > 1){
    // gneiting nonsep 
    par(0) = log(par(0));
    par(1) = log(par(1));
    par(2) = logit(par(2));
    return par;
  } else {
    return log(par);
  }
}

inline arma::vec par_transf_back(arma::vec par){
  if(par.n_elem > 1){
    // gneiting nonsep 
    par(0) = exp(par(0));
    par(1) = exp(par(1));
    par(2) = logistic(par(2));
    return par;
  } else {
    return exp(par);
  }
}

inline arma::vec par_huvtransf_fwd(arma::vec par, const arma::mat& set_unif_bounds){
  for(int j=0; j<par.n_elem; j++){
    par(j) = logit(par(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1));
  }
  return par;
}

inline arma::vec par_huvtransf_back(arma::vec par, const arma::mat& set_unif_bounds){
  for(int j=0; j<par.n_elem; j++){
    par(j) = logistic(par(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1));
  }
  return par;
}

inline bool unif_bounds(arma::vec& par, const arma::mat& bounds){
  bool out_of_bounds = false;
  for(int i=0; i<par.n_elem; i++){
    arma::rowvec ibounds = bounds.row(i);
    if( par(i) < ibounds(0) ){
      out_of_bounds = true;
      par(i) = ibounds(0) + 1e-10;
    }
    if( par(i) > ibounds(1) ){
      out_of_bounds = true;
      par(i) = ibounds(1) - 1e-10;
    }
  }
  return out_of_bounds;
}

inline double lognormal_proposal_logscale(const double& xnew, const double& xold){
  // returns  + log x' - log x
  // to be + to log prior ratio log pi(x') - log pi(x)
  return log(xnew) - log(xold);
}

inline double normal_proposal_logitscale(const double& x, double l=0, double u=1){
  return //log(xnew * (l-xnew)) - log(xold * (l-xold)); 
  -log(u-x) - log(x-l);
}

inline double lognormal_logdens(const double& x, const double& m, const double& ssq){
  return -.5*(2*PI*ssq) - .5/ssq * pow(log(x) - m, 2) - log(x);
}

inline double gamma_logdens(const double& x, const double& a, const double& b){
  return -lgamma(a) + a*log(b) + (a-1.0)*log(x) - b*x;
}
inline double invgamma_logdens(const double& x, const double& a, const double& b){
  return -lgamma(a) + a*log(b) + (-a-1.0)*log(x) - b/x;
}
inline double beta_logdens(const double& x, const double& a, const double& b, const double& c=1.0){
  // unnormalized
  return (a-1.0)*log(x) + (b-1.0)*log(c-x);
}

inline double calc_jacobian(const arma::vec& new_param, const arma::vec& param, 
                            const arma::mat& set_unif_bounds){
  
  double jac = 0;
  for(int j=0; j<param.n_elem; j++){
    jac += normal_proposal_logitscale(param(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1)) -
      normal_proposal_logitscale(new_param(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1));
  }
  return jac;
}


inline double calc_prior_logratio(const arma::vec& new_param, 
                                  const arma::vec& param){
  
  double plr=0;
  for(int j=0; j<param.n_elem; j++){
    plr += 
      invgamma_logdens(new_param(0), 2.01, 3.0) -
      invgamma_logdens(param(0), 2.01, 3.0);
  }
  return plr;
}

