#include <RcppArmadillo.h>

inline bool do_I_accept(double logaccept){
  double u = R::runif(0,1);
  bool answer = exp(logaccept) > u;
  return answer;
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
  for(unsigned int j=0; j<par.n_elem; j++){
    
    if( (set_unif_bounds(j, 0) > -arma::datum::inf) || (set_unif_bounds(j, 1) < arma::datum::inf) ){
      if(set_unif_bounds(j, 1) == arma::datum::inf){
        // lognormal proposal
        par(j) = log(par(j));
      } else {
        // logit normal proposal
        par(j) = logit(par(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1));
      }
    }
  }
  return par;
}

inline arma::vec par_huvtransf_back(arma::vec par, const arma::mat& set_unif_bounds){
  for(unsigned int j=0; j<par.n_elem; j++){
    if( (set_unif_bounds(j, 0) > -arma::datum::inf) || (set_unif_bounds(j, 1) < arma::datum::inf) ){
      if(set_unif_bounds(j, 1) == arma::datum::inf){
        // lognormal proposal
        par(j) = exp(par(j));
      } else {
        // logit normal proposal
        par(j) = logistic(par(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1));
      }
    }
  }
  return par;
}

inline bool unif_bounds(arma::vec& par, const arma::mat& bounds){
  bool out_of_bounds = false;
  for(unsigned int i=0; i<par.n_elem; i++){
    arma::rowvec ibounds = bounds.row(i);
    if( par(i) < ibounds(0) ){
      out_of_bounds = true;
      par(i) = ibounds(0) + 1e-1;
    }
    if( par(i) > ibounds(1) ){
      out_of_bounds = true;
      par(i) = ibounds(1) - 1e-1;
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
  return -.5*(2*M_PI*ssq) - .5/ssq * pow(log(x) - m, 2) - log(x);
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
  for(unsigned int j=0; j<param.n_elem; j++){
    if( (set_unif_bounds(j, 0) > -arma::datum::inf) || (set_unif_bounds(j, 1) < arma::datum::inf) ){
      if(set_unif_bounds(j, 1) == arma::datum::inf){
        // lognormal proposal
        jac += lognormal_proposal_logscale(new_param(j), param(j));
      } else {
        // logit normal proposal
        jac += normal_proposal_logitscale(param(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1)) -
          normal_proposal_logitscale(new_param(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1));
      }
    }
  }
  
  return jac;
}


inline double calc_prior_logratio(const arma::vec& new_param, 
                            const arma::vec& param, double a=2.01, double b=1){
  
  double plr=0;
  for(unsigned int j=0; j<param.n_elem; j++){
    plr += 
      invgamma_logdens(new_param(0), a, b) -
      invgamma_logdens(param(0), a, b);
  }
  return plr;
}



class RAMAdapt {
public:
  
  arma::mat starting_sd;
  
  // Robust adaptive MCMC Vihala 2012
  int p;
  arma::mat Ip;
  arma::mat paramsd;
  arma::mat Sigma; // (Ip + update)
  arma::mat S;
  double alpha_star;
  double eta;
  double gamma;
  
  // startup period variables
  int g0;
  int i;
  int c;
  bool flag_accepted;
  arma::mat prodparam;
  bool started;
  
  double propos_count;
  double accept_count;
  double accept_ratio;
  int history_length;
  arma::vec acceptreject_history;
  
  void count_proposal();
  void count_accepted();
  void update_ratios();
  void reset();
  void adapt(const arma::vec&, double, int);
  void print(int itertime, int mc);
  void print_summary(int time_tick, int time_mcmc, int m, int mcmc);
  void print_acceptance();
  
  RAMAdapt(){};
  RAMAdapt(int npars, const arma::mat& metropolis_sd, double);
};

inline RAMAdapt::RAMAdapt(int npars, const arma::mat& metropolis_sd, double target_accept=.234){
  starting_sd = metropolis_sd;
  
  p = npars;
  alpha_star = target_accept;
  gamma = 0.5 + 1e-16;
  Ip = arma::eye(p,p);
  g0 = 100;
  S = metropolis_sd * metropolis_sd.t();
  
  paramsd = arma::chol(S, "lower");
  
  //Rcpp::Rcout << "starting paramsd: " << endl;
  //Rcpp::Rcout << paramsd << endl;
  
  prodparam = paramsd / (g0 + 1.0);
  started = false;
  flag_accepted = false;
  
  propos_count = 0;
  accept_count = 0;
  accept_ratio = 0;
  history_length = 200;
  acceptreject_history = arma::zeros(history_length);
  c = 0;
}

inline void RAMAdapt::reset(){
  S = starting_sd * starting_sd.t();
  paramsd = arma::chol(S, "lower");
  
  prodparam = paramsd / (g0 + 1.0);
  started = false;
  
  propos_count = 0;
  accept_count = 0;
  accept_ratio = 0;
  history_length = 200;
  acceptreject_history = arma::zeros(history_length);
  c = 0;
}

inline void RAMAdapt::count_proposal(){
  propos_count++;
  c ++;
  flag_accepted = false;
}

inline void RAMAdapt::count_accepted(){
  accept_count++;
  acceptreject_history(c % history_length) = 1;
  flag_accepted = true;
}

inline void RAMAdapt::update_ratios(){
  accept_ratio = accept_count/propos_count;
  if(!flag_accepted){
    acceptreject_history(c % history_length) = 0;
  }
}

inline void RAMAdapt::adapt(const arma::vec& U, double alpha, int mc){
  //if((c < g0) & false){
  //  prodparam += U * U.t() / (mc + 1.0);
  //} else {
    if(!started & (c < 2*g0)){
      // if mc > 2*g0 this is being called from a restarted mcmc
      // (and if not, it would make no difference since g0 is small)
      //paramsd = prodparam;
      started = true;
    }
    i = c-g0;
    eta = std::min(1.0, (p+.0) * pow(i+1.0, -gamma));
    alpha = std::min(1.0, alpha);
    
    if(started){
      Sigma = Ip + eta * (alpha - alpha_star) * U * U.t() / arma::accu(U % U);
      //Rcpp::Rcout << "Sigma: " << endl << Sigma;
      S = paramsd * Sigma * paramsd.t();
      //Rcpp::Rcout << "S: " << endl << S;
      paramsd = arma::chol(S, "lower");
    }
  
    //Rcpp::Rcout << "mc: " << mc << " paramsd: " << endl;
    //Rcpp::Rcout << paramsd << endl;
  //}
}

inline void RAMAdapt::print(int itertime, int mc){
  Rprintf("%5d-th iteration [ %dms ] ~ (Metropolis acceptance for theta: %.2f%%, average %.2f%%) \n", 
         mc+1, itertime, arma::mean(acceptreject_history)*100, accept_ratio*100);
}

inline void RAMAdapt::print_summary(int time_tick, int time_mcmc, int m, int mcmc){
  double time_iter = (.0 + time_mcmc)/(m+1);
  
  double etr = (mcmc-m-1) * time_iter * 1.1 / 1000;  // seconds
  const char* unit = etr>60 ? "min" : "s";
  
  etr = etr > 60 ? etr/60 : etr;
  
  //Rcpp::Rcout << m+1 << " " << mcmc << " " << time_iter << " " << mcmc-m-1 << " " << (mcmc-m-1) * time_iter << "\n";
  Rprintf("%.1f%% elapsed: %5dms (+%5dms). ETR: %.2f%s. \n",
         100.0*(m+1.0)/mcmc,
         time_mcmc,
         time_tick,
         etr, unit);
}

inline void RAMAdapt::print_acceptance(){
  Rprintf("  theta: Metrop. acceptance %.2f%%, average %.2f%% \n",
          arma::mean(acceptreject_history)*100, accept_ratio*100);
}
