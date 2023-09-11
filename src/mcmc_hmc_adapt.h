#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>

class AdaptE {
public:
  int i;
  
  int n;
  double mu;
  double eps;
  double eps_bar;
  double H_bar;
  double gamma;
  double t0;
  double kappa;
  int M_adapt;
  int T;
  double delta;
  
  double alpha;
  double n_alpha;
  
  int which_hmc;
  
  bool adapt_C;
  
  arma::vec m_const;
  arma::mat Cchol_const;
  
  arma::mat C_const;
  arma::mat Ci_const;
  arma::mat Ccholinv_const;
  //double sweight;
  
  AdaptE();
  
  void init(double, int, int, int);
  void step();
  bool adapting();
  void adapt_step();
  bool use_C_const(double);
  void preconditioner_adapt_step(arma::mat&);
  void mean_adapt_step(const arma::vec& m); // yamala
  void preconditioner_store_update(const arma::mat&, const arma::mat&);
  void get_C_const();
};


inline AdaptE::AdaptE(){
  
}

inline void AdaptE::init(double eps0, int size, int which_hmc_in, int M_adapt_in=0){
  which_hmc = which_hmc_in;
  i = 0;
  mu = log(10 * eps0);
  eps = eps0;
  eps_bar = M_adapt_in == 0? eps0 : 1;
  H_bar = 0;
  gamma = .05;
  t0 = 10;
  kappa = 0.75;
  delta = which_hmc == 2? 0.7 : 0.575; // target accept
  M_adapt = M_adapt_in; // default is no adaptation
  
  alpha = 0;
  n_alpha = 0;
  
  adapt_C = which_hmc == 0;
  
  T = 500;
  n = size;

  C_const = arma::eye(n, n);
  Ci_const = C_const;
  Ccholinv_const = C_const;
  
  if(which_hmc == 7){
    m_const = arma::zeros(n);
    Cchol_const = arma::eye(n, n);
  }
}

inline bool AdaptE::use_C_const(double ru){
  double kappa = 0.33;
  return (i > T) & (ru > pow(i-T, -kappa));  
}


inline void AdaptE::preconditioner_store_update(const arma::mat& M, const arma::mat& Minvchol){
  if(which_hmc != 7){
    // simpa
    C_const = M; 
    Ccholinv_const = Minvchol; 
    Ci_const = Minvchol.t() * Minvchol;
  } else {
    // yamala
    C_const = M;
    Cchol_const = Minvchol; // this is actually Mchol
    Ccholinv_const = arma::inv(arma::trimatl(Minvchol));
    Ci_const = Ccholinv_const.t() * Ccholinv_const;
  }
}

inline void AdaptE::preconditioner_adapt_step(arma::mat& M){
  double gamma = i < T? 1.0/10.0 : 1/100.0;
  M = C_const + gamma * (M - C_const);  
}
inline void AdaptE::mean_adapt_step(const arma::vec& m){
  double gamma = i < T? 1.0/10.0 : 1/100.0;
  m_const = m_const + gamma * (m - m_const);  
}


inline void AdaptE::step(){
  i++;
}

inline bool AdaptE::adapting(){
  return (i < M_adapt);
}


inline void AdaptE::adapt_step(){
  int m = i+1; 
  if(m < M_adapt){
    H_bar = (1.0 - 1.0/(m + t0)) * H_bar + 1.0/(m + t0) * (delta - alpha/n_alpha);
    eps = exp(mu - sqrt(m)/gamma * H_bar);
    
    eps_bar = exp(pow(m, -kappa) * log(eps) + (1-pow(m, -kappa)) * log(eps_bar));
    //Rcpp::Rcout << "eps: " << eps << ", eps_bar: " << eps_bar << " | alpha: " << alpha << ", n_alpha: " << n_alpha << "\n";
  } else {
    eps = eps_bar;
  }
}


