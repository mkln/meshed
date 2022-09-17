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
  double delta;
  
  double alpha;
  double n_alpha;
  
  bool adapt_C;
  int i_C_adapt;
  arma::mat C_const;
  //arma::mat Cinv_const;
  arma::mat Ccholinv_const;
  double sweight;
  
  AdaptE();
  
  void init(double, int, bool, bool, int);
  void step();
  bool adapting();
  void adapt_step();
  bool use_C_const(double);
  void weight_average_C_temp(arma::mat&);
  void update_C_const(const arma::mat&, const arma::mat&);
  void get_C_const();
};


inline AdaptE::AdaptE(){
  
}

inline void AdaptE::init(double eps0, int size, bool rm_warmup=true, bool nuts=false, int M_adapt_in=0){
  i = 0;
  mu = log(10 * eps0);
  eps = eps0;
  eps_bar = M_adapt_in == 0? eps0 : 1;
  H_bar = 0;
  gamma = .05;
  t0 = 10;
  kappa = 0.75;
  delta = nuts? 0.7 : 0.575; // target accept
  M_adapt = M_adapt_in; // default is no adaptation
  
  alpha = 0;
  n_alpha = 0;
  
  adapt_C = rm_warmup;
  i_C_adapt = 2000;
  n = size;

  if(adapt_C){
    C_const = arma::eye(n, n);
    //Cinv_const = C_const;
    Ccholinv_const = C_const;
    sweight = arma::accu(pow(arma::regspace<arma::vec>(1, i_C_adapt), 2.0));
  }
}

inline bool AdaptE::use_C_const(double ru){
  //return adapt_C & (i>i_C_adapt);
  // outside of initial burn period AND not randomly adapting
  if(adapt_C){
    double T = 500.0;
    double kappa = 0.33;
    return (i > T) & (ru > pow(i-T, -kappa));  
  } else {
    return false;
  }
  
}


inline void AdaptE::update_C_const(const arma::mat& M, const arma::mat& Minvchol){
  C_const = M;//C_const + gamma * (M - C_const);
  Ccholinv_const = Minvchol;//arma::inv(arma::trimatl(arma::chol(arma::symmatu(C_const), "lower")));
}

inline void AdaptE::weight_average_C_temp(arma::mat& M){
  if(adapt_C){
    double gamma = 1.0/100.0;
    M = C_const + gamma * (M - C_const);  
  }
}

/*
inline void AdaptE::update_C_const(const arma::mat& M){
  if(adapt_C){
    if(i <= i_C_adapt){
      // keep averaging in the burn period
      //C_const = (C_const*(i+.0) + M)/(i+1.0);
      C_const += pow(i+.0, 2.0) * M/sweight;
      if(i == i_C_adapt){
        // switch time from RMMALA to simplified MMALA
        Ccholinv_const = arma::inv(arma::trimatl(arma::chol(arma::symmatu(C_const), "lower")));
        //Cinv_const = Ccholinv_const.t() * Ccholinv_const;
      } 
    }
  }
}

inline void AdaptE::weight_average_C_temp(arma::mat& M){
  // returns a weighted average of the input matrix and the current C_const
  // based on the current weight schedule
  if(adapt_C){
    if(i <= i_C_adapt){
      // keep averaging in the burn period
      //C_const = (C_const*(i+.0) + M)/(i+1.0);
      double curweight = arma::accu(pow(arma::regspace<arma::vec>(1, i), 2.0)) / sweight;
      double remweight = 1-curweight;
      M = C_const + (1-curweight) * M;
    } 
  } 
}
*/

inline void AdaptE::step(){
  i++;
}

inline bool AdaptE::adapting(){
  return (i < M_adapt);
}


inline void AdaptE::adapt_step(){
  int m = i+1; //i<i_C_adapt? 0 : i+1-i_C_adapt;
  if(m < M_adapt){
    
    H_bar = (1.0 - 1.0/(m + t0)) * H_bar + 1.0/(m + t0) * (delta - alpha/n_alpha);
    eps = exp(mu - sqrt(m)/gamma * H_bar);
    
    eps_bar = exp(pow(m, -kappa) * log(eps) + (1-pow(m, -kappa)) * log(eps_bar));
    //Rcpp::Rcout << "eps: " << eps << ", eps_bar: " << eps_bar << " | alpha: " << alpha << ", n_alpha: " << n_alpha << "\n";
  } else {
    eps = eps_bar;
  }
}


