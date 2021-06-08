#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>

class AdaptE {
public:
  int i;
  
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
  
  AdaptE();
  AdaptE(double, int);
  void step();
  bool adapting();
  void adapt_step();
};


inline AdaptE::AdaptE(){
  
}

inline AdaptE::AdaptE(double eps0, int M_adapt_in=0){
  i = 0;
  mu = log(10 * eps0);
  eps = eps0;
  eps_bar = M_adapt_in == 0? eps0 : 1;
  H_bar = 0;
  gamma = .05;
  t0 = 10;
  kappa = .75;
  delta = 0.6; // target accept
  M_adapt = M_adapt_in;
  
  alpha = 0;
  n_alpha = 0;
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

/*
 * 
#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>
 
 class AdaptE {
 public:
 int i;
 
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
 
 AdaptE();
 AdaptE(double, int);
 void step();
 bool adapting();
 void adapt_step();
 };
 
 
 inline AdaptE::AdaptE(){
 
 }
 
 inline AdaptE::AdaptE(double eps0, int M_adapt_in=0){
 i = 0;
 mu = log(10 * eps0);
 eps = eps0;
 eps_bar = M_adapt_in == 0? eps0 : 1;
 H_bar = 0;
 gamma = .05;
 t0 = 10;
 kappa = .75;
 delta = 0.65;
 M_adapt = M_adapt_in;
 
 alpha = 0;
 n_alpha = 0;
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
 //double a_over_na = n_alpha!=0? alpha/n_alpha : 0.0;
 H_bar = (1 - 1.0/(m + t0)) * H_bar + 1.0/(m + t0) * (delta - alpha/n_alpha);
 eps = exp(mu - sqrt(m)/gamma * H_bar);
 eps_bar = exp(pow(m, -kappa) * log(eps) + (1-pow(m, -kappa)) * log(eps_bar));
 Rcpp::Rcout << "eps: " << eps << ", eps_bar: " << eps_bar << " | alpha: " << alpha << ", n_alpha: " << n_alpha << "\n";
 } else {
 eps = eps_bar;
 }
 }
 
 */
