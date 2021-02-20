#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>

#include "../distributions/mvnormal.h"

//#include "distparams.h"
#include "hmc_nodes.h"
#include "hmc_adapt.h"

inline arma::mat unvec(arma::vec q, int k){
  arma::mat result = arma::mat(q.memptr(), q.n_elem/k, k);
  return result;
}

// Position q and momentum p
struct pq_point {
  arma::vec q;
  arma::vec p;
  
  explicit pq_point(int n): q(n), p(n) {}
  pq_point(const pq_point& z): q(z.q.size()), p(z.p.size()) {
    q = z.q;
    p = z.p;
  }
  
  pq_point& operator= (const pq_point& z) {
    if (this == &z)
      return *this;
    
    q = z.q;
    p = z.p;
    
    return *this;
  }
};


template <class T>
inline void leapfrog(pq_point &z, float epsilon, T& postparams, int k=1){
  arma::mat qmat = unvec(z.q, k);
  z.p += epsilon * 0.5 * postparams.gradient_logfullcondit(qmat);
  //arma::vec qvecplus = arma::vectorise(z.q) + epsilon * z.p;
  z.q += epsilon * z.p;
  qmat = unvec(z.q, k);
  z.p += epsilon * 0.5 * postparams.gradient_logfullcondit(qmat);
}

template <class T>
inline double find_reasonable_stepsize(const arma::mat& current_q, T& postparams){
  int K = current_q.n_elem;
  
  pq_point z(K);
  arma::vec p0 = //postparams.Michol * 
    arma::randn(K);
  
  double epsilon = 1;
  
  arma::vec veccurq = arma::vectorise(current_q);
  
  z.q = veccurq;
  z.p = p0;
  
  double p_orig = postparams.logfullcondit(current_q) - 0.5* arma::conv_to<double>::from(z.p.t() * //postparams.M * 
                                                                                    z.p);//sum(z.p % z.p); 
  //Rcpp::Rcout << "before:  " << p_orig << "\n";
  leapfrog(z, epsilon, postparams, current_q.n_cols);
  //Rcpp::Rcout << "done leapfrog " << endl;
  arma::mat newq = unvec(z.q, current_q.n_cols);
  double p_prop = postparams.logfullcondit(newq) - 0.5* arma::conv_to<double>::from(z.p.t() * //postparams.M * 
                                                                                    z.p);//sum(z.p % z.p); 
  //Rcpp::Rcout << "after:  " << p_prop << "\n";
  double p_ratio = exp(p_prop - p_orig);
  double a = 2 * (p_ratio > .5) - 1;
  int it=0;
  bool condition = (pow(p_ratio, a) > pow(2.0, -a)) || std::isnan(p_ratio);
  
  while( condition & (it < 50) ){
    it ++;
    double twopowera = pow(2.0, a);
    epsilon = twopowera * epsilon;
    
    leapfrog(z, epsilon, postparams, current_q.n_cols);
    newq = unvec(z.q, current_q.n_cols);
    p_prop = postparams.logfullcondit(newq) - 0.5* arma::conv_to<double>::from(z.p.t() * //postparams.M * 
                                                                               z.p);//sum(z.p % z.p); 
    p_ratio = exp(p_prop - p_orig);
    
    condition = (pow(p_ratio, a)*twopowera > 1.0) || std::isnan(p_ratio);
    //Rcpp::Rcout << "epsilon : " << epsilon << " p_ratio " << p_ratio << " " << p_prop << "," << p_orig << " .. " << pow(p_ratio, a) << "\n";
    // reset
    z.q = veccurq;
    z.p = p0;
  }
  if(it == 50){
    epsilon = .01;
    Rcpp::Rcout << "Set epsilon to " << epsilon << " after no reasonable stepsize could be found. (?)\n";
  }
  return epsilon/2.0;
} 



template <class T>
inline arma::mat sample_one_mala_cpp(arma::mat current_q, 
                                     T& postparams,
                                     AdaptE& adaptparams, bool rm=true, bool adapt=true, bool debug=false){
  
  
  int k = current_q.n_cols;
  
  // if(true){
  // // via leapfrog
  // arma::vec q = current_q;
  // double joint0 = postparams.logfullcondit(current_q) - 0.5* arma::conv_to<double>::from(p.t() * p);
  // p += adaptparams.eps * 0.5 * postparams.gradient_logfullcondit(q);
  // q += adaptparams.eps * p;
  // p += adaptparams.eps * 0.5 * postparams.gradient_logfullcondit(q);
  // double joint1 = postparams.logfullcondit(q) - 0.5 * arma::conv_to<double>::from(p.t() * p);
  // }
  
  std::chrono::steady_clock::time_point t0;
  std::chrono::steady_clock::time_point t1;
  double timer=0;
  
  double eps1, eps2;
  arma::mat MM, Minvchol, Minv;

  if(!rm){
    MM = arma::eye(current_q.n_elem, current_q.n_elem);
  } else {
    //Rcpp::Rcout << "neghess_logfullcondit start " << endl;
    MM = postparams.neghess_logfullcondit(current_q);
    //Rcpp::Rcout << "neghess_logfullcondit start " << endl;
  }
  
  //Rcpp::Rcout << "hess done " << endl;
  
  if(adapt){
    eps2 = pow(adaptparams.eps, 2.0);
    eps1 = adaptparams.eps;
  } else {
    eps2 = 2;// * adaptparams.eps;
    eps1 = 1;// * adaptparams.eps;
  }
  
  //Rcpp::Rcout << MM << endl;
  try {
    Minvchol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(MM), "lower")));
  } catch(...) {
   MM = arma::eye(current_q.n_elem, current_q.n_elem);
   Minvchol = MM;
  }
  
  Minv = eps2 * 0.5 * Minvchol.t() * Minvchol;
  
  // currents
  arma::vec xgrad;
  double joint0;
  
  //Rcpp::Rcout << "compute_dens_and_grad start " << endl;
  t0 = std::chrono::steady_clock::now();
  postparams.compute_dens_and_grad(joint0, xgrad, current_q);
  t1 = std::chrono::steady_clock::now();
  timer += std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
  //cpp::Rcout << "compute_dens_and_grad end " << endl;
  
  if(xgrad.has_inf() || std::isnan(joint0)){
    adaptparams.alpha = 0.0;
    adaptparams.n_alpha = 1.0;
    adaptparams.adapt_step();
    
    return current_q;
  }
  
  arma::vec veccurq = arma::vectorise(current_q);
  
  //arma::vec xgrad = postparams.gradient_logfullcondit(current_q);
  arma::vec proposal_mean = veccurq + Minv * xgrad;// / max(eps2, arma::norm(xgrad));
  
  // proposal value
  arma::vec p = arma::randn(current_q.n_elem);  
  arma::vec q = proposal_mean + eps1 * Minvchol.t() * p;
  arma::mat qmat = arma::mat(q.memptr(), q.n_elem/k, k);
  // target at current and proposed values
  
  //double joint0 = postparams.logfullcondit(current_q);
  
  // proposal
  double joint1; // = postparams.logfullcondit(qmat);
  arma::vec revgrad;// = postparams.gradient_logfullcondit(qmat);
  
  t0 = std::chrono::steady_clock::now();
  postparams.compute_dens_and_grad(joint1, revgrad, qmat);
  t1 = std::chrono::steady_clock::now();
  timer += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  
  if(revgrad.has_inf() || std::isnan(joint1) || std::isinf(joint1)){
    adaptparams.alpha = 0.0;
    adaptparams.n_alpha = 1.0;
    adaptparams.adapt_step();
    
    return current_q;
  }
  
  arma::vec reverse_mean = q + Minv * revgrad;// / max(eps2, arma::norm(revgrad));;
  
  double prop0to1 = -.5/(eps1*eps1) * arma::conv_to<double>::from(
    (q - proposal_mean).t() * MM * (q - proposal_mean) );
  
  double prop1to0 = -.5/(eps1*eps1) * arma::conv_to<double>::from(
    (veccurq - reverse_mean).t() * MM * (veccurq - reverse_mean) );
  
  adaptparams.alpha = std::min(1.0, exp(joint1 + prop1to0 - joint0 - prop0to1));
  adaptparams.n_alpha = 1.0;
  
  if(R::runif(0, 1) < adaptparams.alpha){ 
    current_q = qmat;
    
    if(debug){
      Rcpp::Rcout << "accepted logdens " << joint1 << endl;
    }
  }
  //if(!rm){
    adaptparams.adapt_step();
  //}  
  
  //Rcpp::Rcout << "exiting mala" << endl;
  return current_q;
}

