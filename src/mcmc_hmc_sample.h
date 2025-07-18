#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>

//#include "../distributions/mvnormal.h"

//#include "distparams.h"
#include "mcmc_hmc_nodes.h"
#include "mcmc_hmc_adapt.h"

// mala and rm-mala
template <class T>
arma::mat mala_cpp(arma::mat current_q, 
                                   T& postparams,
                                   AdaptE& adaptparams, 
                                   const arma::mat& rnorm_mat,
                                   const double& runifvar,
                                   bool debug=false);
  
template <class T>
arma::mat smmala_cpp(arma::mat current_q, 
                                      T& postparams,
                                      AdaptE& adaptparams, 
                                      const arma::mat& rnorm_mat,
                                      const double& runifvar,
                                      bool debug=false);

template <class T>
void bounder(T& x, double DD = 1e10);

template <class T>
arma::mat simpa_cpp(arma::mat current_q, 
                               T& postparams,
                               AdaptE& adaptparams, 
                               const arma::mat& rnorm_mat,
                               const double& runifvar, const double& runifadapt, 
                               bool debug=false);

template <class T>
arma::mat yamala_cpp(arma::mat current_x, 
                            T& model,
                            AdaptE& adaptparams, const arma::mat& rnorm_mat,
                            const double& runifvar, const double& runifadapt, 
                            bool debug=false);

template <class T>
arma::mat ellipt_slice_sampler(arma::mat current_q, 
                               T& postparams,
                               AdaptE& adaptparams, 
                               const arma::mat& rnorm_mat,
                               const double& runifvar,
                               bool debug=false);


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
void leapfrog(pq_point &z, float epsilon, T& postparams,  int k=1);

// mala and rm-mala
template <class T>
arma::mat hmc_cpp(arma::mat current_q, 
                          T& postparams,
                          AdaptE& adaptparams, 
                          const arma::mat& rnorm_mat,
                          const double& runifvar,
                          double hmc_lambda=1,
                          bool debug=false);
template <class T>
double find_reasonable_stepsize(const arma::mat& current_q, T& postparams, const arma::mat& rnorm_mat);
// nuts
struct nuts_util {
  // Constants through each recursion
  double log_u; // uniform sample
  double H0; 	// Hamiltonian of starting point?
  int sign; 	// direction of the tree in a given iteration/recursion
  
  // Aggregators through each recursion
  int n_tree;
  double sum_prob; 
  bool criterion;
  
  // just to guarantee bool initializes to valid value
  nuts_util() : criterion(false) { }
};


bool compute_criterion(const arma::vec& p_sharp_minus, 
                              const arma::vec& p_sharp_plus,
                              const arma::vec& rho);

template <class T>
int BuildTree(pq_point& z, pq_point& z_propose, 
                     arma::vec& p_sharp_left, 
                     arma::vec& p_sharp_right, 
                     arma::vec& rho, 
                     nuts_util& util, 
                     int depth, float epsilon,
                     T& postparams,
                     double& alpha,
                     double& n_alpha,
                     double joint_zero,
                       int k=1);
template <class T>
arma::mat nuts_cpp(arma::mat current_q, 
                                     T& postparams,
                                     AdaptE& adaptparams);

