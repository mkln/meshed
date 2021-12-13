#include <RcppArmadillo.h>
#include <string>

using namespace std;

arma::vec gaussian_linkinv(const arma::vec& eta);

arma::vec gaussian_mueta(const arma::vec& eta);

arma::vec gaussian_variance(const arma::vec& mu);

arma::vec binomial_linkinv(const arma::vec& eta);

arma::vec binomial_mueta(const arma::vec& eta);

arma::vec binomial_variance(const arma::vec& mu);


arma::vec negbinomial_linkinv(const arma::vec& eta);

arma::vec negbinomial_mueta(const arma::vec& eta);

arma::vec negbinomial_variance(const arma::vec& mu, const double& tausq);


arma::vec poisson_linkinv(const arma::vec& eta);

arma::vec poisson_mueta(const arma::vec& eta);

arma::vec poisson_variance(const arma::vec& mu);

arma::vec irls_bayes_cpp(const arma::vec& y, 
                         const arma::mat& X, 
                         const arma::vec& offset,
                         const arma::mat& Vi,
                         double scaling=1,
                         std::string family = "gaussian", 
                         int maxit = 25,
                         double tol = 1e-08);

arma::vec irls_step(const arma::vec& start_x,
                    const arma::vec& y, 
                    const arma::mat& X, 
                    const arma::vec& offset,
                    const arma::mat& Vi,
                    int fam_id,
                    double scaling=1);
