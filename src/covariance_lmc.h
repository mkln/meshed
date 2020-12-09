#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>

using namespace std;


struct MaternParams {
  double * bessel_ws;
  int twonu;
};


arma::mat CmaternInv(const arma::mat& x,
                     const double& sigmasq,
                     const double& effrange, const double& nu, 
                     const double& tausq);

// matern
arma::mat matern(const arma::mat& x, const arma::mat& y, const double& phi, const double& nu, double * bessel_ws, bool same);

// matern half integer correlation + reparametrization
//[[Rcpp::export]]
arma::mat matern_halfint(const arma::mat& x, const arma::mat& y, const double& phi, bool same, int numinushalf=0);

//[[Rcpp::export]]
arma::mat gneiting2002(const arma::mat& x, const arma::mat& y, 
                       const double& a, const double& c, const double& beta, bool same=false);

arma::mat Correlationf(const arma::mat& x, const arma::mat& y, const arma::vec& theta, MaternParams& matern, bool same);

arma::mat CviaKron(const arma::mat& coords, 
                   const arma::uvec& indx, const arma::uvec& indy,
                   int k, const arma::mat& theta, MaternParams& matern, bool same=false);

// inplace functions
void CviaKron_invsympd_(arma::cube& CCi, 
                        const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta, MaternParams& matern);
  
double CviaKron_HRi_(arma::cube& H, arma::cube& Ri, arma::cube& Richol,
                     const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta, MaternParams& matern);

double CviaKron_invsympd_wdet_(arma::cube& res,
                         const arma::mat& coords, const arma::uvec& indx, 
                         int k, const arma::mat& theta, MaternParams& matern);

void CviaKron_HRj_bdiag_(
    arma::cube& Hj, arma::cube& Rj, 
    const arma::cube& Kxxi_cache,
    const arma::mat& coords, const arma::uvec& indx, 
    const arma::uvec& naix, const arma::uvec& indy, 
    int k, const arma::mat& theta, MaternParams& matern);


// for predictions
void CviaKron_HRj_chol_bdiag_wcache(
    arma::cube& Hj, arma::mat& Rjchol, 
    const arma::cube& Kxxi_cache, const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta, MaternParams& matern);

void CviaKron_HRj_chol_bdiag(
    arma::cube& Hj, arma::mat& Rjchol, arma::cube& Kxxi_parents,
    const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta, MaternParams& matern);