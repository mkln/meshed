#include <RcppArmadillo.h>

using namespace std;


// matern half integer correlation + reparametrization
//[[Rcpp::export]]
arma::mat matern_halfint(const arma::mat& x, const arma::mat& y, const double& phi, bool same, int numinushalf=0);

//[[Rcpp::export]]
arma::mat gneiting2002(const arma::mat& x, const arma::mat& y, 
                       const double& a, const double& c, const double& beta, bool same=false);

arma::mat Correlationf(const arma::mat& x, const arma::mat& y, const arma::vec& theta, bool same);

arma::mat CviaKron(const arma::mat& coords, 
                   const arma::uvec& indx, const arma::uvec& indy,
                   int k, const arma::mat& theta, bool same=false);
  
arma::mat CviaKron_invsympd(const arma::mat& coords, const arma::uvec& indx, 
                            int k, const arma::mat& theta);

arma::mat CviaKron_chol(const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta);

arma::mat CviaKron_H(const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta);

arma::mat CviaKron_R(const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta);

void CviaKron_HRj_bdiag(
    arma::field<arma::mat>& Hj, arma::field<arma::mat>& Rj, 
    const arma::field<arma::mat>& Kxxi_cache,
    const arma::mat& coords, const arma::uvec& indx, 
    const arma::uvec& naix, const arma::uvec& indy, 
    int k, const arma::mat& theta);
  
double CviaKron_invchol(arma::mat& res,
                        const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta);
  
double CviaKron_HRi(arma::mat& H, arma::mat& R,
                 const arma::mat& coords, 
                 const arma::uvec& indx, const arma::uvec& indy, 
                 int k, const arma::mat& theta);

arma::mat CviaKron_Ri(const arma::mat& coords, 
                      const arma::uvec& indx, const arma::uvec& indy,  
                      int k, const arma::mat& theta);



arma::mat CviaKron_Rchol(const arma::mat& coords, 
                         const arma::uvec& indx, const arma::uvec& indy,  
                         int k, const arma::mat& theta);


arma::mat CviaKron_Rcholinv(const arma::mat& coords, 
                            const arma::uvec& indx, const arma::uvec& indy,  
                            int k, const arma::mat& theta);

// inplace functions
double CviaKron_HRi_(arma::cube& H, arma::cube& Ri,
                     const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta);

double CviaKron_invsympd_wdet_(arma::cube& res,
                         const arma::mat& coords, const arma::uvec& indx, 
                         int k, const arma::mat& theta);

void CviaKron_HRj_bdiag_(
    arma::cube& Hj, arma::cube& Rj, 
    const arma::cube& Kxxi_cache,
    const arma::mat& coords, const arma::uvec& indx, 
    const arma::uvec& naix, const arma::uvec& indy, 
    int k, const arma::mat& theta);


// for predictions
void CviaKron_HRj_chol_bdiag_wcache(
    arma::cube& Hj, arma::mat& Rjchol, 
    const arma::cube& Kxxi_cache, const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta);

void CviaKron_HRj_chol_bdiag(
    arma::cube& Hj, arma::mat& Rjchol, const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta);