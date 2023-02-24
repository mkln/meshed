//#define ARMA_WARN_LEVEL 1

#ifndef XCOV_LMC 
#define XCOV_LMC

#ifdef _OPENMP
#include <omp.h>
#endif


#include <RcppArmadillo.h>

using namespace std;

struct MaternParams {
  bool using_ps;
  bool estimating_nu;
  double * bessel_ws;
  int twonu;
};

// matern
arma::mat matern(const arma::mat& x, const arma::mat& y, const double& phi, const double& nu, double * bessel_ws, bool same);

void gneiting2002_inplace(arma::mat& res, const arma::mat& coords, const arma::uvec& ix, const arma::uvec& iy, 
                       const double& a, const double& c, const double& beta, const double& sigmasq, const double& nu, bool same=false);

void kernelp_inplace(arma::mat& res,
                     const arma::mat& Xcoords, const arma::uvec& ind1, const arma::uvec& ind2, 
                     const arma::vec& theta, bool same);
  
arma::mat Correlationf(const arma::mat& coords, const arma::uvec& ix, const arma::uvec& iy, 
                       const arma::vec& theta, MaternParams& matern, bool same);

arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, 
                       const arma::vec& theta, MaternParams& matern, bool same);

// inplace functions
void CviaKron_invsympd_(arma::cube& CCi, 
                        const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta, MaternParams& matern);
  
double CviaKron_HRi_(arma::cube& H, arma::cube& Ri, arma::cube& Kppi, 
                     const arma::cube& Cxx,
                     const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta, MaternParams& matern);

//double CviaKron_invsympd_wdet_(arma::cube& res,
//                         const arma::mat& coords, const arma::uvec& indx, 
//                         int k, const arma::mat& theta, MaternParams& matern);

void CviaKron_HRj_bdiag_(
    arma::cube& Hj, arma::cube& Rj, arma::cube& Rij,
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


#endif
