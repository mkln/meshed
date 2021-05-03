#include <RcppArmadillo.h>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <omp.h>


//[[Rcpp::export]]
arma::mat Cov_matern(const arma::mat& x, const arma::mat& y, 
                     const double& sigmasq,
                     const double& phi, const double& nu, 
                     const double& tausq,
                     bool same, int nThreads=1){
  
  //thread safe stuff
  int threadID = 0;
  int bessel_ws_inc = static_cast<int>(1.0+nu);//see bessel_k.c for working space needs
  double *bessel_ws = (double *) R_alloc(nThreads*bessel_ws_inc, sizeof(double));
  
  omp_set_num_threads(nThreads);
  
  //create the correlation matrix (now thread-safe)
  
  double pow2_nu1_gammanu = pow(2.0, 1.0-nu) / R::gammafn(nu);
  
  arma::mat res = arma::zeros(x.n_rows, y.n_rows);
  
  if(same){
#ifdef _OPENMP
#pragma omp parallel for private(threadID)
#endif
    for(int i=0; i<x.n_rows; i++){
#ifdef _OPENMP
      threadID = omp_get_thread_num();
#endif
      arma::rowvec cri = x.row(i);
      for(int j=i; j<y.n_rows; j++){
        arma::rowvec delta = cri - y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          res(i, j) = sigmasq * pow(hphi, nu) * pow2_nu1_gammanu *
            R::bessel_k_ex(hphi, nu, 1.0, &bessel_ws[threadID*bessel_ws_inc]);
        } else {
          res(i, j) = sigmasq + tausq;
        }
      }
    }
    res = arma::symmatu(res);
  } else {
#ifdef _OPENMP
#pragma omp parallel for private(threadID)
#endif
    for(int i=0; i<x.n_rows; i++){
#ifdef _OPENMP
      threadID = omp_get_thread_num();
#endif
      arma::rowvec cri = x.row(i);
      for(int j=0; j<y.n_rows; j++){
        arma::rowvec delta = cri - y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          res(i, j) = sigmasq * pow(hphi, nu) * pow2_nu1_gammanu *
            R::bessel_k_ex(hphi, nu, 1.0, &bessel_ws[threadID*bessel_ws_inc]);
        } else {
          res(i, j) = sigmasq + tausq;
        }
      }
    }
  }
  return res;
}

//[[Rcpp::export]]
arma::mat Cov_matern2(const arma::mat& x, const arma::mat& y, const double& phi, bool same, int twonu){
  // 0 based indexing
  arma::mat res = arma::zeros(x.n_rows, y.n_rows);
  double nugginside = 0;
  if(same){
    for(int i=0; i<x.n_rows; i++){
      arma::rowvec cri = x.row(i);
      for(int j=i; j<y.n_rows; j++){
        arma::rowvec delta = cri - y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          if(twonu == 1){
            res(i, j) = exp(-hphi);
          } else {
            if(twonu == 3){
              res(i, j) = exp(-hphi) * (1 + hphi);
            } else {
              if(twonu == 5){
                res(i, j) = (1 + hphi + hphi*hphi / 3.0) * exp(-hphi);
              }
            }
          }
        } else {
          res(i, j) = 1.0 + nugginside;
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for(int i=0; i<x.n_rows; i++){
      arma::rowvec cri = x.row(i);
      for(int j=0; j<y.n_rows; j++){
        arma::rowvec delta = cri - y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          if(twonu == 1){
            res(i, j) = exp(-hphi);
          } else {
            if(twonu == 3){
              res(i, j) = exp(-hphi) * (1 + hphi);
            } else {
              if(twonu == 5){
                res(i, j) = (1 + hphi + hphi*hphi / 3.0) * exp(-hphi);
              }
            }
          }
        } else {
          res(i, j) = 1.0 + nugginside;
        }
      }
    }
  }
  return res;
}


//[[Rcpp::export]]
double Cov_matern_h(const double& h, 
                    const double& sigmasq,
                    const double& phi, const double& nu, const double& tausq){
  
  double hphi = h*phi;
  double pow2_nu1_gammanu = pow(2.0, 1.0-nu) / R::gammafn(nu);
  if(hphi > 0.0){
    return sigmasq * pow(hphi, nu) * pow2_nu1_gammanu *
      R::bessel_k(hphi, nu, 1.0);
  } else {
    return sigmasq + tausq;
  }
}

//[[Rcpp::export]]
double Cov_powexp_h(const double& h, 
                    const double& sigmasq,
                    const double& phi, const double& nu, const double& tausq){
  
  double hphi = h*phi;
  if(hphi > 0.0){
    return sigmasq * exp(-pow(hphi, nu)) + tausq;
  } else {
    return sigmasq + tausq;
  }
}


// gneiting 2002 eq. 15 with a,c,beta left unknown
//[[Rcpp::export]]
double gneiting2002_h(const double& h, const double& u,
                         const double& a, const double& c, const double& beta){
  double umod = 1.0 / (a * u + 1.0);
  return umod * exp(-c * h * pow(umod, beta/2.0) );
}
