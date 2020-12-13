#include "covariance_lmc.h"

using namespace std;

arma::mat CmaternInv(const arma::mat& x,
                     const double& sigmasq,
                     const double& phi, const double& nu, 
                     const double& tausq){

  double pow2_nu1_gammanu = pow(2.0, 1.0-nu) / R::gammafn(nu);
  arma::mat res = arma::zeros(x.n_rows, x.n_rows);
  for(int i = 0; i < x.n_rows; i++){
    arma::rowvec cri = x.row(i);
    for(int j = i; j < x.n_rows; j++){
      arma::rowvec delta = cri - x.row(j);
      double hphi = arma::norm(delta) * phi;
      if(hphi > 0.0){
        res(i, j) = sigmasq * pow(hphi, nu) * pow2_nu1_gammanu *
          R::bessel_k(hphi, nu, 1.0);
      } else {
        res(i, j) = sigmasq + tausq;
      }
    }
  }
  res = arma::symmatu(res);
  return res;
}


// matern
arma::mat matern_internal(const arma::mat& x, const arma::mat& y, const double& phi, const double& nu, 
                 double * bessel_ws, const double& nugginside=1e-7,  bool same=false){
  
  int threadid;
#ifdef _OPENMP
  threadid = omp_get_thread_num();
#endif
  
  int bessel_ws_inc = 3; // nu+1 // increase?
  double pow2_nu1_gammanu = pow(2.0, 1.0-nu) / R::gammafn(nu);
  
  arma::mat res = arma::zeros(x.n_rows, y.n_rows);
  if(same){
    for(int i=0; i<x.n_rows; i++){
      arma::rowvec cri = x.row(i);
      for(int j=i; j<y.n_rows; j++){
        arma::rowvec delta = cri - y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          res(i, j) = pow(hphi, nu) * pow2_nu1_gammanu *
            R::bessel_k_ex(hphi, nu, 1.0, &bessel_ws[threadid*bessel_ws_inc]);
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
          res(i, j) = pow(hphi, nu) * pow2_nu1_gammanu *
            R::bessel_k_ex(hphi, nu, 1.0, &bessel_ws[threadid*bessel_ws_inc]);
        } else {
          res(i, j) = 1.0 + nugginside;
        }
      }
    }
  }
  return res;
}

/*
// matern
arma::mat matern(const arma::mat& x, const arma::mat& y, const double& phi, const double& nu, 
                 double * bessel_ws,  bool same){
  
  int threadid;
#ifdef _OPENMP
  threadid = omp_get_thread_num();
#endif
  
  arma::mat D;
  if(same){
    arma::mat pmag = arma::sum(x % x, 1);
    int np = x.n_rows;
    D = sqrt(abs(arma::repmat(pmag.t(), np, 1) + arma::repmat(pmag, 1, np) - 2 * x * x.t()));
  } else {
    arma::mat pmag = arma::sum(x % x, 1);
    arma::mat qmag = arma::sum(y % y, 1);
    int np = x.n_rows;
    int nq = y.n_rows;
    D = sqrt(abs(arma::repmat(qmag.t(), np, 1) + arma::repmat(pmag, 1, nq) - 2 * x * y.t()));
  }
  
  arma::mat R = arma::zeros(arma::size(D));
  int bessel_ws_inc = 5; // nu+1 // increase?
  
  //create the correlation matrix (now thread-safe)
  for(int i = 0; i < D.n_rows; i++){
    for(int j = 0; j < D.n_cols; j++){
      if(D(i, j)*phi > 0.0){
        R(i, j) = pow(D(i,j)*phi, nu)/(pow(2, nu-1)*R::gammafn(nu))*
          R::bessel_k_ex(D(i,j)*phi, nu, 1.0, &bessel_ws[threadid*bessel_ws_inc]);
      } else {
        R(i, j) = 1.0;
      }
    }
  }
  
  return R;
}*/

// matern covariance with nu = p + 1/2, and p=0,1,2
arma::mat matern_halfint(const arma::mat& x, const arma::mat& y, const double& phi, bool same, int twonu){
  // 0 based indexing
  arma::mat res = arma::zeros(x.n_rows, y.n_rows);
  double nugginside = 0;//1e-7;
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
  
  
  /*
  if(same){
    arma::mat pmag = arma::sum(x % x, 1);
    int np = x.n_rows;
    D = sqrt(abs(arma::repmat(pmag.t(), np, 1) + arma::repmat(pmag, 1, np) - 2 * x * x.t()));
  } else {
    arma::mat pmag = arma::sum(x % x, 1);
    arma::mat qmag = arma::sum(y % y, 1);
    int np = x.n_rows;
    int nq = y.n_rows;
    D = sqrt(abs(arma::repmat(qmag.t(), np, 1) + arma::repmat(pmag, 1, nq) - 2 * x * y.t()));
  }
  
  if(twonu == 1){ // nu = 1/2, exponential covariance
    return exp(-phi * D);
  }
  if(twonu == 3){ // nu = 3/2
    arma::mat Dstar = phi * D;
    Dstar = (1 + Dstar) % exp(-Dstar);
    if(same){
      Dstar.diag() += 1e-7;
    }
    return Dstar;
  }
  if(twonu == 5){ // nu = 5/2
    arma::mat Dstar = phi * D;
    Dstar = (1 + Dstar + pow(Dstar,2.0) / 3.0) % exp(-Dstar);
    if(same){
      Dstar.diag() += 1e-7;
    }
    return Dstar;
  }*/
}


// powered exponential nu<2
arma::mat powerexp(const arma::mat& x, const arma::mat& y, const double& phi, const double& nu, bool same){
  arma::mat res = arma::zeros(x.n_rows, y.n_rows);
  double nugginside = 0;//1e-9;
  if(same){
    for(int i=0; i<x.n_rows; i++){
      arma::rowvec cri = x.row(i);
      for(int j=i; j<y.n_rows; j++){
        arma::rowvec delta = cri - y.row(j);
        double hnuphi = pow(arma::norm(delta), nu) * phi;
        if(hnuphi > 0.0){
          res(i, j) = exp(-hnuphi);
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
        double hnuphi = pow(arma::norm(delta), nu) * phi;
        if(hnuphi > 0.0){
          res(i, j) = exp(-hnuphi);
        } else {
          res(i, j) = 1.0 + nugginside;
        }
      }
    }
  }
  return res;
}


// gneiting 2002 eq. 15 with a,c,beta left unknown
arma::mat gneiting2002(const arma::mat& x, const arma::mat& y, 
                       const double& a, const double& c, const double& beta, bool same){
  // NOT reparametrized here
  arma::mat xH = x.cols(0, 1);
  arma::mat xU = x.col(2);
  arma::mat yH = y.cols(0, 1);
  arma::mat yU = y.col(2);
  arma::mat H, U;
  if(same){
    arma::mat pmagH = arma::sum(xH % xH, 1);
    int np = x.n_rows;
    H = sqrt(abs(arma::repmat(pmagH.t(), np, 1) + arma::repmat(pmagH, 1, np) - 2 * xH * xH.t()));
    U = abs(arma::repmat(xU.t(), np, 1) - arma::repmat(xU, 1, np));
  } else {
    arma::mat pmagH = arma::sum(xH % xH, 1);
    arma::mat qmagH = arma::sum(yH % yH, 1);
    int np = x.n_rows;
    int nq = y.n_rows;
    H = sqrt(abs(arma::repmat(qmagH.t(), np, 1) + arma::repmat(pmagH, 1, nq) - 2 * xH * yH.t()));
    U = abs(arma::repmat(yU.t(), np, 1) - arma::repmat(xU, 1, nq));
  }
  arma::mat Umod = 1.0/(a*U+1);
  return Umod % exp(-c*H % pow(Umod, beta/2.0));
  //arma::mat Umod = 1.0/(U+1.0/a);
  //return Umod % exp(-c*H/a % pow(Umod, beta/2.0))/c;
}

arma::mat Correlationf(const arma::mat& x, const arma::mat& y, 
                       const arma::vec& theta,
                       MaternParams& matern, bool same){
  // these are not actually correlation functions because they are reparametrized to have 
  // C(0) = 1/spatial decay
  if(x.n_cols == 2){
    // spatial matern
    // reparametrized here
    if(theta.n_rows == 2){
      // exponential
      double phi = theta(0);
      double sigmasq = theta(1);
      
      int nutimes2 = matern.twonu;
      double reparam = pow(phi, .0 + nutimes2);
      
      return sigmasq * 
        matern_halfint(x, y, phi, same, nutimes2)/reparam;
    } else {
      double phi = theta(0);
      double nu = theta(1);
      double sigmasq = theta(2);
      
      double reparam = pow(phi, 2*nu);
      
      double nugginside = 0;//1e-7;
      
      //return powerexp(x, y, phi, nu, same)/phi;
      // we divide by phi given the equivalence in 
      // zhang 2004, corrollary to Thm.2: 
      return sigmasq * 
        matern_internal(x, y, phi, nu, matern.bessel_ws, nugginside, same)/reparam;
    }
    //return squaredexp(x, y, theta(0), same)/theta(0);
  } else {
    // theta 0: temporal decay, 
    // theta 1: spatial decay,
    // theta 2: separability
    // reparametrized here
    return gneiting2002(x, y, theta(0), theta(1), theta(2), same)/theta(1);
  }
}


arma::mat CviaKron(const arma::mat& coords, 
                   const arma::uvec& indx, const arma::uvec& indy,
                   int k, const arma::mat& theta, MaternParams& matern, bool same){
  arma::mat res = arma::zeros(indx.n_rows * k, indy.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat CC = Correlationf(coordsx, coordsy, theta.col(j), matern, same);
    res += arma::kron(arma::diagmat(Iselect.col(j)), CC);
  }
  return res;
}


void CviaKron_invsympd_(arma::cube& CCi, 
                        const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta, MaternParams& matern){
  arma::mat coordsx = coords.rows(indx);
  for(int j=0; j<k; j++){
    CCi.slice(j) = arma::inv_sympd( Correlationf(coordsx, coordsx, theta.col(j), matern, true) );
  }
}


arma::mat CviaKron_chol(const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta, MaternParams& matern){
  arma::mat res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  
  for(int j=0; j<k; j++){
    arma::mat CC_chol = arma::chol( arma::symmatu( 
      Correlationf(coordsx, coordsx, theta.col(j), matern, true) ), "lower");
    res += arma::kron(arma::diagmat(Iselect.col(j)), CC_chol);
  }
  return res;
}


double CviaKron_invchol(arma::mat& res,
                        const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta, MaternParams& matern){
  res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  double logdet = 0;
  for(int j=0; j<k; j++){
    arma::mat CC_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( 
      Correlationf(coordsx, coordsx, theta.col(j), matern, true) ), "lower")));
    res += arma::kron(arma::diagmat(Iselect.col(j)), CC_chol);
    logdet += arma::accu(log(CC_chol.diag()));
  }
  return logdet;
}


arma::mat CviaKron_H(const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta, MaternParams& matern){
  arma::mat res = arma::zeros(indx.n_rows * k, indy.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), matern, false);
    arma::mat Cyy_i = arma::inv_sympd( arma::symmatu(
      Correlationf(coordsy, coordsy, theta.col(j), matern, true)) );
    arma::mat H = Cxy * Cyy_i;
    res += arma::kron(arma::diagmat(Iselect.col(j)), H);
  }
  return res;
}


arma::mat CviaKron_R(const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta, MaternParams& matern){
  arma::mat res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), matern, true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), matern, false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), matern, true) );
    arma::mat R = Cxx - Cxy * Cyy_i * Cxy.t();
    res += arma::kron(arma::diagmat(Iselect.col(j)), R);
  }
  return res;
}

void CviaKron_HRj_bdiag(
    arma::field<arma::mat>& Hj, arma::field<arma::mat>& Rj, 
    const arma::field<arma::mat>& Kxxi_cache,
    const arma::mat& coords, const arma::uvec& indx, 
    const arma::uvec& naix, const arma::uvec& indy, 
    int k, const arma::mat& theta, MaternParams& matern){
  
  Rj = arma::field<arma::mat> (indx.n_elem);
  Hj = arma::field<arma::mat> (indx.n_elem);
  
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int ix=0; ix<indx.n_rows; ix++){
    Rj(ix) = arma::zeros(k, k);
    Hj(ix) = arma::zeros(k, k*indy.n_rows);
  }
  for(int j=0; j<k; j++){
    arma::mat Cyy_i = Kxxi_cache(j);// arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    for(int ix=0; ix<indx.n_rows; ix++){
      if(naix(ix) == 1){
        arma::mat Cxx = Correlationf(coordsx.row(ix), coordsx.row(ix), theta.col(j), matern, true);
        arma::mat Cxy = Correlationf(coordsx.row(ix), coordsy, theta.col(j), matern, false);
        arma::mat Hloc = Cxy * Cyy_i;
        Hj(ix) += arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
        arma::mat R = Cxx - Hloc * Cxy.t();
        Rj(ix) += arma::kron(arma::diagmat(Iselect.col(j)), R);
      }
    }
  }
  
}


void CviaKron_HRj_chol_bdiag(
    arma::cube& Hj, arma::mat& Rjchol, arma::cube& Kxxi,
    const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta, MaternParams& matern){
  
  //arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  
  Kxxi = arma::zeros(indy.n_elem, indy.n_elem, k);
  for(int j=0; j<k; j++){
    Kxxi.slice(j) = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), matern, true) );
    for(int ix=0; ix<indx.n_rows; ix++){
      if(naix(ix) == 0){ // otherwise it's not missing
        arma::mat Cxx = Correlationf(coordsx.row(ix), coordsx.row(ix), theta.col(j), matern, true);
        arma::mat Cxy = Correlationf(coordsx.row(ix), coordsy, theta.col(j), matern, false);
        arma::mat Hloc = Cxy * Kxxi.slice(j);
        
        Hj.slice(ix).row(j) = Hloc;//+=arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
        double Rcholtemp = arma::conv_to<double>::from(
          Cxx - Hloc * Cxy.t() );
        Rcholtemp = Rcholtemp < 0 ? 0.0 : Rcholtemp;
        Rjchol(j, ix) = pow(Rcholtemp, .5); // 0 could be numerically negative
      }
      
    }
  }
}

void CviaKron_HRj_chol_bdiag_wcache(
    arma::cube& Hj, arma::mat& Rjchol, 
    const arma::cube& Kxxi_cache, const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta, MaternParams& matern){
  
  //arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  
  for(int j=0; j<k; j++){
    arma::mat Cyy_i = Kxxi_cache.slice(j);//
    //arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), true) );
    
    //Rcpp::Rcout << "difference calculated for pred and cached " 
    //            << abs(arma::accu(Cyy_i - Kxxi_cache.slice(j))) 
    //            << " j: " << j << endl;
    
    for(int ix=0; ix<indx.n_rows; ix++){
      if(naix(ix) == 0){ // otherwise it's not missing
        arma::mat Cxx = Correlationf(coordsx.row(ix), coordsx.row(ix), theta.col(j), matern, true);
        arma::mat Cxy = Correlationf(coordsx.row(ix), coordsy, theta.col(j), matern, false);
        arma::mat Hloc = Cxy * Cyy_i;
        
        Hj.slice(ix).row(j) = Hloc;//+=arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
        double Rcholtemp = arma::conv_to<double>::from(
          Cxx - Hloc * Cxy.t() );
        Rcholtemp = Rcholtemp < 0 ? 0.0 : Rcholtemp;
        Rjchol(j,ix) = pow(Rcholtemp, .5); // 0 could be numerically negative
      }
    }
  }
}

double CviaKron_HRi(arma::mat& H, arma::mat& Ri,
                    const arma::mat& coords, 
                    const arma::uvec& indx, const arma::uvec& indy, 
                    int k, const arma::mat& theta, MaternParams& matern){
  H = arma::zeros(indx.n_rows * k, indy.n_rows * k);
  Ri = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  double logdet=0;
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), matern, true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), matern, false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), matern, true) );
    arma::mat Hloc = Cxy * Cyy_i;
    arma::mat Rloc_ichol = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
      Cxx - Hloc * Cxy.t()) , "lower")));
    logdet += arma::accu(log(Rloc_ichol.diag()));
    H += arma::kron(arma::diagmat(Iselect.col(j)), Hloc);
    Ri += arma::kron(arma::diagmat(Iselect.col(j)), Rloc_ichol.t() * Rloc_ichol);
  }
  return logdet;
}


arma::mat CviaKron_Ri(const arma::mat& coords, 
                      const arma::uvec& indx, const arma::uvec& indy,  
                      int k, const arma::mat& theta, MaternParams& matern){
  arma::mat res = arma::zeros(indx.n_rows * k, indy.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), matern, true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), matern, false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), matern, true) );
    arma::mat Ri = arma::inv_sympd( Cxx - Cxy * Cyy_i * Cxy.t() );
    res += arma::kron(arma::diagmat(Iselect.col(j)), Ri);
  }
  return res;
}


arma::mat CviaKron_Rchol(const arma::mat& coords, 
                         const arma::uvec& indx, const arma::uvec& indy,  
                         int k, const arma::mat& theta, MaternParams& matern){
  arma::mat res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), matern, true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), matern, false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), matern, true) );
    arma::mat Rchol = arma::chol(arma::symmatu( Cxx - Cxy * Cyy_i * Cxy.t() ), "lower");
    res += arma::kron(arma::diagmat(Iselect.col(j)), Rchol);
  }
  return res;
}


arma::mat CviaKron_Rcholinv(const arma::mat& coords, 
                            const arma::uvec& indx, const arma::uvec& indy,  
                            int k, const arma::mat& theta, MaternParams& matern){
  arma::mat res = arma::zeros(indx.n_rows * k, indx.n_rows * k);
  arma::mat Iselect = arma::eye(k, k);
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), matern, true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), matern, false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), matern, true) );
    arma::mat Rcholinv = arma::inv(arma::trimatl(
      arma::chol(arma::symmatu( Cxx - Cxy * Cyy_i * Cxy.t() ), "lower")));
    res += arma::kron(arma::diagmat(Iselect.col(j)), Rcholinv);
  }
  return res;
}


// inplace functions


double CviaKron_HRi_(arma::cube& H, arma::cube& Ri, 
                     const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta, MaternParams& matern){
  // inplace version of CviaKron_HRi
  int dimx = indx.n_elem;
  int dimy = indy.n_elem;
  
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  double logdet=0;
  for(int j=0; j<k; j++){
    arma::mat Cxx = Correlationf(coordsx, coordsx, theta.col(j), matern, true);
    arma::mat Cxy = Correlationf(coordsx, coordsy, theta.col(j), matern, false);
    arma::mat Cyy_i = arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), matern, true) );
    arma::mat Hloc = Cxy * Cyy_i;
    arma::mat Rloc_ichol = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
      Cxx - Hloc * Cxy.t()) , "lower")));
    logdet += arma::accu(log(Rloc_ichol.diag()));
    
    int firstrow = j*dimx;
    int lastrow = (j+1)*dimx-1;
    int firstcol = j*dimy;
    int lastcol = (j+1)*dimy-1;
    if(indy.n_elem > 0){
      H.slice(j) = Hloc;//H.submat(firstrow, firstcol, lastrow, lastcol) = Hloc;
    }
    //Richol.slice(j) = Rloc_ichol;
    Ri.slice(j) = Rloc_ichol.t() * Rloc_ichol;//Ri.submat(firstrow, firstrow, lastrow, lastrow) = Rloc_ichol.t() * Rloc_ichol; // symmetric
  }
  return logdet;
}

double CviaKron_invsympd_wdet_(arma::cube& res,
                               const arma::mat& coords, const arma::uvec& indx, 
                               int k, const arma::mat& theta, MaternParams& matern){
  // inplace
  int dimx = indx.n_elem;
  arma::mat coordsx = coords.rows(indx);
  double logdet = 0;
  for(int j=0; j<k; j++){
    arma::mat CC_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( 
      Correlationf(coordsx, coordsx, theta.col(j), matern, true) ), "lower")));
    int firstrow = j*dimx;
    int lastrow = (j+1)*dimx-1;
    res.slice(j) = CC_chol.t()*CC_chol;//res.submat(firstrow, firstrow, lastrow, lastrow) = CC_chol.t()*CC_chol;
    logdet += arma::accu(log(CC_chol.diag()));
  }
  return logdet;
}

void CviaKron_HRj_bdiag_(
    arma::cube& Hj, arma::cube& Rj, 
    const arma::cube& Kxxi_cache,
    const arma::mat& coords, const arma::uvec& indx, 
    const arma::uvec& naix, const arma::uvec& indy, 
    int k, const arma::mat& theta, MaternParams& matern){
  //Rcpp::Rcout << "CviaKron_HRj_bdiag_ " << endl;
  //Rcpp::Rcout << indx.n_elem << " " << indy.n_elem << " k=" << k << endl;
  // inplace version of CviaKron_HRj_bdiag
  arma::mat coordsx = coords.rows(indx);
  arma::mat coordsy = coords.rows(indy);
  int dimx = indx.n_elem;
  int dimy = indy.n_elem;
  for(int j=0; j<k; j++){
    arma::mat Cyy_i = Kxxi_cache.slice(j);// 
      //arma::inv_sympd( Correlationf(coordsy, coordsy, theta.col(j), matern, true) );
    
    int firstcol = j*dimy;
    int lastcol = (j+1)*dimy-1;
    for(int ix=0; ix<indx.n_rows; ix++){
      if(naix(ix) == 1){
        arma::mat Cxx = Correlationf(coordsx.row(ix), coordsx.row(ix), theta.col(j), matern, true);
        arma::mat Cxy = Correlationf(coordsx.row(ix), coordsy, theta.col(j), matern, false);
        arma::mat Hloc = Cxy * Cyy_i;
        arma::mat R = Cxx - Hloc * Cxy.t();
        
        Hj.subcube(j, 0, ix, j, Hj.n_cols-1, ix) = Hloc;
        Rj(j, j, ix) = R(0,0) < 0 ? 0.0 : R(0,0);
      }
    }
  }
}


