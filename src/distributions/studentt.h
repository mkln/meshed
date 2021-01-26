#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>


inline double fwdcond_dmvt(const arma::mat& x, 
                           const arma::cube& Ri,
                           const arma::vec& parKxxpar, 
                           const arma::mat& Kcxpar,
                           const arma::vec& nu,
                           double num_par){
  // conditional of x | parents
  double result = 0;
  
  for(int j=0; j<x.n_cols; j++){
    double nustar = nu(j) + num_par;
    double n2 = x.n_rows + .0;
    double denom = nu(j) - 2;
    double numer = 0;
    
    arma::vec xcenter = x.col(j);
    if(parKxxpar(j) > -1){
      denom += parKxxpar(j);
      xcenter -= Kcxpar.col(j);
    }
    numer += arma::conv_to<double>::from( xcenter.t() * Ri.slice(j) * xcenter );
    result += - .5*(nustar + n2) * log1p( numer/denom );
  }

  return result;
}

inline arma::vec grad_fwdcond_dmvt(const arma::mat& x, 
                                   const arma::cube& Ri,
                                   const arma::vec& parKxxpar, 
                                   const arma::mat& Kcxpar,
                                   const arma::vec& nu,
                                   double num_par){
  
  // gradient of conditional of x | parents
  arma::mat result = arma::zeros(arma::size(x));
  int k = Ri.n_slices;
  for(int j=0; j<k; j++){
    double numer = nu(j) + x.n_rows + num_par;
    double denom = nu(j) - 2;
    
    arma::vec xcenter = x.col(j);
    arma::vec norm_grad = Ri.slice(j) * x.col(j);
    
    if(parKxxpar(j) > -1){
      denom += parKxxpar(j);
      norm_grad -= Ri.slice(j) * Kcxpar.col(j);
      xcenter -= Kcxpar.col(j);
    }
    denom += arma::conv_to<double>::from(xcenter.t() * norm_grad);
    
    result.col(j) = -numer/denom * norm_grad;
    
  }
  
  return arma::vectorise(result);
}

inline arma::mat neghess_fwdcond_dmvt(const arma::mat& x, 
                                   const arma::cube& Ri,
                                   const arma::vec& parKxxpar, 
                                   const arma::mat& Kcxpar,
                                   const arma::vec& nu,
                                   double num_par){
  // negative hessian of conditional of x | parents
  int k = Ri.n_slices;
  int nr = Ri.n_rows;
  int nc = Ri.n_cols;
  
  arma::mat result = arma::zeros(nr * k, nc * k);
  
  for(int j=0; j<k; j++){
    double a = nu(j) + x.n_rows + num_par;
    double b = nu(j) - 2;
    double denom = 0;
    arma::vec xcenter = x.col(j);
    arma::vec norm_grad = Ri.slice(j) * x.col(j);
    
    if(parKxxpar(j) > -1){
      b += parKxxpar(j);
      norm_grad -= Ri.slice(j) * Kcxpar.col(j);
      xcenter -= Kcxpar.col(j);
    }
    denom = b + arma::conv_to<double>::from(xcenter.t() * norm_grad);
    
    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) = 
      a/denom * Ri.slice(j) - (2.0*a)/(denom*denom) * norm_grad * norm_grad.t();
  }
  
  return result;
}

inline double bwdcond_dmvt(const arma::mat& x, 
                           const arma::mat& w_child,
                           const arma::cube& Ri_of_child,
                           const arma::cube& Kcx_x,
                           const arma::cube& Kxxi_x,
                           const arma::mat& Kxo_wo,
                           const arma::mat& Kco_wo,
                           const arma::vec& woKoowo,
                           const arma::vec& nu,
                           double num_par){
  // conditional of Y | x, others
  double result = 0;
  
  for(int j=0; j<x.n_cols; j++){
    double alpha = -0.5 *( nu(j) + w_child.n_rows + num_par ); // num_par = x.n_elem + otherparents.n_elem
    double denom = nu(j) - 2;
    double numer = 0;
    //double nresult = 0;
    
    if(woKoowo(j) > -1){
      denom += arma::conv_to<double>::from(x.col(j).t() * Kxxi_x.slice(j) * x.col(j) + 2 * x.col(j).t() * Kxo_wo.col(j) + woKoowo(j));
      numer += arma::conv_to<double>::from(
        (w_child.col(j) - Kcx_x.slice(j)*x.col(j) - Kco_wo.col(j)).t() * 
          Ri_of_child.slice(j) * (w_child.col(j) - Kcx_x.slice(j)*x.col(j) - Kco_wo.col(j)));
      
      //nresult = -.5*numer;
    } else {
      denom += arma::conv_to<double>::from(x.col(j).t() * Kxxi_x.slice(j) * x.col(j));
      numer += arma::conv_to<double>::from(
        (w_child.col(j) - Kcx_x.slice(j)*x.col(j)).t() * Ri_of_child.slice(j) * (w_child.col(j) - Kcx_x.slice(j)*x.col(j)));
      //nresult = -.5*numer;
    }
    
    double front_numer = denom;
    double front_denom = nu(j) - 2.0 + num_par;
    
    double front_component = - (w_child.n_rows+.0)/2.0 * log(.0+front_numer/front_denom);
    //Rcpp::Rcout << w_child.n_elem/2.0*log(front_numer/front_denom) << " " << front_component << "\n";
    //double nresult = -.5*numer;
    result += front_component + alpha * log1p( numer/denom );
  }
  return result;
}

inline arma::vec grad_bwdcond_dmvt(const arma::mat& x, 
                                   const arma::mat& w_child,
                                   const arma::cube& Ri_of_child,
                                   const arma::cube& Kcx_x,
                                   const arma::cube& Kxxi_x,
                                   const arma::mat& Kxo_wo,
                                   const arma::mat& Kco_wo,
                                   const arma::vec& woKoowo,
                                   const arma::vec& nu,
                                   double num_par){
  // gradient of conditional of Y | x, others
  
  arma::mat result = arma::zeros(arma::size(x));
  
  for(int j=0; j<x.n_cols; j++){
    
    double alpha = 0.5*(nu(j) + w_child.n_rows + num_par);
    double beta = nu(j) - 2;
    
    arma::vec ytilde = w_child.col(j) - Kcx_x.slice(j) * x.col(j);
    
    arma::vec mult = Kxxi_x.slice(j) * x.col(j);
    double denom_2 = arma::conv_to<double>::from(x.col(j).t() * mult);
    
    if(woKoowo(j) > -1){
      ytilde -= Kco_wo.col(j);
      beta += woKoowo(j);
      denom_2 += arma::conv_to<double>::from( 2*x.col(j).t()*Kxo_wo.col(j) );
      mult += Kxo_wo.col(j);
    }
    
    arma::vec mvn_gradient = Ri_of_child.slice(j) * ytilde;
    double numer = arma::conv_to<double>::from(ytilde.t() * mvn_gradient);
    double denom = beta + denom_2;
    mvn_gradient = Kcx_x.slice(j).t() * mvn_gradient;
    
    arma::vec front_term = -(w_child.n_rows+.0)/denom * mult;
    arma::vec result_j = 2*alpha*mvn_gradient/(denom + numer) + 2*alpha*numer/(denom*denom + denom*numer) * mult;
    result.col(j) = front_term + result_j;
  }
  return arma::vectorise(result);
}


inline arma::mat neghess_bwdcond_dmvt(const arma::mat& x,
                                   const arma::mat& w_child,
                                   const arma::cube& Ri_of_child,
                                   const arma::cube& Kcx_x,
                                   const arma::cube& Kxxi_x,
                                   const arma::mat& Kxo_wo,
                                   const arma::mat& Kco_wo,
                                   const arma::vec& woKoowo,
                                   const arma::vec& nu,
                                   double num_par){
  // gradient of conditional of Y | x, others
  int k = Ri_of_child.n_slices;
  int nr = x.n_rows;
  int nc = x.n_rows;

  arma::mat result = arma::zeros(nr * k, nc * k);

  for(int j=0; j<k; j++){

    double alpha = 0.5*(nu(j) + w_child.n_rows + num_par);
    double beta = nu(j) - 2;

    arma::vec ytilde = w_child.col(j) - Kcx_x.slice(j) * x.col(j);

    arma::vec mult = Kxxi_x.slice(j) * x.col(j);
    double denom_2 = arma::conv_to<double>::from(x.col(j).t() * mult);

    if(woKoowo(j) > -1){
      ytilde -= Kco_wo.col(j);
      beta += woKoowo(j);
      denom_2 += arma::conv_to<double>::from( 2*x.col(j).t()*Kxo_wo.col(j) );
      mult += Kxo_wo.col(j);
    }

    arma::vec mvn_gradient = Ri_of_child.slice(j) * ytilde;
    double numer = arma::conv_to<double>::from(ytilde.t() * mvn_gradient);
    double denom = beta + denom_2;
    mvn_gradient = Kcx_x.slice(j).t() * mvn_gradient;

    arma::vec front_term = -(w_child.n_rows+.0)/denom * mult;
    arma::vec result_j = 2*alpha*mvn_gradient/(denom + numer) + 2*alpha*numer/(denom*denom + denom*numer) * mult;
    
    
    double a = 2 * alpha;
    double t5 = denom + numer;
    arma::mat A = Kcx_x.slice(j);
    arma::mat R = Ri_of_child.slice(j);
    arma::mat C = Kxxi_x.slice(j);
    arma::vec d = Kxo_wo.col(j);

    arma::vec t7t9 = a/t5 * (mvn_gradient + numer/denom * mult);
    double t11 = a * numer / ( denom*denom * (denom + numer) );
    arma::mat t13 = ytilde.t() * R * A;


    result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) =
      - w_child.n_rows/denom * C + ( 2.0 * mult * mult.t()  / (denom * denom)  )
      +
      a * numer / (denom * t5) * C - (t7t9*(mult.t() - t13) + a/t5*A.t() * R * A
                                   + t11*mult*mult.t() + 2*a/(denom*(numer+denom)) * mult * t13);

  }
  return result;
}


// 
// 
// inline arma::mat neghess_bwdcond_dmvt(const arma::mat& x, 
//                                       const arma::mat& w_child,
//                                       const arma::cube& Ri_of_child,
//                                       const arma::cube& Kcx_x,
//                                       const arma::cube& Kxxi_x,
//                                       const arma::mat& Kxo_wo,
//                                       const arma::mat& Kco_wo,
//                                       const arma::vec& woKoowo,
//                                       const arma::vec& nu,
//                                       double num_par){
//   // gradient of conditional of Y | x, others
//   int k = Ri_of_child.n_slices;
//   int nr = x.n_rows;
//   int nc = x.n_rows;
//   
//   arma::mat result = arma::zeros(nr * k, nc * k);
//   
//   for(int j=0; j<k; j++){
//     
//     double alpha = 0.5*(nu(j) + w_child.n_rows + num_par);
//     double beta = nu(j) - 2;
//     
//     arma::vec ytilde = w_child.col(j) - Kcx_x.slice(j) * x.col(j);
//     
//     arma::vec mult = Kxxi_x.slice(j) * x.col(j);
//     double denom_2 = arma::conv_to<double>::from(x.col(j).t() * mult);
//     
//     if(woKoowo(j) > -1){
//       ytilde -= Kco_wo.col(j);
//       beta += woKoowo(j);
//       denom_2 += arma::conv_to<double>::from( 2*x.col(j).t()*Kxo_wo.col(j) );
//       mult += Kxo_wo.col(j);
//     }
//     
//     arma::vec mvn_gradient = Ri_of_child.slice(j) * ytilde;
//     double numer = arma::conv_to<double>::from(ytilde.t() * mvn_gradient);
//     double denom = beta + denom_2;
//     mvn_gradient = Kcx_x.slice(j).t() * mvn_gradient;
//     
//     // arma::vec front_term = -(w_child.n_rows+.0)/denom * mult;
//     // arma::vec result_j = 2*alpha*mvn_gradient/(denom + numer) + 2*alpha*numer/(denom*denom + denom*numer) * mult;
//     // result.col(j) = front_term + result_j;
//     // 
//     double a = 2 * alpha;
//     double t5 = denom + numer;
//     arma::mat A = Kcx_x.slice(j); 
//     arma::mat R = Ri_of_child.slice(j);
//     arma::mat C = Kxxi_x.slice(j);
//     arma::vec d = Kxo_wo.col(j);
//     
//     arma::vec t7t9 = a/t5 * (mvn_gradient + numer/denom * mult);
//     double t11 = a * numer / ( denom*denom * (denom + numer) );
//     //arma::mat t13 = ytilde.t() * R * A;
//     arma::mat RA = R*A;
//     arma::mat mmt = mult * mult.t();
//     
//     result.submat(nr*j, nc*j, (j+1)*nr-1, (j+1)*nc-1) = 
//       - w_child.n_rows/denom * C + ( 2.0 * mmt / (denom * denom)  )
//       + a * numer / (denom * t5) * C - 
//       (t7t9*(mult.t() - mvn_gradient.t()) + 
//       a/t5*A.t() * RA + t11*mmt + 2*a/(denom*(numer+denom)) * mult * mvn_gradient.t());
//   }
//   return result;
// }
