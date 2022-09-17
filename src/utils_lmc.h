#ifndef MGP_UTILS 
#define MGP_UTILS

#include "RcppArmadillo.h"


using namespace std;

struct MeshDataLMC {
  arma::mat theta; 
  arma::vec nu;
  
  // x coordinates here
  // p parents coordinates
  
  arma::field<arma::cube> CC_cache; // C(x,x)
  arma::field<arma::cube> Kxxi_cache; // Ci(x,x)
  arma::field<arma::cube> H_cache; // C(x,p) Ci(p,p)
  arma::field<arma::cube> Ri_cache; // ( C(x,x) - C(x,p)Ci(p,p)C(p,x) )^{-1}
  arma::field<arma::cube> Kppi_cache; // Ci(p,p)
  arma::vec Ri_chol_logdet;
  
  std::vector<arma::cube *> w_cond_prec_ptr;
  std::vector<arma::cube *> w_cond_mean_K_ptr;
  std::vector<arma::cube *> w_cond_prec_parents_ptr;
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::mat wcore; 
  arma::mat loglik_w_comps;
  
  arma::vec ll_y;
  
  double loglik_w; // will be pml
  double ll_y_all;
  
  arma::field<arma::cube> Hproject; // moves from w to observed coords
  arma::field<arma::cube> Rproject; // stores R for obs
  arma::field<arma::cube> Riproject;
  
  arma::cube DplusSi;
  arma::cube DplusSi_c;
  arma::vec DplusSi_ldet;
  
  // w cache
  arma::field<arma::mat> Sigi_chol;
  arma::field<arma::mat> Smu_start;
  
  arma::field<arma::field<arma::cube> > AK_uP;
  //arma::field<arma::field<arma::mat> > LambdaH_Ditau; // for forced grids;
};

inline bool compute_block(bool predicting, int block_ct, bool rfc){
  return predicting == false? (block_ct > 0) : predicting;;
}

inline arma::vec drowcol_uv(const arma::field<arma::uvec>& diag_blocks){
  int M=diag_blocks.n_elem;
  arma::vec drow = arma::zeros(M+1);
  for(int i=0; i<M; i++){
    drow(i+1) = diag_blocks(i).n_rows;
  }
  drow = arma::cumsum(drow);
  return drow;
}

inline arma::uvec field_v_concat_uv(arma::field<arma::uvec> const& fuv){
  // takes a field of matrices (same n cols) and outputs a single matrix concatenating all
  arma::vec ddims = drowcol_uv(fuv);
  arma::uvec result = arma::zeros<arma::uvec>(ddims(fuv.n_elem));
  for(unsigned int j=0; j<fuv.n_elem; j++){
    if(fuv(j).n_elem>0){
      result.rows(ddims(j), ddims(j+1)-1) = fuv(j);
    }
  }
  return result;
}

inline void block_invcholesky_(arma::mat& X, const arma::uvec& upleft_cumblocksizes){
  // inplace
  // this function computes inv(chol(X)) when 
  // X is a block matrix in which the upper left block itself is block diagonal
  // the cumulative block sizes of the blockdiagonal block of X are specified in upleft_cumblocksizes
  // the size of upleft_cumblocksizes is 1+number of blocks, and its first element is 0
  // --
  // the function proceeds by computing the upper left inverse cholesky
  // ends with the schur matrix for the lower right block of the result
  
  int upperleft_nblocks = upleft_cumblocksizes.n_elem - 1;
  int n_upleft = upleft_cumblocksizes(upleft_cumblocksizes.n_elem-1);
  
  X.submat(0, n_upleft, n_upleft-1, X.n_cols-1).fill(0); // upper-right block-corner erased
  
  arma::mat B = X.submat(n_upleft, 0, X.n_rows-1, n_upleft-1);
  arma::mat D = X.submat(n_upleft, n_upleft, X.n_rows-1, X.n_rows-1);
  arma::mat BLAinvt = arma::zeros(B.n_rows, n_upleft);
  arma::mat BLAinvtLAinv = arma::zeros(B.n_rows, n_upleft);
  
  for(int i=0; i<upperleft_nblocks; i++){
    arma::mat LAinv = arma::inv(arma::trimatl(arma::chol(
      arma::symmatu(X.submat(upleft_cumblocksizes(i), upleft_cumblocksizes(i), 
                             upleft_cumblocksizes(i+1) - 1, upleft_cumblocksizes(i+1) - 1)), "lower")));
  
    X.submat(upleft_cumblocksizes(i), upleft_cumblocksizes(i), 
             upleft_cumblocksizes(i+1) - 1, upleft_cumblocksizes(i+1) - 1) = LAinv;
    arma::mat BLAinvt_temp = B.cols(upleft_cumblocksizes(i), upleft_cumblocksizes(i+1) - 1) * LAinv.t();
    BLAinvt.cols(upleft_cumblocksizes(i), upleft_cumblocksizes(i+1) - 1) = BLAinvt_temp;
    BLAinvtLAinv.cols(upleft_cumblocksizes(i), upleft_cumblocksizes(i+1) - 1) = BLAinvt_temp * LAinv;
  }
  
  arma::mat invcholSchur = arma::inv(arma::trimatl(arma::chol(
    arma::symmatu(D - BLAinvt * BLAinvt.t()), "lower")));
  X.submat(n_upleft, 0, X.n_rows-1, n_upleft-1) = - invcholSchur * BLAinvtLAinv;
  X.submat(n_upleft, n_upleft, X.n_rows-1, X.n_rows-1) = invcholSchur;

}

inline void add_smu_parents_(arma::mat& result, 
                      const arma::cube& condprec,
                      const arma::cube& cmk,
                      const arma::mat& wparents){
  int n_blocks = condprec.n_slices;
  int bsize = condprec.n_rows;
  for(int i=0; i<n_blocks; i++){
    //result.rows(blockdims(i), blockdims(i+1)-1) +=
    result.rows(i*bsize, (i+1)*bsize-1) +=
      condprec.slice(i) * cmk.slice(i) * wparents.col(i);
  }
}

inline void add_smu_parents_ptr_(arma::mat& result, 
                          const arma::cube* condprec,
                          const arma::cube* cmk,
                          const arma::mat& wparents){
  int n_blocks = (*condprec).n_slices;
  int bsize = (*condprec).n_rows;
  for(int i=0; i<n_blocks; i++){
    //result.rows(blockdims(i), blockdims(i+1)-1) +=
    result.rows(i*bsize, (i+1)*bsize-1) +=
      (*condprec).slice(i) * (*cmk).slice(i) * wparents.col(i);
  }
}


inline arma::cube AKuT_x_R(arma::cube& result, const arma::cube& x, const arma::cube& y){ 
  //arma::cube result = arma::zeros(x.n_cols, y.n_cols, x.n_slices);
  for(unsigned int i=0; i<x.n_slices; i++){
    result.slice(i) = arma::trans(x.slice(i)) * y.slice(i); 
  }
  return result;
}

inline arma::cube AKuT_x_R_ptr(arma::cube& result, const arma::cube& x, const arma::cube* y){ 
  //arma::cube result = arma::zeros(x.n_cols, y.n_cols, x.n_slices);
  for(unsigned int i=0; i<x.n_slices; i++){
    result.slice(i) = arma::trans(x.slice(i)) * (*y).slice(i); 
  }
  return result;
}

inline void add_AK_AKu_multiply_(arma::mat& result,
                          const arma::cube& x, const arma::cube& y){
  int n_blocks = x.n_slices;
  int bsize = x.n_rows;
  for(int i=0; i<n_blocks; i++){
    //result.submat(outerdims(i), outerdims(i), 
    //              outerdims(i+1)-1, outerdims(i+1)-1) +=
    result.submat(i*bsize, i*bsize, (i+1)*bsize-1, (i+1)*bsize-1) += 
      x.slice(i) * y.slice(i);
  }
}

inline arma::mat cube_times_mat(const arma::cube& x, const arma::mat& y){ 
  arma::mat result = arma::zeros(x.n_rows, y.n_cols);
  int n_blocks = x.n_slices;
  for(int i=0; i<n_blocks; i++){
    result.col(i) = x.slice(i) * y.col(i);
  }
  return result;
}

inline arma::mat AK_vec_multiply(const arma::cube& x, const arma::mat& y){ 
  arma::mat result = arma::zeros(x.n_rows, y.n_cols);
  int n_blocks = x.n_slices;
  for(int i=0; i<n_blocks; i++){
    result.col(i) = x.slice(i) * y.col(i);
  }
  return result;
}

inline void add_lambda_crossprod(arma::mat& result, const arma::mat& X, int j, int q, int k, int blocksize){
  // X has blocksize rows and k*blocksize columns
  // we want to output X.t() * X
  //arma::mat result = arma::zeros(X.n_cols, X.n_cols);
  
  // WARNING: this does NOT update result to the full crossproduct,
  // but ONLY the lower-blocktriangular part!
  int kstar = k-q;
  arma::uvec lambda_nnz = arma::zeros<arma::uvec>(1+kstar);
  lambda_nnz(0) = j;
  for(int i=0; i<kstar; i++){
    lambda_nnz(i+1) = q + i;
  }
  for(unsigned int h=0; h<lambda_nnz.n_elem; h++){
    int indh = lambda_nnz(h);
    for(unsigned int i=h; i<lambda_nnz.n_elem; i++){
      int indi = lambda_nnz(i);
      result.submat(indi*blocksize, indh*blocksize, (indi+1)*blocksize-1, (indh+1)*blocksize-1) +=
        X.cols(indi*blocksize, (indi+1)*blocksize-1).t() * X.cols(indh*blocksize, (indh+1)*blocksize-1);
    }
  }
}

inline void add_LtLxD(arma::mat& result, const arma::mat& LjtLj, const arma::vec& Ddiagvec){
  // computes LjtLj %x% diag(Ddiagvec)
  //arma::mat result = arma::zeros(LjtLj.n_rows * Ddiagvec.n_elem, LjtLj.n_cols * Ddiagvec.n_elem);
  int blockx = Ddiagvec.n_elem;
  int blocky = Ddiagvec.n_elem;
  
  for(unsigned int i=0; i<LjtLj.n_rows; i++){
    int startrow = i*blockx;
    for(unsigned int j=0; j<LjtLj.n_cols; j++){
      int startcol = j*blocky;
      if(LjtLj(i,j) != 0){
        for(unsigned int h=0; h<Ddiagvec.n_elem; h++){
          if(Ddiagvec(h) != 0){
            result(startrow + h, startcol+h) += LjtLj(i, j) * Ddiagvec(h);
          }
        }
      }
    }
  }
}

inline arma::mat build_block_diagonal_ptr(const arma::cube* x){
  int nrow = (*x).n_rows;
  int nslice = (*x).n_slices;
  arma::mat result = arma::zeros(nrow * nslice, nrow * nslice);
  for(int i=0; i<nslice; i++){
    //result.submat(blockdims(i), blockdims(i), blockdims(i+1)-1, blockdims(i+1)-1) = x.slice(i);
    result.submat(i*nrow, i*nrow, (i+1)*nrow-1, (i+1)*nrow-1) = (*x).slice(i);
  }
  return result;
}

inline arma::mat build_block_diagonal(const arma::cube& x){
  int nrow = x.n_rows;
  int nslice = x.n_slices;
  arma::mat result = arma::zeros(nrow * nslice, nrow * nslice);
  for(int i=0; i<nslice; i++){
    //result.submat(blockdims(i), blockdims(i), blockdims(i+1)-1, blockdims(i+1)-1) = x.slice(i);
    result.submat(i*nrow, i*nrow, (i+1)*nrow-1, (i+1)*nrow-1) = x.slice(i);
  }
  return result;
}


inline arma::cube cube_cols(const arma::cube& x, const arma::uvec& sel_cols){
  arma::cube result = arma::zeros(x.n_rows, sel_cols.n_elem, x.n_slices);
  for(unsigned int i=0; i<x.n_slices; i++){
    result.slice(i) = x.slice(i).cols(sel_cols);
  }
  return result;
}

inline arma::cube cube_cols_ptr(const arma::cube* x, const arma::uvec& sel_cols){
  arma::cube result = arma::zeros((*x).n_rows, sel_cols.n_elem, (*x).n_slices);
  for(unsigned int i=0; i<(*x).n_slices; i++){
    result.slice(i) = (*x).slice(i).cols(sel_cols);
  }
  return result;
}

inline arma::cube subcube_ptr(const arma::cube* x, const arma::uvec& sel_rows, const arma::uvec& sel_cols){
  arma::cube result = arma::zeros(sel_rows.n_elem, sel_cols.n_elem, (*x).n_slices);
  for(unsigned int i=0; i<(*x).n_slices; i++){
    result.slice(i) = (*x).slice(i).submat(sel_rows, sel_cols);
  }
  return result;
}

inline arma::mat ortho(const arma::mat& x){
  return arma::eye(arma::size(x)) - x * arma::inv_sympd(arma::symmatu(x.t() * x)) * x.t();
}


#endif
