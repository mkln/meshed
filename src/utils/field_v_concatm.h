#include <RcppArmadillo.h>



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
  for(int j=0; j<fuv.n_elem; j++){
    if(fuv(j).n_elem>0){
      result.rows(ddims(j), ddims(j+1)-1) = fuv(j);
    }
  }
  return result;
}

inline arma::vec drowcol_v(const arma::field<arma::vec>& diag_blocks){
  int M=diag_blocks.n_elem;
  arma::vec drow = arma::zeros(M+1);
  for(int i=0; i<M; i++){
    drow(i+1) = diag_blocks(i).n_rows;
  }
  drow = arma::cumsum(drow);
  return drow;
}

inline arma::vec field_v_concatv(arma::field<arma::vec> const& fuv){
  // takes a field of matrices (same n cols) and outputs a single matrix concatenating all
  arma::vec ddims = drowcol_v(fuv);
  arma::vec result = arma::zeros<arma::vec>(ddims(fuv.n_elem));
  for(int j=0; j<fuv.n_elem; j++){
    if(fuv(j).n_elem>0){
      result.rows(ddims(j), ddims(j+1)-1) = fuv(j);
    }
  }
  return result;
}


inline arma::vec drowcol_s(const arma::field<arma::mat>& diag_blocks){
  int M=diag_blocks.n_elem;
  arma::vec drow = arma::zeros(M+1);
  for(int i=0; i<M; i++){
    drow(i+1) = diag_blocks(i).n_rows;
  }
  drow = arma::cumsum(drow);
  return drow;
}

inline arma::mat field_v_concatm(arma::field<arma::mat> const& fieldmats){
  // takes a field of matrices (same n cols) and outputs a single matrix concatenating all
  arma::vec ddims = drowcol_s(fieldmats);
  arma::mat result = arma::zeros(ddims(fieldmats.n_elem), fieldmats(0).n_cols);
  //#pragma omp parallel for //**
  for(int j=0; j<fieldmats.n_elem; j++){
    if(fieldmats(j).n_rows>0){
      result.rows(ddims(j), ddims(j+1)-1) = fieldmats(j);
    }
  }
  return result;
}


inline arma::mat field_v_concatm_s(arma::field<arma::mat> const& fieldmats){
  // takes a field of matrices (same n cols) and outputs a single matrix concatenating all
  arma::vec ddims = drowcol_s(fieldmats);
  arma::mat result = arma::zeros(ddims(fieldmats.n_elem), fieldmats(0).n_cols);
  
  for(int j=0; j<fieldmats.n_elem; j++){
    if(fieldmats(j).n_rows>0){
      result.rows(ddims(j), ddims(j+1)-1) = fieldmats(j);
    }
  }
  return result;
}

inline void field_v_concatm_r(arma::mat& result, arma::field<arma::mat> const& fieldmats){
  // takes a field of matrices (same n cols) and outputs a single matrix concatenating all
  arma::vec ddims = drowcol_s(fieldmats);
  //#pragma omp parallel for //**
  for(int j=0; j<fieldmats.n_elem; j++){
    if(fieldmats(j).n_rows>0){
      result.rows(ddims(j), ddims(j+1)-1) = fieldmats(j);
    }
  }
}

inline void field_v_concatm_rs(arma::mat& result, arma::field<arma::mat> const& fieldmats){
  // takes a field of matrices (same n cols) and outputs a single matrix concatenating all
  arma::vec ddims = drowcol_s(fieldmats);
  for(int j=0; j<fieldmats.n_elem; j++){
    if(fieldmats(j).n_rows>0){
      result.rows(ddims(j), ddims(j+1)-1) = fieldmats(j);
    }
  }
}

