#include "mgp_utils.h"
using namespace std;



bool compute_block(bool predicting, int block_ct, bool rfc){
  return predicting == false? (block_ct > 0) : predicting;;
}



arma::vec drowcol_uv(const arma::field<arma::uvec>& diag_blocks){
  int M=diag_blocks.n_elem;
  arma::vec drow = arma::zeros(M+1);
  for(int i=0; i<M; i++){
    drow(i+1) = diag_blocks(i).n_rows;
  }
  drow = arma::cumsum(drow);
  return drow;
}

arma::uvec field_v_concat_uv(arma::field<arma::uvec> const& fuv){
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

