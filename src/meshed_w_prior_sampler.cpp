#include "meshed.h"

using namespace std;

void Meshed::w_prior_sample(MeshDataLMC& data){
  if(verbose & debug){
    Rcpp::Rcout << "[w_prior_sample] " << "\n";
  }
  //Rcpp::Rcout << "Lambda from:  " << Lambda_orig(0, 0) << " to  " << Lambda(0, 0) << endl;
  
  start_overall = std::chrono::steady_clock::now();
  
  //int ns = coords.n_rows;
  
  bool acceptable = refresh_cache(data);
  if(!acceptable){
    Rcpp::stop("Something went wrong went getting the conditional Gaussians. Try different theta? ");
  }
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i = 0; i<n_blocks; i++){
    int u = block_names(i)-1;
    update_block_covpars(u, data);
  }
  
  // assuming that the ordering in block_names is the ordering of the product of conditional densities
  for(unsigned int i=0; i<n_blocks; i++){
    
    int u = block_names(i) - 1;
    
    // recompute conditional mean
    arma::mat Sigi_tot = (*data.w_cond_prec_ptr.at(u)).slice(0); //
    
    arma::mat w_mean = arma::zeros(indexing(u).n_elem);
    if(parents(u).n_elem > 0){
      w_mean = (*data.w_cond_mean_K_ptr.at(u)).slice(0) * w.rows(parents_indexing(u));
    } 
    arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol(Sigi_tot, "lower")));
    
    // sample
    arma::vec rnvec = arma::randn(indexing(u).n_elem);
    arma::vec wtemp = w_mean + Sigi_chol.t() * rnvec;
    
    w.rows(indexing(u)) = wtemp;
    
  }
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    
    Rcpp::Rcout << "[w_prior_sample] loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
}

/*
 void Meshed::w_prior_sample(MeshDataLMC& data){
 
 int ns = coords.n_rows;
 message("[conj_using_cg] sending to refresh_cache.");
 bool acceptable = refresh_cache(data);
 
 Rcpp::Rcout << "cached " << endl;
 
 message("[conj_using_cg] updating vals.");
 for(int i = 0; i<n_blocks; i++){
 int u = block_names(i)-1;
 update_block_covpars(u, data);
 }
 
 
 // storing precision matrix as 2d block list
 arma::field<arma::mat> Ci_bsp(n_blocks, n_blocks);
 
 // building the precision matrix directly:
 start = std::chrono::steady_clock::now();
 arma::field<arma::umat> Ci_brow_tripls(n_blocks);
 arma::field<arma::mat> Ci_brow_vals(n_blocks);
 
 //arma::field<arma::umat> HDH_brow_tripls(n_blocks);
 //arma::field<arma::mat> HDH_brow_vals(n_blocks);
 
 arma::field<arma::umat> PRE_brow_tripls(n_blocks);
 arma::field<arma::mat> PRE_brow_vals(n_blocks);
 
 arma::vec HDy_long = arma::zeros(ns);
 arma::vec Dtaui_long = arma::zeros(ns);
 
 Rcpp::Rcout << "building matrices "<< endl;
 for(int i = 0; i<n_blocks; i++){
 //int ri = reference_blocks(i);
 int ui = block_names(i)-1;
 Ci_brow_tripls(i) = arma::zeros<arma::umat>(0, 2);
 Ci_brow_vals(i) = arma::zeros(0, 1);
 
 //HDH_brow_tripls(i) = arma::zeros<arma::umat>(0, 2);
 //HDH_brow_vals(i) = arma::zeros(0, 1);
 
 PRE_brow_tripls(i) = arma::zeros<arma::umat>(0, 2);
 PRE_brow_vals(i) = arma::zeros(0, 1);
 
 for(int j=i; j<n_blocks; j++){
 int uj = block_names(j)-1;
 arma::mat Ci_block, PRE_block;
 
 if(ui == uj){
 // block diagonal part
 Ci_block = (*data.w_cond_prec_ptr.at(ui));
 for(int c=0; c<children(ui).n_elem; c++){
 int child = children(ui)(c);
 arma::mat AK_u = (*data.w_cond_mean_K_ptr.at(child)).slice(0).cols(u_is_which_col_f(ui)(c)(0));
 Ci_block += AK_u.t() * (*data.w_cond_prec_ptr.at(child)).slice(0) * AK_u;
 }
 
 //Ci_block += HDH_block;
 
 Ci_bsp(i, j) = Ci_block;
 
 PRE_block = arma::inv_sympd( Ci_block ); //+ HDH_block );
 
 // locations to fill: indexing(ui) x indexing(uj) 
 arma::umat tripl_locs(indexing(ui).n_elem * indexing(uj).n_elem, 2);
 
 arma::mat Ci_tripl_val = arma::zeros(Ci_block.n_elem);
 //arma::mat HDH_tripl_val = arma::zeros(Ci_block.n_elem);
 arma::mat PRE_tripl_val = arma::zeros(Ci_block.n_elem);
 
 for(int ix=0; ix<indexing(ui).n_elem; ix++){
 for(int jx=0; jx<indexing(uj).n_elem; jx++){
 int vecix = arma::sub2ind(arma::size(indexing(ui).n_elem, indexing(uj).n_elem), ix, jx);
 tripl_locs(vecix, 0) = indexing(ui)(ix);
 tripl_locs(vecix, 1) = indexing(uj)(jx);
 Ci_tripl_val(vecix, 0) = Ci_block(ix, jx);
 //HDH_tripl_val(vecix, 0) = HDH_block(ix, jx);
 PRE_tripl_val(vecix, 0) = PRE_block(ix, jx);
 }
 }
 
 Ci_brow_tripls(i) = arma::join_vert(Ci_brow_tripls(i), tripl_locs);
 Ci_brow_vals(i) = arma::join_vert(Ci_brow_vals(i), Ci_tripl_val);
 
 //HDH_brow_tripls(i) = arma::join_vert(HDH_brow_tripls(i), tripl_locs);
 //HDH_brow_vals(i) = arma::join_vert(HDH_brow_vals(i), HDH_tripl_val);
 
 PRE_brow_tripls(i) = arma::join_vert(PRE_brow_tripls(i), tripl_locs);
 PRE_brow_vals(i) = arma::join_vert(PRE_brow_vals(i), PRE_tripl_val);
 
 } else {
 bool nonempty=false;
 arma::uvec oneuv = arma::ones<arma::uvec>(1);
 arma::uvec uj_is_uis_parent = arma::find(children(uj) == ui);
 if(uj_is_uis_parent.n_elem > 0){
 Rcpp::Rcout << "?!? " << endl;
 }
 
 arma::uvec ui_is_ujs_parent = arma::find(children(ui) == uj);
 if(ui_is_ujs_parent.n_elem > 0){
 nonempty = true;
 
 // ui is a parent of uj
 int c = ui_is_ujs_parent(0); // ui is uj's c-th parent
 arma::mat AK_u = (*data.w_cond_mean_K_ptr.at(uj)).slice(0).cols(u_is_which_col_f(ui)(c)(0));
 Ci_block = -AK_u.t() * (*data.w_cond_prec_ptr.at(uj)).slice(0);
 
 } else {
 // common children? in this case we can only have one common child
 arma::uvec commons = arma::intersect(children(uj), children(ui));
 if(commons.n_elem > 0){
 nonempty = true;
 
 int child = commons(0);
 arma::uvec find_ci = arma::find(children(ui) == child);
 int ci = find_ci(0);
 arma::uvec find_cj = arma::find(children(uj) == child);
 int cj = find_cj(0);
 arma::mat AK_ui = (*data.w_cond_mean_K_ptr.at(child)).slice(0).cols(u_is_which_col_f(ui)(ci)(0));
 arma::mat AK_uj = (*data.w_cond_mean_K_ptr.at(child)).slice(0).cols(u_is_which_col_f(uj)(cj)(0));
 Ci_block = AK_ui.t() * (*data.w_cond_prec_ptr.at(child)).slice(0) * AK_uj;
 }
 }
 
 if(nonempty){
 Ci_bsp(i, j) = Ci_block;
 
 // locations to fill: indexing(ui) x indexing(uj) and the transposed lower block-triangular part
 arma::umat tripl_locs1(indexing(ui).n_elem * indexing(uj).n_elem, 2);
 arma::mat tripl_val1 = arma::vectorise(Ci_block);
 
 for(int jx=0; jx<indexing(uj).n_elem; jx++){
 for(int ix=0; ix<indexing(ui).n_elem; ix++){
 int vecix = arma::sub2ind(arma::size(indexing(ui).n_elem, indexing(uj).n_elem), ix, jx);
 tripl_locs1(vecix, 0) = indexing(ui)(ix);
 tripl_locs1(vecix, 1) = indexing(uj)(jx);
 }
 }
 
 Ci_brow_tripls(i) = arma::join_vert(Ci_brow_tripls(i), tripl_locs1);
 Ci_brow_vals(i) = arma::join_vert(Ci_brow_vals(i), tripl_val1);
 
 arma::umat tripl_locs2(indexing(ui).n_elem * indexing(uj).n_elem, 2);
 arma::mat tripl_val2 = arma::vectorise(arma::trans(Ci_block));
 
 for(int jx=0; jx<indexing(uj).n_elem; jx++){
 for(int ix=0; ix<indexing(ui).n_elem; ix++){
 int vecix = arma::sub2ind(arma::size(indexing(uj).n_elem, indexing(ui).n_elem), jx, ix);
 tripl_locs2(vecix, 0) = indexing(uj)(jx); // reversed! care
 tripl_locs2(vecix, 1) = indexing(ui)(ix);
 }
 }
 Ci_brow_tripls(i) = arma::join_vert(Ci_brow_tripls(i), tripl_locs2);
 Ci_brow_vals(i) = arma::join_vert(Ci_brow_vals(i), tripl_val2);
 }
 
 }
 }
 }
 
 Rcpp::Rcout << "done with matrices "<< endl;
 
 arma::umat Cilocs = field_v_concatm(Ci_brow_tripls); 
 arma::vec Civals = field_v_concatm(Ci_brow_vals);
 
 //arma::umat HDHlocs = field_v_concatm(HDH_brow_tripls);
 //arma::vec HDHvals = field_v_concatm(HDH_brow_vals);
 
 arma::umat PRElocs = field_v_concatm(PRE_brow_tripls);
 arma::vec PREvals = field_v_concatm(PRE_brow_vals);
 
 end_overall = std::chrono::steady_clock::now();
 
 // arma -- 
 Rcpp::Rcout << "size: " << Civals.n_rows << " " << Cilocs.n_rows << endl;
 
 arma::sp_mat armaCi( arma::trans(Cilocs), Civals );
 //arma::sp_mat armaHDH( arma::trans(HDHlocs), HDHvals );
 arma::sp_mat arma_PRE( arma::trans(PRElocs), PREvals );
 
 Rcpp::Rcout << "summing " << endl; 
 
 //arma::sp_mat arma_Ci_HDH = armaCi + armaHDH;
 
 Rcpp::Rcout << "starting conjgrad " << endl;
 start = std::chrono::steady_clock::now();
 int itern = 0;
 arma::vec uu = arma::randn(armaCi.n_rows);
 arma::vec mu = sp_conjgrad(armaCi, uu,//HDy_long,  arma_PRE,  x0, 
 itern);
 end = std::chrono::steady_clock::now();
 
 Rcpp::Rcout << "cg done: itern=" << itern << ", " <<
 std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << endl;
 
 w = mu;//arma::randn(ns);
 
 }
 */
