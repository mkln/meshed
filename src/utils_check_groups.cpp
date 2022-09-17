#include "utils_check_groups.h"

using namespace std;

arma::vec check_gibbs_groups(arma::vec block_groups,
                        const arma::field<arma::vec>& parents,
                        const arma::field<arma::vec>& children,
                        const arma::vec& block_names,
                        const arma::vec& blocks,
                        int maxit){
  
  int n_blocks = block_names.n_elem;
  int it = 0;
  
  //Rcpp::Rcout << "~ grouping blocks into ci groups... ";
              //<< ".. n_blocks: " << n_blocks << endl;
  
  while(it < maxit){
    arma::vec unique_groups = arma::unique(block_groups);
    int n_gibbs_groups = unique_groups.n_elem;
    
    //int n_changes = 0;
    //Rcpp::Rcout << ".. n_gibbs_groups: " << n_gibbs_groups << endl;
    //Rcpp::Rcout << unique_groups << endl;
    
    arma::umat changes = arma::zeros<arma::umat>(n_gibbs_groups, n_blocks);
    //Rcpp::Rcout << "... iter: " << it+1 << endl;
    
    for(int g=0; g<n_gibbs_groups; g++){
      //Rcpp::Rcout << "g: " << g << endl;
    
      for(int i=0; i<n_blocks; i++){
        int u = block_names(i) - 1;
        //Rcpp::Rcout << "i: " << i << " u: " << u << endl;
        if(block_groups(u) == unique_groups(g)){
          //Rcpp::Rcout << "stuff here?" << endl;
          arma::uvec thisblock = arma::find(blocks == block_names(i));
          int n_in_block = thisblock.n_elem;
          if(n_in_block > 0){ //**
            //Rcpp::Rcout << "yes." << endl;
            for(unsigned int pp=0; pp<parents(u).n_elem; pp++){
              //Rcpp::Rcout << "parent: #" << pp << " is:" << parents(u)(pp) << endl;
              if(block_groups(parents(u)(pp)) == unique_groups(g)){
                //Rcpp::Rcout << u << " <--- " << parents(u)(pp) 
                //            << ": same group (" << block_groups(u) 
                //            << "). Creating new." << endl;
                //Rcpp::Rcout << "+";
                block_groups(parents(u)(pp)) += n_gibbs_groups;
                //n_changes += 1;
                changes(g, i) = 1;
              }
            }
            for(unsigned int cc=0; cc<children(u).n_elem; cc++){
              //Rcpp::Rcout << "child: #" << cc << " is:" << children(u)(cc) << endl;
              if(block_groups(children(u)(cc)) == unique_groups(g)){
                //Rcpp::Rcout << u << " ---> " << children(u)(cc) 
                //            << ": same group (" << block_groups(u) 
                //            << "). Creating new." << endl;
                //Rcpp::Rcout << "+";
                block_groups(children(u)(cc)) += n_gibbs_groups;
                //n_changes += 1;
                changes(g, i) = 1;
              }
            }
            
          }
        }
      }
      
    }
    
    int n_changes = arma::accu(changes);
    
    it ++;
    //Rcpp::Rcout << it << " ";
    if(n_changes == 0){
      // no changes were made, so groups are ok!
      //Rcpp::Rcout << " done." << endl;
      return block_groups;
    }
    
  }
  
  //Rcpp::Rcout << endl << "~ done, but too messy -- sampler will run sequentially." << endl;
  // group fixing failed, return sequential groups
  return arma::regspace(0, block_groups.n_elem-1);
}



