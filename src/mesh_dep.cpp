#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

arma::vec noseqdup(arma::vec x, bool& has_changed, int maxc, int na=-1, int pred=4){
  arma::uvec locs = arma::find( (x != na) % (x != pred) );
  arma::vec xlocs = x.elem(locs);
  for(unsigned int i=1; i<xlocs.n_elem; i++){
    if(xlocs(i)==xlocs(i-1)){
      xlocs(i) += 1+maxc;
      has_changed = true;
    }
  }
  x(locs) = xlocs;
  return x;
}

//[[Rcpp::export]]
arma::field<arma::uvec> blanket(const arma::field<arma::uvec>& parents, 
                                const arma::field<arma::uvec>& children,
                                const arma::uvec& names,
                                const arma::uvec& block_ct_obs){
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  unsigned int n_blocks = names.n_elem;
  arma::field<arma::uvec> mb(n_blocks);
  
  for(unsigned int i=0; i<n_blocks; i++){
    
    int u = names(i) - 1;
    if(block_ct_obs(u) > 0){
      // block cannot be the same color as other nodes in blanket
      arma::uvec u_blanket = arma::zeros<arma::uvec>(0);
      for(unsigned int p=0; p<parents(u).n_elem; p++){
        int px = parents(u)(p);
        u_blanket = arma::join_vert(u_blanket, px*oneuv);
      }
      for(unsigned int c=0; c<children(u).n_elem; c++){
        int cx = children(u)(c);
        u_blanket = arma::join_vert(u_blanket, cx*oneuv);
        
        for(unsigned int pc=0; pc<parents(cx).n_elem; pc++){
          int pcx = parents(cx)(pc);
          if(pcx != u){
            u_blanket = arma::join_vert(u_blanket, pcx*oneuv);
          }
        }
      }
      mb(u) = u_blanket;
    }
  }
  return mb;
}

arma::ivec std_setdiff(arma::ivec& x, arma::ivec& y) {
  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  return arma::conv_to<arma::ivec>::from(out);
}

//[[Rcpp::export]]
arma::ivec coloring(const arma::field<arma::uvec>& blanket, const arma::uvec& block_names, const arma::uvec& block_ct_obs){
  int n_blocks = blanket.n_elem;
  arma::ivec oneiv = arma::ones<arma::ivec>(1);
  arma::ivec color_picker = arma::zeros<arma::ivec>(1);
  arma::ivec colors = arma::zeros<arma::ivec>(n_blocks) - 1;
  
  int start_i=0;
  bool goon=true;
  while(goon){
    int u = block_names(start_i) - 1;
    if(block_ct_obs(u) > 0){
      colors(u) = 0;
      goon = false;
    } else {
      colors(u) = -1;
    }
    start_i ++;
  }
  
  for(int i=start_i; i<n_blocks; i++){
    int u = block_names(i) - 1;
    if(block_ct_obs(u) > 0){
      arma::ivec neighbor_colors = colors(blanket(u));
      arma::ivec neighbor_colors_used = neighbor_colors(arma::find(neighbor_colors > -1));
      arma::ivec colors_available = std_setdiff(color_picker, neighbor_colors_used);
      
      
      int choice_color = -1;
      if(colors_available.n_elem > 0){
        choice_color = arma::min(colors_available);
      } else {
        choice_color = arma::max(color_picker) + 1;
        color_picker = arma::join_vert(color_picker, oneiv * choice_color);
      }
      
      colors(u) = choice_color;
    }
    
  }
  return colors;
}

arma::vec turbocolthreshold(const arma::vec& col1, const arma::vec& thresholds){
  arma::vec result = arma::zeros(col1.n_elem);
  for(unsigned int i=0; i<col1.n_elem; i++){
    int overthreshold = 1;
    for(unsigned int j=0; j<thresholds.n_elem; j++){
      if(col1(i) >= thresholds(j)){
        overthreshold += 1;
      }
    }
    result(i) = overthreshold;
  }
  return result;
}

//[[Rcpp::export]]
arma::vec kthresholdscp(arma::vec x,
                      unsigned int k){
  arma::vec res(k-1);
  
  for(unsigned int i=1; i<k; i++){
    unsigned int Q1 = i * x.n_elem / k;
    std::nth_element(x.begin(), x.begin() + Q1, x.end());
    res(i-1) = x(Q1);
  }
  
  return res;
}

//[[Rcpp::export]]
arma::mat part_axis_parallel(const arma::mat& coords, const arma::vec& Mv, int n_threads, bool verbose=false){
  if(verbose){
    Rcpp::Rcout << "  Axis-parallel partitioning... ";
  }
  arma::mat resultmat = arma::zeros(arma::size(coords));
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(unsigned int j=0; j<coords.n_cols; j++){
    arma::vec cja = coords.col(j);
    arma::vec thresholds = kthresholdscp(cja, Mv(j));
    resultmat.col(j) = turbocolthreshold(coords.col(j), thresholds);
  }
  if(verbose){
    Rcpp::Rcout << "done.\n";
  }
  return resultmat;
}

//[[Rcpp::export]]
arma::mat part_axis_parallel_fixed(const arma::mat& coords, const arma::field<arma::vec>& thresholds, int n_threads){
  arma::mat resultmat = arma::zeros(arma::size(coords));
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(unsigned int j=0; j<thresholds.n_elem; j++){
    arma::vec cja = coords.col(j);
    arma::vec thresholds_col = thresholds(j);
    resultmat.col(j) = turbocolthreshold(coords.col(j), thresholds_col);
  }
  return resultmat;
}

//[[Rcpp::export]]
Rcpp::List mesh_graph_cpp(const arma::mat& layers_descr, 
                          const arma::uvec& Mv, 
                          bool verbose=true,
                          bool both_spatial_axes=true,
                          int n_threads=1){
  // coords_layering is a matrix
  // Var1 Var2 [Var3] L1 L2 [L3] layer na_which
  // layers_descr = coords_layering %>% select(-contains("Var")) 
  //                                %>% group_by(L1, L2, L3, layer) 
  //                                %>% summarize(na_which = sum(na_which))
  //                                %>% unique()
  
  
#ifdef _OPENMP
  omp_set_num_threads(n_threads);
#endif
  
  
  std::chrono::steady_clock::time_point start, start_all;
  std::chrono::steady_clock::time_point end, end_all;
  
  start_all = std::chrono::steady_clock::now();
  
  int dimen = 2;
  if(layers_descr.n_cols > 4){
    dimen = 3;
  }
  
  if(verbose){
    Rcpp::Rcout << "~ Building cubic mesh, d = " << dimen << "\n";
  }
  int num_blocks = arma::prod(Mv);
  
  arma::vec lnames = layers_descr.col(dimen);
  arma::field<arma::vec> parents(num_blocks);
  arma::field<arma::vec> children(num_blocks);
  arma::uvec uzero = arma::zeros<arma::uvec>(1);
  
  for(int i=0; i<num_blocks; i++){
    parents(i) = arma::zeros(dimen) - 1;
    children(i) = arma::zeros(dimen) - 1;
  }
  
  arma::mat blocks_ref = layers_descr.rows(arma::find(layers_descr.col(dimen+1) > 0));//layers_preds;
  arma::cube Q, Qall;
  arma::mat Qm, Qmall;
  
  if(dimen == 2){
    
    start = std::chrono::steady_clock::now();
    Qm = arma::zeros(Mv(0), Mv(1))-1;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(unsigned int i=1; i<Mv(0)+1; i++){
      arma::mat filter_i    = blocks_ref.rows(arma::find(blocks_ref.col(0) == i));
      if(filter_i.n_rows > 0){
        for(unsigned int j=1; j<Mv(1)+1; j++){
          arma::mat filter_j    = filter_i.rows(arma::find(filter_i.col(1) == j));
          if(filter_j.n_rows > 0){ 
            Qm(i-1, j-1) = filter_j(0, 2);
          } 
        }
      }
    }
    
    Qmall = arma::zeros(Mv(0), Mv(1))-1;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(unsigned int i=1; i<Mv(0)+1; i++){
      arma::mat filter_alli = layers_descr.rows(arma::find(layers_descr.col(0) == i));
      if(filter_alli.n_rows > 0){
        for(unsigned int j=1; j<Mv(1)+1; j++){
          arma::mat filter_allj = filter_alli.rows(arma::find(filter_alli.col(1) == j));
          if(filter_allj.n_rows > 0){
            Qmall(i-1, j-1) = filter_allj(0, 2);
          }
        }
      }
    }
    end = std::chrono::steady_clock::now();
    
    
    int Imax = Qm.n_rows-1;
    int Jmax = Qm.n_cols-1;
    
    start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(unsigned int i=1; i<Mv(0)+1; i++){
      for(unsigned int j=1; j<Mv(1)+1; j++){
        int layern = Qm(i-1, j-1);
        if(layern != -1){
          
          if(both_spatial_axes){
            if(j < Mv(1)){
              arma::vec q1 = arma::vectorise(Qm.submat(i-1, j,  
                                                       i-1, Jmax));
              
              arma::uvec locator_sub_ijh_1 = arma::find(q1 != -1);
              if(locator_sub_ijh_1.n_elem > 0){
                int prop_layern = q1(locator_sub_ijh_1(0));
                children(layern-1)(0) = prop_layern-1;
                parents(prop_layern-1)(0) = layern-1;
              }
            }
          }
        
          if(i < Mv(0)){
            arma::vec q2 = Qm.submat(i,    j-1, 
                                     Imax, j-1);
            
            arma::uvec locator_sub_ijh_2 = arma::find(q2 != -1);
            if(locator_sub_ijh_2.n_elem > 0){
              int prop_layern = q2(locator_sub_ijh_2(0));
              children(layern-1)(1) = prop_layern-1;
              parents(prop_layern-1)(1) = layern-1;
            }
          }
          
          
        }
      }
    }
    end = std::chrono::steady_clock::now();

    start = std::chrono::steady_clock::now();
    arma::uvec empties = arma::find(Qm == -1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(unsigned int i=0; i<empties.n_elem; i++){
      arma::uvec ijh = arma::ind2sub(arma::size(Qm), empties(i));
      
      int layern = Qmall(ijh(0), ijh(1));
      if(layern > -1){
        arma::vec axis1 = arma::vectorise(Qm.row(ijh(0))); // fix row ijh(0), look at cols
        arma::vec axis2 = arma::vectorise(Qm.col(ijh(1)));
        
        arma::uvec nne1 = arma::find(axis1 > -1);
        arma::uvec nne2 = arma::find(axis2 > -1);
        
        // free cols
        arma::uvec loc1before = arma::find(nne1 < ijh(1));
        arma::uvec loc1after = arma::find(nne1 > ijh(1));
        
        // free rows
        arma::uvec loc2before = arma::find(nne2 < ijh(0));
        arma::uvec loc2after = arma::find(nne2 > ijh(0));
        
        parents(layern-1) = arma::zeros(4) -1;
        
        if(both_spatial_axes){
          if(loc1before.n_elem > 0){
            arma::uvec nne1_before = nne1(loc1before);
            parents(layern-1)(0) = arma::conv_to<int>::from(
              axis1(nne1_before.tail(1)) - 1);
          }
          if(loc1after.n_elem > 0){
            arma::uvec nne1_after  = nne1(loc1after);
            parents(layern-1)(1) = arma::conv_to<int>::from(
              axis1(nne1_after.head(1)) - 1);
          }
        }
        
        if(loc2before.n_elem > 0){
          arma::uvec nne2_before = nne2(loc2before);
          parents(layern-1)(2) = arma::conv_to<int>::from(
            axis2(nne2_before.tail(1)) - 1);
        }
        if(loc2after.n_elem > 0){
          arma::uvec nne2_after  = nne2(loc2after);
          parents(layern-1)(3) = arma::conv_to<int>::from(
            axis2(nne2_after.head(1)) - 1);
        }
        
        
      } 
    }
    end = std::chrono::steady_clock::now();
    
    
  } else {
    start = std::chrono::steady_clock::now();
    Q = arma::zeros(Mv(0), Mv(1), Mv(2))-1;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(unsigned int i=1; i<Mv(0)+1; i++){
      arma::mat filter_i = blocks_ref.rows(arma::find(blocks_ref.col(0) == i));
      if(filter_i.n_rows > 0){
        for(unsigned int j=1; j<Mv(1)+1; j++){
          arma::mat filter_j = filter_i.rows(arma::find(filter_i.col(1) == j));
          if(filter_j.n_rows > 0){
            for(unsigned int h=1; h<Mv(2)+1; h++){
              arma::mat filter_h = filter_j.rows(arma::find(filter_j.col(2) == h));
              if(filter_h.n_rows > 0){ 
                Q(i-1, j-1, h-1) = filter_h(0, 3);
              } 
            }
          }
        }
      }
    }
    
  
    Qall = arma::zeros(Mv(0), Mv(1), Mv(2))-1;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(unsigned int i=1; i<Mv(0)+1; i++){
      arma::mat filter_alli = layers_descr.rows(arma::find(layers_descr.col(0) == i));
      if(filter_alli.n_rows > 0){
        for(unsigned int j=1; j<Mv(1)+1; j++){
          arma::mat filter_allj = filter_alli.rows(arma::find(filter_alli.col(1) == j));
          if(filter_allj.n_rows > 0){
            for(unsigned int h=1; h<Mv(2)+1; h++){
              arma::mat filter_allh = filter_allj.rows(arma::find(filter_allj.col(2) == h));
              if(filter_allh.n_rows > 0){ 
                Qall(i-1, j-1, h-1) = filter_allh(0, 3);
              }
            }
          }
        }
      }
    }
    end = std::chrono::steady_clock::now();
    
    int Imax = Q.n_rows-1;
    int Jmax = Q.n_cols-1;
    int Hmax = Q.n_slices-1;
    start = std::chrono::steady_clock::now();
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(unsigned int i=1; i<Mv(0)+1; i++){
      for(unsigned int j=1; j<Mv(1)+1; j++){
        for(unsigned int h=1; h<Mv(2)+1; h++){
          int layern = Q(i-1, j-1, h-1);
          if(layern != -1){
            
            if(j < Mv(1)){
              arma::vec q1 = arma::vectorise(Q.subcube(i-1, j,    h-1, 
                                                       i-1, Jmax, h-1));
              
              arma::uvec locator_sub_ijh_1 = arma::find(q1 != -1);
              if(locator_sub_ijh_1.n_elem > 0){
                int prop_layern = q1(locator_sub_ijh_1(0));
                children(layern-1)(0) = prop_layern-1;
                parents(prop_layern-1)(0) = layern-1;
              }
            }
            
            if(i < Mv(0)){
              arma::vec q2 = Q.subcube(i,    j-1, h-1, 
                                       Imax, j-1, h-1);
              
              arma::uvec locator_sub_ijh_2 = arma::find(q2 != -1);
              if(locator_sub_ijh_2.n_elem > 0){
                int prop_layern = q2(locator_sub_ijh_2(0));
                children(layern-1)(1) = prop_layern-1;
                parents(prop_layern-1)(1) = layern-1;
              }
            }
            
            if(h < Mv(2)){
              arma::vec q3 = arma::vectorise(Q.subcube(i-1, j-1, h, 
                                                       i-1, j-1, Hmax));
              
              arma::uvec locator_sub_ijh_3 = arma::find(q3 != -1);
              if(locator_sub_ijh_3.n_elem > 0){
                int prop_layern = q3(locator_sub_ijh_3(0));
                children(layern-1)(2) = prop_layern-1;
                parents(prop_layern-1)(2) = layern-1;
              }
            }
            
          }
          
        }
      }
    }
    
    end = std::chrono::steady_clock::now();
    
    start = std::chrono::steady_clock::now();
    arma::uvec empties = arma::find(Q == -1);
    for(unsigned int i=0; i<empties.n_elem; i++){
      arma::uvec ijh = arma::ind2sub(arma::size(Q), empties(i));
      
      int layern = Qall(ijh(0), ijh(1), ijh(2));
      if(layern > -1){
        arma::vec axis1 = arma::vectorise(Q.subcube(ijh(0), ijh(1), 0, 
                                                    ijh(0), ijh(1), Hmax)); 
        arma::vec axis2 = arma::vectorise(Q.subcube(ijh(0), 0,    ijh(2), 
                                                    ijh(0), Jmax, ijh(2))); 
        arma::vec axis3 = arma::vectorise(Q.subcube(0,    ijh(1), ijh(2), 
                                                    Imax, ijh(1), ijh(2))); 
        
        arma::uvec nne1 = arma::find(axis1 > -1);
        arma::uvec nne2 = arma::find(axis2 > -1);
        arma::uvec nne3 = arma::find(axis3 > -1);
        
        // free slices
        arma::uvec loc1before = arma::find(nne1 < ijh(2));
        arma::uvec loc1after = arma::find(nne1 > ijh(2));
        
        // free cols
        arma::uvec loc2before = arma::find(nne2 < ijh(1));
        arma::uvec loc2after = arma::find(nne2 > ijh(1));
        
        // free rows
        arma::uvec loc3before = arma::find(nne3 < ijh(0));
        arma::uvec loc3after = arma::find(nne3 > ijh(0));
        
        parents(layern-1) = arma::zeros(6) -1;
        
        if(loc1before.n_elem > 0){
          arma::uvec nne1_before = nne1(loc1before);
          parents(layern-1)(0) = arma::conv_to<int>::from(
            axis1(nne1_before.tail(1)) - 1);
        }
        if(loc1after.n_elem > 0){
          arma::uvec nne1_after  = nne1(loc1after);
          parents(layern-1)(1) = arma::conv_to<int>::from(
            axis1(nne1_after.head(1)) - 1);
        }
        if(both_spatial_axes){
          if(loc2before.n_elem > 0){
            arma::uvec nne2_before = nne2(loc2before);
            parents(layern-1)(2) = arma::conv_to<int>::from(
              axis2(nne2_before.tail(1)) - 1);
          }
          if(loc2after.n_elem > 0){
            arma::uvec nne2_after  = nne2(loc2after);
            parents(layern-1)(3) = arma::conv_to<int>::from(
              axis2(nne2_after.head(1)) - 1);
          }
        }
        
        if(loc3before.n_elem > 0){
          arma::uvec nne3_before = nne3(loc3before); 
          parents(layern-1)(4) = arma::conv_to<int>::from(
            axis3(nne3_before.tail(1)) - 1);
        }
        if(loc3after.n_elem > 0){
          arma::uvec nne3_after  = nne3(loc3after);
          parents(layern-1)(5) = arma::conv_to<int>::from(
            axis3(nne3_after.head(1)) - 1);
        }
      }
      
      
    }
    end = std::chrono::steady_clock::now();
  
  }
  
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(unsigned int i=0; i<parents.n_elem; i++){
    parents(i) = parents(i).elem(arma::find(parents(i) != -1));
    children(i) = children(i).elem(arma::find(children(i) != -1));
  }
  
  end_all = std::chrono::steady_clock::now();
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children,
    Rcpp::Named("names") = lnames
  );
}

//[[Rcpp::export]]
arma::cube cube_from_df(const arma::mat& indices, const arma::vec& values){
  // indices: n rows, 2 or 3 columns
  int d = indices.n_cols;
  
  int dimx = indices.col(0).max();
  int dimy = indices.col(1).max();
  int dimz = d==2? 1 : indices.col(2).max();
  
  arma::cube result = arma::zeros(dimx, dimy, dimz) - 1; // set to -1 values;
  for(unsigned int i=0; i<indices.n_rows; i++){
    int x = indices(i, 0)-1;
    int y = indices(i, 1)-1;
    int z = d==2 ? 0 : indices(i, 2)-1;
    
    result(x, y, z) = values(i);
  }
  return result;
}


arma::mat edist(const arma::mat& x, const arma::mat& y, const arma::vec& w, bool same){
  // 0 based indexing
  
  arma::rowvec delta;
  arma::rowvec xi;
  arma::mat res = arma::zeros(x.n_rows, y.n_rows);
  if(same){
    for(unsigned int i=0; i<x.n_rows; i++){
      xi = x.row(i);
      for(unsigned int j=i; j<y.n_rows; j++){
        delta = xi - y.row(j);
        res(i, j) = arma::accu(w.t() % delta % delta);
      }
    }
    res = arma::symmatu(res);
  } else {
    //int cc = 0;
    for(unsigned int i=0; i<x.n_rows; i++){
      xi = x.row(i);
      for(unsigned int j=0; j<y.n_rows; j++){
        delta = xi - y.row(j);
        res(i, j) = arma::accu(w.t() % delta % delta);
      }
    }
  }
  return res;
}

//[[Rcpp::export]]
arma::umat knn_naive(const arma::mat& x, const arma::mat& search_here, const arma::vec& weights){
  arma::mat D = edist(x, search_here, weights, false);
  arma::umat Dfound = arma::zeros<arma::umat>(arma::size(D));
  for(unsigned int i=0; i<x.n_rows; i++){
    Dfound.row(i) = arma::trans(arma::sort_index(D.row(i)));
  }
  return Dfound;
}


//[[Rcpp::export]]
Rcpp::List mesh_graph_cpp3(const arma::mat& blocks_descr){
  
  int d = blocks_descr.n_cols - 2;
  
  arma::cube blocknames = cube_from_df(blocks_descr.cols(0, d-1), blocks_descr.col(d)-1);
  arma::cube nonmissing_perc = cube_from_df(blocks_descr.cols(0, d-1), blocks_descr.col(d+1));
  
  arma::field<arma::ivec> parents(blocknames.max()+1);
  arma::field<arma::ivec> children(blocknames.max()+1);
  
  for(unsigned int i=0; i<parents.n_elem; i++){
    parents(i) = arma::zeros<arma::ivec>(4) - 1;
    children(i) = arma::zeros<arma::ivec>(4) - 1;
  }
  
  unsigned int nx = blocknames.n_rows;
  unsigned int ny = blocknames.n_cols;
  unsigned int nz = blocknames.n_slices;
  
  for(unsigned int i=0; i<nx; i++){
    for(unsigned int j=0; j<ny; j++){
      for(unsigned int z=0; z<nz; z++){
        int name = blocknames(i, j, z);
        bool not_reference = nonmissing_perc(i, j, z) == 0;
        
          if(i > 0){
            bool found = false;
            unsigned int t = 1;
            while((!found) & (t <= i)){
              if(nonmissing_perc(i-t, j, z) > 0){
                int parent_name = blocknames(i-t, j, z);
                parents(name)(0) = parent_name;
                children(parent_name)(0) = not_reference ? -1 : name;
                found = true;
              } else {
                t ++;
              }
            }
            
          }
          if(j > 0){
            bool found = false;
            unsigned int t = 1;
            while((!found) & (t <= j)){
              if(nonmissing_perc(i, j-t, z) > 0){
                int parent_name = blocknames(i, j-t, z);
                parents(name)(1) = parent_name;
                children(parent_name)(1) = not_reference ? -1 : name;
                found = true;
              } else {
                t ++;
              }
            }
          }
          if((i > 0) & (j > 0)){
            if(nonmissing_perc(i-1, j-1, z) > 0){
              int parent_name = blocknames(i-1, j-1, z);
              parents(name)(2) = parent_name;
              children(parent_name)(2) = not_reference ? -1 : name;
            } 
          }
          if(z > 0){
            bool found = false;
            unsigned int t = 1;
            while((!found) & (t <= z)){
              if(nonmissing_perc(i, j, z-t) > 0){
                int parent_name = blocknames(i, j, z-t);
                parents(name)(3) = parent_name;
                children(parent_name)(3) = not_reference ? -1 : name;
                found = true;
              } else {
                t ++;
              }
            }
          }
          
          if(not_reference & (parents(name)(0) == -1)){
            // search other spatial direction along x axis
            bool found = false;
            unsigned int t = 1;
            while((!found) & (i+t < nx)){
              if(nonmissing_perc(i+t, j, z) > 0){
                int parent_name = blocknames(i+t, j, z);
                parents(name)(0) = parent_name;
                children(parent_name)(0) = not_reference ? -1 : name;
                found = true;
              } else {
                t ++;
              }
            }
          }
          if(not_reference & (parents(name)(1) == -1)){
            // search other spatial direction along y axis
            bool found = false;
            unsigned int t = 1;
            while((!found) & (j+t < ny)){
              if(nonmissing_perc(i, j+t, z) > 0){
                int parent_name = blocknames(i, j+t, z);
                parents(name)(1) = parent_name;
                children(parent_name)(1) = not_reference ? -1 : name;
                found = true;
              } else {
                t ++;
              }
            }
          }
          if(not_reference & (parents(name)(3) == -1) & (d==3)){
            // search other spatial direction along z axis
            bool found = false;
            unsigned int t = 1;
            while((!found) & (z+t < nz)){
              if(nonmissing_perc(i, j, z+t) > 0){
                int parent_name = blocknames(i, j, z+t);
                parents(name)(3) = parent_name;
                children(parent_name)(3) = not_reference ? -1 : name;
                found = true;
              } else {
                t ++;
              }
            }
          }
        
      }
    }
  }
  
  arma::vec lnames = blocks_descr.col(d);
  
  for(unsigned int i=0; i<parents.n_elem; i++){
    parents(i) = parents(i).elem(arma::find(parents(i) != -1));
    children(i) = children(i).elem(arma::find(children(i) != -1));
  }
  
  Rcpp::List result;
  result["parents"] = parents;
  result["children"] = children;
  result["names"] = lnames;
  return result;
}


arma::umat filter_col_smaller(const arma::umat& base_mat, int r_col_ix, int value){
  return base_mat.rows(arma::find(base_mat.col(r_col_ix-1) < value));
}

arma::umat filter_col_greater(const arma::umat& base_mat, int r_col_ix, int value){
  return base_mat.rows(arma::find(base_mat.col(r_col_ix-1) > value));
}

arma::umat filter_col_equal(const arma::umat& base_mat, int r_col_ix, int value){
  return base_mat.rows(arma::find(base_mat.col(r_col_ix-1) == value));
}

arma::umat filter_cols_equal(const arma::umat& base_mat, const arma::uvec& r_col_ix, const arma::uvec& values){
  arma::uvec all_conditions = arma::ones<arma::uvec>(base_mat.n_rows);
  for(unsigned int j=0; j<r_col_ix.n_elem; j++){
    all_conditions %= base_mat.col(r_col_ix(j)-1) == values(j);
  }
  return base_mat.rows(arma::find(all_conditions));
}

//[[Rcpp::export]]
Rcpp::List mesh_graph_hyper(const arma::umat& bucbl, const arma::umat& bavail,
                            const arma::vec& na_which, 
                            const arma::mat& centroids, 
                            const arma::mat& avcentroids, const arma::uvec& avblocks, int k=1){
  
  int dimen = bucbl.n_cols - 1; // last column is the block uid
  int nblocks = bucbl.n_rows;
  arma::uvec block_names = bucbl.col(dimen);
  
  arma::field<arma::ivec> parents(nblocks);
  arma::field<arma::ivec> children(nblocks);
  for(int i=0; i<nblocks; i++){
    parents(i) = arma::zeros<arma::ivec>(dimen * 2) - 1;
    children(i) = arma::zeros<arma::ivec>(dimen) - 1;
  }
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
  for(int i=0; i<nblocks; i++){
    int u = block_names(i) - 1;
    arma::uvec block_info = arma::trans(bucbl.row(i));
    
    if(na_which(i) > 0){
      // block has some observations: reference
      for(int d=0; d<dimen; d++){
        int dimvalue = block_info(d);
        arma::uvec onetodimen_minus_d = arma::regspace<arma::uvec>(1, 1, dimen);
        arma::uvec othervals = block_info;
        othervals.shed_row(d);
        onetodimen_minus_d.shed_row(d);
        
        arma::umat bavail_axis = filter_cols_equal(bavail, onetodimen_minus_d, othervals);
        bavail_axis = filter_col_greater(bavail_axis, d+1, dimvalue);
        if(bavail_axis.n_rows > 0){
          int next_in_line = bavail_axis(0, dimen)-1;
          parents(next_in_line)(d) = u;
          children(u)(d) = next_in_line;
        }
      }
    } else {
      // block does not have observ: predicted
      arma::uvec oneuv = arma::ones<arma::uvec>(1);
      arma::vec weights = arma::ones(dimen);
      arma::umat knn_results = knn_naive(centroids.rows(oneuv * u), avcentroids, weights);
      arma::uvec knn = arma::trans( knn_results.submat(0, 0, 0, k) );
      arma::uvec knn_blocks = avblocks.elem(knn);
      parents(u) = arma::conv_to<arma::ivec>::from(knn_blocks - 1);
    
    }
  }
  
  for(int i=0; i<nblocks; i++){
    parents(i) = parents(i)(arma::find(parents(i) != -1));
    children(i) = children(i)(arma::find(children(i) != -1));
  }
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children,
    Rcpp::Named("names") = block_names
  );
}

//[[Rcpp::export]]
arma::mat repeat_centroid_perturb(const arma::mat& x, const arma::uvec& times){
  unsigned int n = arma::accu(times);
  arma::mat result = arma::zeros(n, x.n_cols);
  int rix=0;
  for(unsigned int i=0; i<x.n_rows; i++){
    for(unsigned int j=0; j<times(i); j++){
      result.row(rix) = x.row(i) + arma::trans(arma::randn(x.n_cols)*1e-5);
      rix++;
    }
  }
  return result;
}