meshedgp <- function(y, X, coords, k=NULL,
                     block_size = 30,
                     axis_partition = NULL, 
                     grid_size=NULL,
                   n_samples = 1000,
                   n_burnin = 100,
                   n_thin = 1,
                   n_threads = 4,
                   settings    = list(adapting=T, mcmcsd=.05, forced_grid=NULL),
                   prior       = list(beta=NULL, tausq=NULL,
                                      toplim = NULL, btmlim = NULL, set_unif_bounds=NULL),
                   starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL),
                   debug       = list(sample_beta=T, sample_tausq=T, 
                                      sample_theta=T, sample_w=T, sample_lambda=T,
                                      verbose=F, debug=F, print_every=100)
                   ){

  # init
  cat("Bayesian Meshed GP regression model\n
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o\n\n")
  
  set_default <- function(x, default_val){
    return(if(is.null(x)){
      default_val} else {
        x
      })}
  
  # data management pt 1
  if(1){
    mcmc_keep <- n_samples
    mcmc_burn <- n_burnin
    mcmc_thin <- n_thin
    
    mcmc_adaptive    <- settings$adapting %>% set_default(T)
    
    mcmc_verbose     <- debug$verbose %>% set_default(F)
    mcmc_debug       <- debug$debug %>% set_default(F)
    
    dd             <- ncol(coords)
    p              <- ncol(X)
    
    # data management part 0 - reshape/rename
    if(is.null(dim(y))){
      y <- matrix(y, ncol=1)
      orig_y_colnames <- colnames(y) <- "Y_1"
    } else {
      if(is.null(colnames(y))){
        orig_y_colnames <- colnames(y) <- paste0('Y_', 1:ncol(y))
      } else {
        orig_y_colnames <- colnames(y)
        colnames(y)     <- paste0('Y_', 1:ncol(y))
      }
    }
    
    effective_dimension <- prod(dim(y))
    if(is.null(debug$print_every)){
      if(effective_dimension > 1e6){
        mcmc_print_every <- 5
      } else {
        if(effective_dimension > 1e5){
          mcmc_print_every <- 50
        } else {
          mcmc_print_every <- 500
        }
      }
    } else {
      mcmc_print_every <- debug$print_every
    }
    
    if(is.null(colnames(X))){
      orig_X_colnames <- colnames(X) <- paste0('X_', 1:ncol(X))
    } else {
      orig_X_colnames <- colnames(X)
      colnames(X)     <- paste0('X_', 1:ncol(X))
    }
    
    colnames(coords)  <- paste0('Var', 1:dd)
  
    q              <- ncol(y)
    k              <- ifelse(is.null(k), q, k)
    n_par_each_process <- 1 # for spatial data and independent processes
    
    nr             <- nrow(X)
    
    if(length(axis_partition) == 1){
      axis_partition <- rep(axis_partition, dd)
    }
    if(is.null(axis_partition)){
      axis_partition <- rep(round((nr/block_size)^(1/dd)), dd)
    }
    
    # -- heuristics for gridded data --
    # if we observed all unique combinations of coordinates, this is how many rows we'd have
    heuristic_gridded <- prod( coords %>% apply(2, function(x) length(unique(x))) )
    # if we're not too far off maybe the dataset is actually gridded
    if((nrow(coords) < .5*heuristic_gridded) & (heuristic_gridded*1.5 < nrow(coords))){
      data_likely_gridded <- F
    } else {
      data_likely_gridded <- T
    }
    if(ncol(coords) == 3){
      # with time, check if there's equal spacing
      timepoints <- coords[,3] %>% unique()
      time_spacings <- timepoints %>% sort() %>% diff() %>% round(5) %>% unique()
      if(length(time_spacings) < .1 * length(timepoints)){
        data_likely_gridded <- T
      }
    }
    
    if(is.null(settings$forced_grid)){
      if(data_likely_gridded){
        use_forced_grid <- F
      } else {
        use_forced_grid <- T
      }
    } else {
      use_forced_grid <- settings$forced_grid %>% set_default(T)
      if(!use_forced_grid & !data_likely_gridded){
        warning("Data look not gridded: force a grid with settings$forced_grid=T.")
      }
    }
    
    sample_beta    <- debug$sample_beta
    sample_tausq   <- debug$sample_tausq
    
    sample_theta   <- debug$sample_theta
    sample_w       <- debug$sample_w
    sample_lambda <- debug$sample_lambda
    
    if(is.null(starting$beta)){
      start_beta   <- rep(0, p)
    } else {
      start_beta   <- starting$beta
    }
    
    if(is.null(prior$btmlim)){
      btmlim <- 1e-2
    } else {
      btmlim <- prior$btmlim
    }
    
    if(is.null(prior$toplim)){
      toplim <- 1e2
    } else {
      toplim <- prior$toplim
    }
    
    if(is.null(prior$set_unif_bounds)){
      if(dd == 2){
        start_theta <- matrix(2, ncol=k, nrow=1) 
        
        set_unif_bounds <- matrix(0, nrow=k, ncol=2)
        set_unif_bounds[,1] <- btmlim
        set_unif_bounds[,2] <- toplim
      }
      
    } else {
      set_unif_bounds <- prior$set_unif_bounds
    }
    
    if(!is.null(starting$theta)){
      start_theta <- starting$theta
    }
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(X)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$tausq)){
      tausq_ab <- c(2.01, 1)
    } else {
      tausq_ab <- prior$tausq
    }
    
    if(length(settings$mcmcsd) == 1){
      mcmc_mh_sd <- diag(k * n_par_each_process) * settings$mcmcsd
    } else {
      mcmc_mh_sd <- settings$mcmcsd
    }
    
    if(is.null(starting$tausq)){
      start_tausq  <- .1
    } else {
      start_tausq    <- starting$tausq
    }
    
    if(is.null(starting$sigmasq)){
      start_sigmasq <- 10
    } else {
      start_sigmasq  <- starting$sigmasq
    }
    
    if(is.null(starting$w)){
      start_w <- rep(0, nrow(coords))
    } else {
      start_w <- starting$w #%>% matrix(ncol=q)
    }
  }

  # data management pt 2
  if(1){
    
    yrownas <- apply(y, 1, function(i) ifelse(sum(is.na(i))==q, NA, 1))
    na_which <- ifelse(!is.na(yrownas), 1, NA)
    simdata <- data.frame(ix=1:nrow(coords)) %>% 
      cbind(coords, y, na_which, X) %>% 
      as.data.frame()
    
    #####
   
    cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    cat("{q} outcome variables on {nrow(unique(coords))} unique locations." %>% glue::glue())
    cat("\n")
    if(use_forced_grid){ #if(!is.null(grid_size)){
      # user showed intention to use fixed grid
      #mean_rows_by_var <- simdata %>% group_by(mv_id) %>% summarise(size=n()) %$% size %>% mean()
      
      gs <- round(nrow(coords)^(1/ncol(coords)))
      gsize <- if(is.null(grid_size)){ rep(gs, ncol(coords)) } else { grid_size }
      #mv_uniques <- unique(simdata$mv_id)
      
      #print(gsize)
      
      xgrids <- list()
      for(j in 1:dd){
        xgrids[[j]] <- seq(min(coords[,j]), max(coords[,j]), length.out=gsize[j])
      }
      
      gridcoords_lmc <- expand.grid(xgrids)
      
      cat("Forced grid built with {nrow(gridcoords_lmc)} locations." %>% glue::glue())
      cat("\n")
      simdata <- dplyr::bind_rows(simdata %>% mutate(thegrid=0), 
                           gridcoords_lmc %>% mutate(thegrid=1))
      
      absize <- round(nrow(gridcoords_lmc)/prod(axis_partition))
    } else {
      simdata %<>% mutate(thegrid = 0)
      absize <- round(nrow(simdata)/prod(axis_partition))
    }
    cat("Partitioning grid axes into {paste0(axis_partition, collapse=', ')} intervals. Approx block size {absize}" %>% glue::glue())
    cat("\n")
    
    cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    
    simdata %<>% dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
    #colnames(simdata)[dd + (2:4)] <- c("mv_id", "y", "na_which")
    
    coords <- simdata %>% dplyr::select(dplyr::contains("Var")) %>% as.matrix()
    #simdata %<>% mutate(type="obs")
    sort_ix     <- simdata$ix
    
    # Domain partitioning and gibbs groups
    if(use_forced_grid){
      gridded_coords <- simdata %>% filter(thegrid==1) %>% dplyr::select(dplyr::contains("Var")) %>% as.matrix()
      fixed_thresholds <- 1:dd %>% lapply(function(i) spmeshed:::kthresholdscp(gridded_coords[,i], axis_partition[i])) 
    } else {
      fixed_thresholds <- 1:dd %>% lapply(function(i) spmeshed:::kthresholdscp(coords[,i], axis_partition[i])) 
    }
    
    # guaranteed to produce blocks using Mv
    system.time(fake_coords_blocking <- coords %>% 
                  as.matrix() %>% 
                  spmeshed:::gen_fake_coords(fixed_thresholds, 1) )
    
    # Domain partitioning and gibbs groups
    system.time(coords_blocking <- coords %>% 
                  as.matrix() %>%
                  spmeshed:::tessellation_axis_parallel_fix(fixed_thresholds, 1) %>% 
                  dplyr::mutate(na_which = simdata$na_which, sort_ix=sort_ix) )
    
    coords_blocking %<>% dplyr::rename(ix=sort_ix)
    
    # check if some blocks come up totally empty
    blocks_prop <- coords_blocking[,paste0("L", 1:dd)] %>% unique()
    blocks_fake <- fake_coords_blocking[,paste0("L", 1:dd)] %>% unique()
    if(nrow(blocks_fake) != nrow(blocks_prop)){
      #cat("Adding fake coords to avoid empty blocks ~ don't like? Use lower [axis_partition]\n")
      # with current Mv, some blocks are completely empty
      # this messes with indexing. so we add completely unobserved coords
      suppressMessages(adding_blocks <- blocks_fake %>% dplyr::setdiff(blocks_prop) %>%
                         dplyr::left_join(fake_coords_blocking))
      coords_blocking <- dplyr::bind_rows(coords_blocking, adding_blocks)
      
      coords_blocking %<>% dplyr::arrange(!!!syms(paste0("Var", 1:dd)))
    }
  }
  nr_full <- nrow(coords_blocking)
  
  # DAG
  
  if(dd < 4){
    suppressMessages(parents_children <- spmeshed:::mesh_graph_build(coords_blocking %>% dplyr::select(-ix), axis_partition, F))
  } else {
    suppressMessages(parents_children <- spmeshed:::mesh_graph_build_hypercube(coords_blocking %>% dplyr::select(-ix)))
  }
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- parents_children[["names"]] 
  block_groups                 <- parents_children[["groups"]]#[order(block_names)]
  
  suppressMessages(simdata_in <- coords_blocking %>% #cbind(data.frame(ix=cbix)) %>% 
    dplyr::select(-na_which) %>% dplyr::left_join(simdata))
  #simdata[is.na(simdata$ix), "ix"] <- seq(nr_start+1, nr_full)
  
  simdata_in %<>% dplyr::arrange(!!!syms(paste0("Var", 1:dd)))
  blocking <- simdata_in$block %>% factor() %>% as.integer()
  indexing <- (1:nrow(simdata_in)-1) %>% split(blocking)
  
  if(use_forced_grid){
    #predictable_blocks <- simdata_in %>% mutate(thegrid = ifelse(is.na(thegrid), 0, 1)) %>%
    #  group_by(block) %>% summarise(thegrid=mean(thegrid, na.rm=T), perc_availab=mean(!is.na(y))) %>%
    #  mutate(predict_here = ifelse(thegrid==1, perc_availab>0, 1)) %>% arrange(block) %$% predict_here
    
    indexing_grid_ids <- simdata_in$thegrid %>% split(blocking)
    indexing_grid <- list()
    indexing_obs <- list()
    for(i in 1:length(indexing)){
      indexing_grid[[i]] <- indexing[[i]][which(indexing_grid_ids[[i]] == 1)]
      indexing_obs[[i]] <- indexing[[i]][which(indexing_grid_ids[[i]] == 0)]#[which(is.na(indexing_grid_ids[[i]]))]
    }
  } else {
    indexing_grid <- indexing
    indexing_obs <- indexing_grid
    #predictable_blocks <- rep(0, 1)
  }
  
  start_w <- rep(0, nrow(simdata_in))
  
  # finally prepare data
  sort_ix     <- simdata_in$ix
  
  y           <- simdata_in %>% dplyr::select(dplyr::contains("Y_")) %>% as.matrix()
  colnames(y) <- orig_y_colnames
  
  X           <- simdata_in %>% dplyr::select(dplyr::contains("X_")) %>% as.matrix()
  colnames(X) <- orig_X_colnames
  X[is.na(X)] <- 0 # NAs if added coords due to empty blocks
  
  na_which    <- simdata_in$na_which

  coords <- simdata_in %>% dplyr::select(dplyr::contains("Var")) %>% as.matrix()
  
  cat("Sending to MCMC > ")
  comp_time <- system.time({
      results <- spmeshed:::lmc_mgp_mcmc(y, X, coords, k,
                              
                              parents, children, 
                              block_names, block_groups,
                              
                              indexing_grid, indexing_obs,
                              
                              set_unif_bounds,
                              beta_Vi, 
                              
                              tausq_ab,
                              
                              start_w, 
                              start_sigmasq,
                              start_theta,
                              start_beta,
                              start_tausq,
                              
                              mcmc_mh_sd,
                              
                              mcmc_keep, mcmc_burn, mcmc_thin,
                              
                              n_threads,
                              
                              mcmc_adaptive, # adapting
                              
                              use_forced_grid,
                              
                              mcmc_verbose, mcmc_debug, # verbose, debug
                              mcmc_print_every, # print all iter
                              # sampling of:
                              # beta tausq sigmasq theta w
                              sample_beta, sample_tausq, 
                              sample_lambda,
                              sample_theta, sample_w) 
    })
  
  returning <- list(coords = coords,
                  sort_ix = sort_ix,
                  data = simdata_in) %>% 
    c(results)
  
  return(returning) #***
    
}

mvmesh_predict_by_block <- function(meshout, newx, newcoords, new_mv_id, n_threads=10){
  dd <- ncol(newcoords)
  pp <- length(unique(meshout$mv_id))
  k <- pp * (pp-1) / 2
  npars <- nrow(meshout$theta_mcmc) - k
  sort_ix <- 1:nrow(newcoords)
  
  # for each predicting coordinate, find which block it belongs to
  # (instead of using original partitioning (convoluted), 
  # use NN since the main algorithm is adding coordinates in empty areas so NN will pick those up)
  in_coords <- meshout$coords#$meshdata$data %>% dplyr::select(contains("Var"))
  nn_of_preds <- in_coords %>% FNN::get.knnx(newcoords, k=1, algorithm="kd_tree") %$% nn.index
  
  block_ref <- meshout$meshdata$data$block[nn_of_preds]
  
  ## by block (same block = same parents)
  newcx_by_block     <- newcoords %>% as.data.frame() %>% split(block_ref) %>% lapply(as.matrix)
  new_mv_id_by_block <- new_mv_id %>% split(block_ref) %>% lapply(as.numeric)
  newx_by_block      <- newx %>% as.data.frame() %>% split(block_ref) %>% lapply(as.matrix)
  names_by_block     <- names(newcx_by_block) %>% as.numeric()
  
  sort_ix_by_block   <- sort_ix %>% split(block_ref)
  
  result <- mvmesh_predict_by_block_base(newcx_by_block, new_mv_id_by_block, newx_by_block, 
                                    names_by_block,
                                    meshout$w_mcmc,
                                    meshout$theta_mcmc, 
                                meshout$beta_mcmc,
                                meshout$tausq_mcmc,
                                meshout$meshdata$indexing,
                                meshout$meshdata$parents_indexing,
                                meshout$meshdata$parents_children$parents,
                                meshout$coords,
                                meshout$mv_id,
                                npars, dd, pp, n_threads)
  
  sort_ix_result <- do.call(c, sort_ix_by_block)
  coords_reconstruct <- do.call(rbind, newcx_by_block)
  mv_id_reconstruct <- do.call(c, new_mv_id_by_block)
  coords_df <- cbind(coords_reconstruct, mv_id_reconstruct, block_ref) %>% as.data.frame() %>%
    rename(mv_id = mv_id_reconstruct)
  
  #coords_df <- coords_df[order(sort_ix_result),]
  w_preds <- do.call(rbind, result$w_pred)#[order(sort_ix_result),]
  y_preds <- do.call(rbind, result$y_pred)#[order(sort_ix_result),]
  

  return(list("coords_pred" = coords_df,
              "w_pred" = w_preds,
              "y_pred" = y_preds))
}


mvmesh_predict <- function(meshout, newx, newcoords, new_mv_id, n_threads=10){
  #meshdata <- meshout$meshdata
  in_coords <- meshout$coords#meshdata$data %>% dplyr::select(contains("Var"))
  dd <- ncol(in_coords)
  pp <- length(unique(meshout$mv_id))
  k <- pp * (pp-1) / 2
  npars <- nrow(meshout$theta_mcmc) - 1
  mcmc <- meshout$w_mcmc %>% length()
  
  nn_of_preds <- in_coords %>% FNN::get.knnx(newcoords, k=1, algorithm="kd_tree") %$% nn.index
  
  #coords_ref <- meshout$meshdata$blocking[nn_of_preds] #%>% arrange(!!!syms(paste0("L", 1:dd)), block) 
  block_ref <- meshout$meshdata$blocking[nn_of_preds]
  
  newcx <- newcoords %>% as.matrix()
  
  result <- meshgp:::mvmesh_predict_base(newcx, new_mv_id, newx, 
                                meshout$beta_mcmc,
                                meshout$theta_mcmc, meshout$w_mcmc,
                                meshout$tausq_mcmc,
                                meshout$meshdata$indexing,
                                meshout$meshdata$parents_indexing,
                                meshout$meshdata$parents_children$parents,
                                meshout$coords,
                                block_ref, meshout$meshdata$data$mv_id,
                                npars, dd, pp, n_threads)
  return(result)
}
