meshedgp <- function(y, x, coords, k=NULL,
                     axis_partition = NULL, 
                     block_size = 30,
                     grid_size=NULL,
                     grid = NULL,
                   n_samples = 1000,
                   n_burnin = 100,
                   n_thin = 1,
                   n_threads = 4,
                   print_every = NULL,
                   settings    = list(adapting=T, forced_grid=NULL, saving=F),
                   prior       = list(beta=NULL, tausq=NULL,
                                      toplim = NULL, btmlim = NULL, set_unif_bounds=NULL,
                                      matern_nu=F),
                   starting    = list(beta=NULL, tausq=NULL, theta=NULL, lambda=NULL, w=NULL, 
                                      mcmcsd=.05, 
                                      mcmc_startfrom=0),
                   debug       = list(sample_beta=T, sample_tausq=T, 
                                      sample_theta=T, sample_w=T, sample_lambda=T,
                                      verbose=F, debug=F)
                   ){

  if(F){
    #y <- ylmc
    #X <- Xlmc 
    #coords <- coordslmc
    n_samples <- 5
    n_burnin <- 1
    n_thin <- 1
    n_threads <- 10
    settings    = list(adapting=T, forced_grid=NULL, saving=F)
    prior       = list(beta=NULL, tausq=NULL,
                       toplim = NULL, btmlim = NULL, set_unif_bounds=NULL)
    starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL, mcmc_startfrom=0, mcmcsd=.05)
    debug       = list(sample_beta=T, sample_tausq=T, 
                       sample_theta=T, sample_w=T, sample_lambda=T,
                       verbose=F, debug=F, print_every=1)
  }
  
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
    saving <- settings$saving %>% set_default(F)
    
    dd             <- ncol(coords)
    p              <- ncol(x)
    
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
    if(is.null(print_every)){
      if(effective_dimension > 1e6-1){
        mcmc_print_every <- 5
      } else {
        if(effective_dimension > 1e5-1){
          mcmc_print_every <- 50
        } else {
          mcmc_print_every <- 500
        }
      }
    } else {
      mcmc_print_every <- print_every
    }
    
    if(is.null(colnames(x))){
      orig_X_colnames <- colnames(x) <- paste0('X_', 1:ncol(x))
    } else {
      orig_X_colnames <- colnames(x)
      colnames(x)     <- paste0('X_', 1:ncol(x))
    }
    
    if(is.null(colnames(coords))){
      orig_coords_colnames <- colnames(coords) <- paste0('Var', 1:dd)
    } else {
      orig_coords_colnames <- colnames(coords)
      colnames(coords)     <- paste0('Var', 1:dd)
    }
    
    q              <- ncol(y)
    k              <- ifelse(is.null(k), q, k)
    # for spatial data: matern or expon, for spacetime: gneiting 2002 
    #n_par_each_process <- ifelse(dd==2, 1, 3) 
    
    nr             <- nrow(x)
    
    if(length(axis_partition) == 1){
      axis_partition <- rep(axis_partition, dd)
    }
    if(is.null(axis_partition)){
      axis_partition <- rep(round((nr/block_size)^(1/dd)), dd)
    }
    
    # -- heuristics for gridded data --
    # if we observed all unique combinations of coordinates, this is how many rows we'd have
    heuristic_gridded <- prod( coords %>% apply(2, function(x) length(unique(x))) )
    # if data are not gridded, then the above should be MUCH larger than the number of rows
    # if we're not too far off maybe the dataset is actually gridded
    if(heuristic_gridded*0.5 < nrow(coords)){
      data_likely_gridded <- T
    } else {
      data_likely_gridded <- F
    }
    if(ncol(coords) == 3){
      # with time, check if there's equal spacing
      timepoints <- coords[,3] %>% unique()
      time_spacings <- timepoints %>% sort() %>% diff() %>% round(5) %>% unique()
      if(length(time_spacings) < .1 * length(timepoints)){
        data_likely_gridded <- T
      }
    }
    
    cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    
    if(is.null(settings$forced_grid)){
      if(data_likely_gridded){
        cat("I think the data look gridded so I'm setting forced_grid=F.\n")
        use_forced_grid <- F
      } else {
        cat("I think the data don't look gridded so I'm setting forced_grid=T.\n")
        use_forced_grid <- T
      }
    } else {
      use_forced_grid <- settings$forced_grid %>% set_default(T)
      if(!use_forced_grid & !data_likely_gridded){
        warning("Data look not gridded: force a grid with settings$forced_grid=T.")
      }
    }
    
    # what are we sampling
    sample_w       <- debug$sample_w %>% set_default(T)
    sample_beta    <- debug$sample_beta %>% set_default(T)
    sample_tausq   <- debug$sample_tausq %>% set_default(T)
    sample_theta   <- debug$sample_theta %>% set_default(T)
    sample_lambda  <- debug$sample_lambda %>% set_default(T)

  }

  # data management pt 2
  if(1){
    yrownas <- apply(y, 1, function(i) ifelse(sum(is.na(i))==q, NA, 1))
    na_which <- ifelse(!is.na(yrownas), 1, NA)
    simdata <- data.frame(ix=1:nrow(coords)) %>% 
      cbind(coords, y, na_which, x) %>% 
      as.data.frame()
    
    #####
    cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    cat("{q} outcome variables on {nrow(unique(coords))} unique locations." %>% glue::glue())
    cat("\n")
    if(use_forced_grid){ 
      # user showed intention to use fixed grid
      if(!is.null(grid)){
        gridcoords_lmc <- grid
      } else {
        gs <- round(nrow(coords)^(1/ncol(coords)))
        gsize <- if(is.null(grid_size)){ rep(gs, ncol(coords)) } else { grid_size }
        
        xgrids <- list()
        for(j in 1:dd){
          xgrids[[j]] <- seq(min(coords[,j]), max(coords[,j]), length.out=gsize[j])
        }
        
        gridcoords_lmc <- expand.grid(xgrids)
      }
      
      cat("Forced grid built with {nrow(gridcoords_lmc)} locations." %>% glue::glue())
      cat("\n")
      simdata <- dplyr::bind_rows(simdata %>% dplyr::mutate(thegrid=0), 
                           gridcoords_lmc %>% dplyr::mutate(thegrid=1))
      
      absize <- round(nrow(gridcoords_lmc)/prod(axis_partition))
    } else {
      if(length(axis_partition) < ncol(coords)){
        stop("Error: axis_partition not specified for all axes.")
      }
      simdata %<>% 
        dplyr::mutate(thegrid = 0)
      absize <- round(nrow(simdata)/prod(axis_partition))
    }
    cat("Partitioning grid axes into {paste0(axis_partition, collapse=', ')} intervals. Approx block size {absize}" %>% glue::glue())
    cat("\n")
    
    cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    
    simdata %<>% 
      dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
    
    coords <- simdata %>% 
      dplyr::select(dplyr::contains("Var")) %>% 
      as.matrix()
    sort_ix     <- simdata$ix
    
    # Domain partitioning and gibbs groups
    if(use_forced_grid){
      gridded_coords <- simdata %>% dplyr::filter(.data$thegrid==1) %>% dplyr::select(dplyr::contains("Var")) %>% as.matrix()
      fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(gridded_coords[,i], axis_partition[i])) 
    } else {
      fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], axis_partition[i])) 
    }
    
    # guaranteed to produce blocks using Mv
    system.time(fake_coords_blocking <- coords %>% 
                  as.matrix() %>% 
                  gen_fake_coords(fixed_thresholds, 1) )
    
    # Domain partitioning and gibbs groups
    system.time(coords_blocking <- coords %>% 
                  as.matrix() %>%
                  tessellation_axis_parallel_fix(fixed_thresholds, 1) %>% 
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
      
      coords_blocking %<>% dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
    }
    
  }
  nr_full <- nrow(coords_blocking)
  
  # DAG
  if(dd < 4){
    suppressMessages(parents_children <- mesh_graph_build(coords_blocking %>% dplyr::select(-.data$ix), axis_partition, F))
  } else {
    suppressMessages(parents_children <- mesh_graph_build_hypercube(coords_blocking %>% dplyr::select(-.data$ix)))
  }
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- parents_children[["names"]] 
  block_groups                 <- parents_children[["groups"]]#[order(block_names)]
  
  suppressMessages(simdata_in <- coords_blocking %>% #cbind(data.frame(ix=cbix)) %>% 
    dplyr::select(-na_which) %>% dplyr::left_join(simdata))
  #simdata[is.na(simdata$ix), "ix"] <- seq(nr_start+1, nr_full)
  
  simdata_in %<>% 
    dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
  blocking <- simdata_in$block %>% 
    factor() %>% as.integer()
  indexing <- (1:nrow(simdata_in)-1) %>% 
    split(blocking)
  
  if(use_forced_grid){
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
  }
  
  
  if(T){
    # prior and starting values for mcmc
    matern_nu <- prior$matern_nu %>% set_default(F)
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(x)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$tausq)){
      tausq_ab <- c(2.01, 1)
    } else {
      tausq_ab <- prior$tausq
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
    
    # starting values
    if(is.null(starting$beta)){
      start_beta   <- matrix(0, nrow=p, ncol=q)
    } else {
      start_beta   <- starting$beta
    }
    
    if(is.null(prior$set_unif_bounds)){
      if(dd == 2){
        if(matern_nu){
          start_theta <- matrix(2, ncol=k, nrow=2) 
          start_theta[2,] <- 0.5
          
          set_unif_bounds <- matrix(0, nrow=2*k, ncol=2)
          set_unif_bounds[1,1] <- btmlim
          set_unif_bounds[1,2] <- toplim
          set_unif_bounds[2*(1:k),] <- matrix(c(0.1, 3-1e-3),nrow=1) %x% matrix(1, nrow=k)
          
        } else {
          start_theta <- matrix(2, ncol=k, nrow=1) 
          
          set_unif_bounds <- matrix(0, nrow=k, ncol=2)
          set_unif_bounds[,1] <- btmlim
          set_unif_bounds[,2] <- toplim
        }
        
      } else {
        start_theta <- matrix(2, ncol=k, nrow=3) 
        start_theta[3,] <- .5 # separability parameter
        
        set_unif_bounds <- matrix(0, nrow=3*k, ncol=2)
        set_unif_bounds[,1] <- btmlim
        set_unif_bounds[,2] <- toplim
        set_unif_bounds[3*(1:k),] <- matrix(c(btmlim, 1-btmlim),nrow=1) %x% matrix(1, nrow=k)
        
      }
      
    } else {
      set_unif_bounds <- prior$set_unif_bounds
    }
    
    # override defaults if starting values are provided
    if(!is.null(starting$theta)){
      start_theta <- starting$theta
    }
    
    n_par_each_process <- nrow(start_theta)
    if(length(starting$mcmcsd) == 1){
      mcmc_mh_sd <- diag(k * n_par_each_process) * starting$mcmcsd
    } else {
      mcmc_mh_sd <- starting$mcmcsd
    }
    
    if(is.null(starting$tausq)){
      start_tausq  <- rep(.1, q)
    } else {
      start_tausq  <- starting$tausq
    }
    
    if(is.null(starting$lambda)){
      start_lambda <- matrix(0, nrow=q, ncol=k)
      diag(start_lambda) <- 10
    } else {
      start_lambda <- starting$lambda
    }
    
    if(is.null(starting$lambda_mask)){
      lambda_mask <- matrix(0, nrow=q, ncol=k)
      lambda_mask[lower.tri(lambda_mask)] <- 1
      diag(lambda_mask) <- 1
    } else {
      lambda_mask <- starting$lambda_mask
    }
    
    if(is.null(starting$mcmc_startfrom)){
      mcmc_startfrom <- 0
    } else {
      mcmc_startfrom <- starting$mcmc_startfrom
    }
    
    if(is.null(starting$w)){
      start_w <- matrix(0, nrow = nrow(simdata_in), ncol = k)
    } else {
      # this is used to restart MCMC
      # assumes the ordering and the sizing is correct, 
      # so no change is necessary and will be input directly to mcmc
      start_w <- starting$w #%>% matrix(ncol=q)
    }
  }
  
  # finally prepare data
  sort_ix <- simdata_in$ix
  
  y <- simdata_in %>% 
    dplyr::select(dplyr::contains("Y_")) %>% 
    as.matrix()
  colnames(y) <- orig_y_colnames
  
  x <- simdata_in %>% 
    dplyr::select(dplyr::contains("X_")) %>% 
    as.matrix()
  colnames(x) <- orig_X_colnames
  x[is.na(x)] <- 0 # NAs if added coords due to empty blocks
  
  na_which <- simdata_in$na_which

  coords <- simdata_in %>% 
    dplyr::select(dplyr::contains("Var")) %>% 
    as.matrix()
  
  cat("Sending to MCMC > ")
  
  mcmc_run <- lmc_mgp_mcmc
  
  comp_time <- system.time({
      results <- mcmc_run(y, x, coords, k,
                              
                              parents, children, 
                              block_names, block_groups,
                              
                              indexing_grid, indexing_obs,
                              
                              set_unif_bounds,
                              beta_Vi, 
                              
                              tausq_ab,
                              
                              start_w, 
                          
                              start_lambda,
                              lambda_mask,
                          
                              start_theta,
                              start_beta,
                              start_tausq,
                              
                              mcmc_mh_sd,
                              
                              mcmc_keep, mcmc_burn, mcmc_thin,
                          
                              mcmc_startfrom,
                              
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
  
  if(saving){
    
    listN <- function(...){
      anonList <- list(...)
      names(anonList) <- as.character(substitute(list(...)))[-1]
      anonList
    }
    
    saved <- listN(y, x, coords, k,
      
      parents, children, 
      block_names, block_groups,
      
      indexing_grid, indexing_obs,
      
      set_unif_bounds,
      beta_Vi, 
      
      tausq_ab,
      
      start_w, 
      
      start_lambda,
      lambda_mask,
      
      start_theta,
      start_beta,
      start_tausq,
      
      mcmc_mh_sd,
      
      mcmc_keep, mcmc_burn, mcmc_thin,
      
      mcmc_startfrom,
      
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
  } else {
    saved <- "Model data not saved."
  }
  
  coords_renamer <- colnames(coords)
  names(coords_renamer) <- orig_coords_colnames
  
  returning <- list(coordsdata = simdata_in %>% 
                      dplyr::select(1:dd, .data$thegrid) %>%
                      dplyr::rename(!!!coords_renamer,
                        forced_grid=.data$thegrid),
                    savedata = saved) %>% 
    c(results)
  return(returning) 
    
}


meshedgp_restart <- function(meshedgp_output, 
                             n_samples = 1000,
                             n_thin = NULL,
                             print_every = NULL){
  
  # assumes no burnin -- isnt that the whole point?
  
  list2env(meshedgp_output, env = environment())
  list2env(meshedgp_output$savedata, env = environment())

  # update starting values based on previous chain
  n_tot_mcmc <- dim(beta_mcmc)[3]
  
  p <- ncol(x)
  q <- ncol(y)
  
  start_beta <- beta_mcmc[,, n_tot_mcmc] %>% matrix(ncol=q)
  start_tausq <- tausq_mcmc[, n_tot_mcmc] %>% matrix(ncol=q)
  start_theta <- theta_mcmc[,, n_tot_mcmc] %>% matrix(ncol=k)
  start_w <- w_mcmc[[mcmc_keep]] %>% matrix(ncol=k)   
  start_lambda <- lambda_mcmc[,, n_tot_mcmc] %>% matrix(ncol=k)
  
  mcmc_startfrom <- n_tot_mcmc
  mcmc_mh_sd <- paramsd
  
  # new mcmc setup
  if(!is.null(print_every)){
    mcmc_print_every <- print_every
  }
  if(!is.null(n_thin)){
    mcmc_thin <- n_thin
  }
  
  mcmc_burn <- 0
  mcmc_keep <- n_samples
  
  cat("Sending to MCMC > ")
  
  mcmc_run <- lmc_mgp_mcmc
  comp_time <- system.time({
    results <- mcmc_run(y, x, coords, k,
                        
                        parents, children, 
                        block_names, block_groups,
                        
                        indexing_grid, indexing_obs,
                        
                        set_unif_bounds,
                        beta_Vi, 
                        
                        tausq_ab,
                        
                        start_w, 
                        
                        start_lambda,
                        lambda_mask,
                        
                        start_theta,
                        start_beta,
                        start_tausq,
                        
                        mcmc_mh_sd,
                        
                        mcmc_keep, mcmc_burn, mcmc_thin,
                        
                        mcmc_startfrom,
                        
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

  listN <- function(...){
    anonList <- list(...)
    names(anonList) <- as.character(substitute(list(...)))[-1]
    anonList
  }
  
  saved <- listN(y, x, coords, k,
                 parents, children, 
                 block_names, block_groups,
                 indexing_grid, indexing_obs,
                 set_unif_bounds,
                 beta_Vi, 
                 tausq_ab,
                 start_w, 
                 start_lambda,
                 lambda_mask,
                 start_theta,
                 start_beta,
                 start_tausq,
                 mcmc_mh_sd,
                 mcmc_keep, mcmc_burn, mcmc_thin,
                 mcmc_startfrom,
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

  returning <- list(coordsdata = coordsdata,
                    savedata = saved) %>% 
    c(results)
  return(returning) 
}