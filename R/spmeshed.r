spmeshed <- function(y, x, coords, k=NULL,
             family = "gaussian",
             axis_partition = NULL, 
             block_size = 30,
             grid_size=NULL,
             grid_custom = NULL,
             n_samples = 1000,
             n_burnin = 100,
             n_thin = 1,
             n_threads = 4,
             verbose = 0,
             predict_everywhere = FALSE,
             settings = list(adapting=TRUE, forced_grid=NULL, cache=NULL, 
                                ps=TRUE, saving=TRUE, low_mem=FALSE, hmc=4),
             prior = list(beta=NULL, tausq=NULL, sigmasq = NULL,
                          phi=NULL, a=NULL, nu = NULL,
                          toplim = NULL, btmlim = NULL, set_unif_bounds=NULL),
             starting = list(beta=NULL, tausq=NULL, theta=NULL, 
                             lambda=NULL, v=NULL,  a=NULL, nu = NULL,
                             mcmcsd=.05, mcmc_startfrom=0),
             debug = list(sample_beta=TRUE, sample_tausq=TRUE, 
                          sample_theta=TRUE, sample_w=TRUE, sample_lambda=TRUE,
                          verbose=FALSE, debug=FALSE),
             indpart=FALSE
){

  # init
  if(verbose > 0){
    cat("Bayesian Meshed GP regression model fit via Markov chain Monte Carlo\n")
  }
  model_tag <- "Bayesian Meshed GP regression model\n
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o\n(Markov chain Monte Carlo)\n"
  
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
    
    which_hmc    <- settings$hmc %>% set_default(4)
    if(which_hmc > 4){
      warning("Invalid HMC algorithm choice. Choose settings$hmc in {1,2,3,4}")
      which_hmc <- 4
    }

    mcmc_adaptive    <- settings$adapting %>% set_default(TRUE)
    mcmc_verbose     <- debug$verbose %>% set_default(FALSE)
    mcmc_debug       <- debug$debug %>% set_default(FALSE)
    saving <- settings$saving %>% set_default(TRUE)
    low_mem <- settings$low_mem %>% set_default(FALSE)
    
    debugdag <- 1#debug$dag %>% set_default(1)
    
    coords %<>% as.matrix()
    
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
    
    if(verbose == 0){
      mcmc_print_every <- 0
    } else {
      if(verbose <= 20){
        mcmc_tot <- mcmc_burn + mcmc_thin * mcmc_keep
        mcmc_print_every <- 1+round(mcmc_tot / verbose)
      } else {
        if(is.infinite(verbose)){
          mcmc_print_every <- 1
        } else {
          mcmc_print_every <- verbose
        }
      }
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
    
    # family id 
    family <- if(length(family)==1){rep(family, q)} else {family}
    
    if(all(family == "gaussian")){ 
      use_ps <- settings$ps %>% set_default(TRUE)
    } else {
      use_ps <- settings$ps %>% set_default(FALSE)
    }
    
    family_in <- data.frame(family=family)
    available_families <- data.frame(id=0:4, family=c("gaussian", "poisson", "binomial", "beta", "negbinomial"))
    
    suppressMessages(family_id <- family_in %>% 
                       left_join(available_families, by=c("family"="family")) %>% pull(.data$id))
    
    latent <- "gaussian"
    if(!(latent %in% c("gaussian"))){
      stop("Latent process not recognized. Choose 'gaussian'")
    }
    
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
      data_likely_gridded <- TRUE
    } else {
      data_likely_gridded <- FALSE
    }
    if(ncol(coords) == 3){
      # with time, check if there's equal spacing
      timepoints <- coords[,3] %>% unique()
      time_spacings <- timepoints %>% sort() %>% diff() %>% round(5) %>% unique()
      if(length(time_spacings) < .1 * length(timepoints)){
        data_likely_gridded <- TRUE
      }
    }
    
    if(is.null(settings$forced_grid)){
      if(data_likely_gridded){
        #cat("I think the data look gridded so I'm setting forced_grid=FALSE.\n")
        use_forced_grid <- FALSE
      } else {
        #cat("I think the data don't look gridded so I'm setting forced_grid=TRUE.\n")
        use_forced_grid <- TRUE
      }
    } else {
      use_forced_grid <- settings$forced_grid %>% set_default(TRUE)
      #if(!use_forced_grid & !data_likely_gridded){
      #  warning("Data look not gridded: force a grid with settings$forced_grid=T.")
      #}
    }
    
    
    use_cache <- settings$cache %>% set_default(TRUE)
    if(use_forced_grid & (!use_cache)){
      warning("Using a forced grid with no cache is a waste of resources.")
    }
     
    # what are we sampling
    sample_w       <- debug$sample_w %>% set_default(TRUE)
    sample_beta    <- debug$sample_beta %>% set_default(TRUE)
    sample_tausq   <- debug$sample_tausq %>% set_default(TRUE)
    sample_theta   <- debug$sample_theta %>% set_default(TRUE)
    sample_lambda  <- debug$sample_lambda %>% set_default(TRUE)

  }

  # data management pt 2
  if(1){
    yrownas <- apply(y, 1, function(i) ifelse(sum(is.na(i))==q, NA, 1))
    na_which <- ifelse(!is.na(yrownas), 1, NA)
    simdata <- data.frame(ix=1:nrow(coords)) %>% 
      cbind(coords, y, na_which, x) %>% 
      as.data.frame()
    
    #####
    #cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    #cat("{q} outcome variables on {nrow(unique(coords))} unique locations." %>% glue::glue())
    #cat("\n")
    if(use_forced_grid){ 
      # user showed intention to use fixed grid
      if(!is.null(grid_custom$grid)){
        grid <- grid_custom$grid
        gridcoords_lmc <- grid %>% as.data.frame()
        colnames(gridcoords_lmc)[1:dd] <- colnames(coords)
        if(ncol(grid) == dd + p){
          # we have the covariate values at the reference grid, so let's use them to make predictions
          colnames(gridcoords_lmc)[-(1:dd)] <- colnames(x)
        }
      } else {
        gs <- round(nrow(coords)^(1/ncol(coords)))
        gsize <- if(is.null(grid_size)){ rep(gs, ncol(coords)) } else { grid_size }
        
        xgrids <- list()
        for(j in 1:dd){
          xgrids[[j]] <- seq(min(coords[,j]), max(coords[,j]), length.out=gsize[j])
        }
        
        gridcoords_lmc <- expand.grid(xgrids)
      }
      
      #cat("Forced grid built with {nrow(gridcoords_lmc)} locations." %>% glue::glue())
      #cat("\n")
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
    #cat("Partitioning grid axes into {paste0(axis_partition, collapse=', ')} intervals. Approx block size {absize}" %>% glue::glue())
    #cat("\n")
    
    #cat("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n")
    
    simdata %<>% 
      dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
    
    coords <- simdata %>% 
      dplyr::select(dplyr::contains("Var")) %>% 
      as.matrix()
    sort_ix     <- simdata$ix
    
    # Domain partitioning and gibbs groups
    if(use_forced_grid){
      if(!is.null(grid_custom$axis_interval_partition)){
        fixed_thresholds <- grid_custom$axis_interval_partition
        axis_partition <- sapply(fixed_thresholds, function(x) length(x) + 1)
      } else {
        gridded_coords <- simdata %>% dplyr::filter(.data$thegrid==1) %>% dplyr::select(dplyr::contains("Var")) %>% as.matrix()
        fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(gridded_coords[,i], axis_partition[i])) 
      }
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
    graph_time <- system.time({
      parents_children <- mesh_graph_build(coords_blocking %>% dplyr::select(-.data$ix), axis_partition, FALSE, n_threads, debugdag)
      })
  } else {
    graph_time <- system.time({
      parents_children <- mesh_graph_build_hypercube(coords_blocking %>% dplyr::select(-.data$ix))
    })
  }
  
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- parents_children[["names"]] 
  block_groups                 <- parents_children[["groups"]]#[order(block_names)]

  if(indpart){
    parents %<>% lapply(function(x) numeric(0))
    children %<>% lapply(function(x) numeric(0))
    block_groups %<>% rep(0, length(block_groups))
  }
  
  
  # these two lines remove the DAG and make all blocks independent
  #parents %<>% lapply(function(x) x[x==-1]) 
  #children %<>% lapply(function(x) x[x==-1])
  
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
      if(predict_everywhere){
        indexing_obs[[i]] <- indexing[[i]]
      } else {
        indexing_obs[[i]] <- indexing[[i]][which(indexing_grid_ids[[i]] == 0)]
      }
    }
  } else {
    indexing_grid <- indexing
    indexing_obs <- indexing_grid
  }
  
  if(1){
    # prior and starting values for mcmc
    
    # nu
    if(is.null(prior$nu)){
      matern_nu <- FALSE
      if(is.null(starting$nu)){
        start_nu <- 0.5
        matern_fix_twonu <- 1
      } else {
        start_nu <- starting$nu
        if(start_nu %in% c(0.5, 1.5, 2.5)){
          matern_fix_twonu <- 2 * start_nu
        }
      }
    } else {
      nu_limits <- prior$nu
      if(length(nu_limits)==1){
        matern_fix_twonu <- floor(nu_limits)*2 + 1
        start_nu <- matern_fix_twonu
        matern_nu <- FALSE
        strmessage <- paste0("nu set to ", start_nu/2)
        message(strmessage)
      } else {
        if(diff(nu_limits) == 0){
          matern_fix_twonu <- floor(nu_limits[1])*2 + 1
          start_nu <- matern_fix_twonu
          matern_nu <- F
          strmessage <- paste0("nu set to ", start_nu/2)
          message(strmessage)
        } else {
          if(is.null(starting$nu)){
            start_nu <- mean(nu_limits)
          } else {
            start_nu <- starting$nu
          }
          matern_nu <- TRUE
          matern_fix_twonu <- 1 # not applicable
        }
      }
      
    }
    
    if(is.null(prior$phi)){
      stop("Need to specify the limits on the Uniform prior for phi via prior$phi.")
    }
    phi_limits <- prior$phi
    if(is.null(starting$phi)){
      start_phi <- mean(phi_limits)
    } else {
      start_phi <- starting$phi
    }
    
    if(dd == 3){
      if(is.null(prior$a)){
        a_limits <- phi_limits
      } else {
        a_limits <- prior$a
      }
      if(is.null(starting$a)){
        start_a <- mean(a_limits)
      } else {
        start_a <- starting$a
      }
    }
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(x)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$tausq)){
      tausq_ab <- c(2, 1)
    } else {
      tausq_ab <- prior$tausq
      if(length(tausq_ab) == 1){
        tausq_ab <- c(tausq_ab[1], 0)
      }
    }
    
    if(is.null(prior$sigmasq)){
      sigmasq_ab <- c(2, 1)
    } else {
      sigmasq_ab <- prior$sigmasq
    }
    
    btmlim <- prior$btmlim %>% set_default(1e-3)
    toplim <- prior$toplim %>% set_default(1e3)
    
    # starting values
    if(is.null(starting$beta)){
      start_beta   <- matrix(0, nrow=p, ncol=q)
    } else {
      start_beta   <- starting$beta
    }
    
    if(is.null(prior$set_unif_bounds)){
      if(dd == 2){
        if(matern_nu){
          theta_names <- c("phi", "nu", "sigmasq")
          npar <- length(theta_names)
          
          start_theta <- matrix(0, ncol=k, nrow=npar) 
          set_unif_bounds <- matrix(0, nrow=npar*k, ncol=2)
          
          # phi
          set_unif_bounds[seq(1, npar*k, npar),] <- matrix(phi_limits,nrow=1) %x% matrix(1, nrow=k)
          start_theta[1,] <- start_phi
          
          # nu
          set_unif_bounds[seq(2, npar*k, npar),] <- matrix(nu_limits,nrow=1) %x% matrix(1, nrow=k)
          start_theta[2,] <- start_nu
          
          # sigmasq expansion
          set_unif_bounds[seq(3, npar*k, npar),] <- matrix(c(btmlim, toplim),nrow=1) %x% matrix(1, nrow=k)
          if((q>1) & (!use_ps)){
            # multivariate without expansion: sigmasq=1 fixed.
            start_theta[3,] <- 1
          } else {
            start_theta[3,] <- btmlim + 1  
          }
          
        } else {
          theta_names <- c("phi", "sigmasq")
          npar <- length(theta_names)
          
          start_theta <- matrix(0, ncol=k, nrow=npar) 
          
          set_unif_bounds <- matrix(0, nrow=npar*k, ncol=2)
          # phi
          set_unif_bounds[seq(1, npar*k, npar),] <- matrix(phi_limits,nrow=1) %x% matrix(1, nrow=k)
          start_theta[1,] <- start_phi
          
          # sigmasq expansion
          set_unif_bounds[seq(2, npar*k, npar),] <- matrix(c(btmlim, toplim),nrow=1) %x% matrix(1, nrow=k)
          if((q>1) & (!use_ps)){
            # multivariate without expansion: sigmasq=1 fixed.
            start_theta[2,] <- 1
          } else {
            start_theta[2,] <- btmlim + 1  
          }
        }
      } else {
        theta_names <- c("a", "phi", "beta", "sigmasq")
        npar <- length(theta_names)
        
        start_theta <- matrix(0, ncol=k, nrow=npar) 
        set_unif_bounds <- matrix(0, nrow=npar*k, ncol=2)
        
        # a
        set_unif_bounds[seq(1, npar*k, npar),] <- matrix(a_limits,nrow=1) %x% matrix(1, nrow=k)
        start_theta[1,] <- start_a
        
        # phi
        set_unif_bounds[seq(2, npar*k, npar),] <- matrix(phi_limits,nrow=1) %x% matrix(1, nrow=k)
        start_theta[2,] <- start_phi
        
        # beta
        set_unif_bounds[seq(3, npar*k, npar),] <- matrix(c(0,1),nrow=1) %x% matrix(1, nrow=k)
        start_theta[3,] <- 0.5
        
        # sigmasq expansion
        set_unif_bounds[seq(4, npar*k, npar),] <- matrix(c(btmlim, toplim),nrow=1) %x% matrix(1, nrow=k)
        if((q>1) & (!use_ps)){
          # multivariate without expansion: sigmasq=1 fixed.
          start_theta[4,] <- 1
        } else {
          start_theta[4,] <- btmlim + 1  
        }
        
      }
      
    } else {
      set_unif_bounds <- prior$set_unif_bounds
    }
    
    # override defaults if starting values are provided
    if(!is.null(starting$theta)){
      start_theta <- starting$theta
    }
    
    n_par_each_process <- nrow(start_theta)
    if(is.null(starting$mcmcsd)){
      mcmc_mh_sd <- diag(k * n_par_each_process) * 0.05
    } else {
      if(length(starting$mcmcsd) == 1){
        mcmc_mh_sd <- diag(k * n_par_each_process) * starting$mcmcsd
      } else {
        mcmc_mh_sd <- starting$mcmcsd
      }
    }
    
    if(is.null(starting$tausq)){
      start_tausq  <- family %>% sapply(function(ff) if(ff == "gaussian"){.1} else {1})
    } else {
      start_tausq  <- starting$tausq
    }
    
    if(is.null(starting$lambda)){
      start_lambda <- matrix(0, nrow=q, ncol=k)
      diag(start_lambda) <- if(use_ps){10} else {1}
    } else {
      start_lambda <- starting$lambda
    }
    
    if(is.null(starting$lambda_mask)){
      if(k<=q){
        lambda_mask <- matrix(0, nrow=q, ncol=k)
        lambda_mask[lower.tri(lambda_mask)] <- 1
        diag(lambda_mask) <- 1 #*** 
      } else {
        stop("starting$lambda_mask needs to be specified")
      }
    } else {
      lambda_mask <- starting$lambda_mask
    }
    
    if(is.null(starting$mcmc_startfrom)){
      mcmc_startfrom <- 0
    } else {
      mcmc_startfrom <- starting$mcmc_startfrom
    }
    
    if(is.null(starting$v)){
      start_v <- matrix(0, nrow = nrow(simdata_in), ncol = k)
    } else {
      # this is used to restart MCMC
      # assumes the ordering and the sizing is correct, 
      # so no change is necessary and will be input directly to mcmc
      start_v <- starting$v #%>% matrix(ncol=q)
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
  
  
  coords_renamer <- colnames(coords)
  names(coords_renamer) <- orig_coords_colnames
  
  coordsdata <- simdata_in %>% 
    dplyr::select(1:dd, .data$thegrid) %>%
    dplyr::rename(!!!coords_renamer,
                  forced_grid=.data$thegrid)
  
  ## checking
  if(use_forced_grid){
    suppressMessages(checking <- coordsdata %>% left_join(coords_blocking) %>% 
      group_by(.data$block) %>% summarise(nfg = sum(.data$forced_grid)) %>% filter(.data$nfg==0))
    if(nrow(checking) > 0){
      stop("Partition is too fine for the current reference set. ")
    }
  }
  
  if(verbose > 0){
    cat("Sending to MCMC.\n")
  }
  
  mcmc_run <- meshed_mcmc
  
  comp_time <- system.time({
      results <- mcmc_run(y, family_id, x, coords, k,
                              
                              parents, children, 
                              block_names, block_groups,
                              
                              indexing_grid, indexing_obs,
                              
                              set_unif_bounds,
                              beta_Vi, 
                              
                          
                              sigmasq_ab,
                              tausq_ab,
                          
                              matern_fix_twonu,
                              
                              start_v, 
                          
                              start_lambda,
                              lambda_mask,
                          
                              start_theta,
                              start_beta,
                              start_tausq,
                              
                              mcmc_mh_sd,
                              
                              mcmc_keep, mcmc_burn, mcmc_thin,
                          
                              mcmc_startfrom,
                              
                              n_threads,
                              
                              which_hmc,
                              mcmc_adaptive, # adapting
                              
                              use_cache,
                              use_forced_grid,
                              use_ps,
                              
                              mcmc_verbose, mcmc_debug, # verbose, debug
                              mcmc_print_every, # print all iter
                              low_mem,
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
    
    saved <- listN(y, x, coords_blocking, k,
      
                   family,
      parents, children, 
      block_names, block_groups,
      
      indexing_grid, indexing_obs,
      
      set_unif_bounds,
      beta_Vi, 
      
      tausq_ab,
      
      start_v, 
      
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
      use_ps,
      matern_fix_twonu,
      
      mcmc_verbose, mcmc_debug, # verbose, debug
      mcmc_print_every, # print all iter
      # sampling of:
      # beta tausq sigmasq theta w
      sample_beta, sample_tausq, 
      sample_lambda,
      sample_theta, sample_w,
      fixed_thresholds)
  } else {
    saved <- "Model data not saved."
  }
  
  returning <- list(coordsdata = coordsdata,
                    savedata = saved) %>% 
    c(results)
  
  class(returning) <- "spmeshed"
  
  return(returning) 
    
}


