

gen_fake_coords <- function(coordsmat, thresholds, n_threads=1){
  # if some blocks become empty later, add coords at empty blocks
  # these will have no covariates and y=NA so they will be "predicted" in MCMC
  # with no effect on the estimates other than to make life easier with coding
  # some tiny slowdown will happen
  
  dd <- ncol(coordsmat)
  fake_coords <- lapply(1:dd, function(i) c(min(coordsmat[,i]), 
                                            thresholds[[i]],
                                            max(coordsmat[,i]))) %>% 
    lapply(function(vec) vec[-length(vec)] + diff(vec) / 2) %>%
    expand.grid() %>% as.matrix()
  
  blocks_by_fakecoord <- part_axis_parallel_fixed(fake_coords, thresholds, n_threads) %>% apply(2, factor)
  colnames(blocks_by_fakecoord) <- paste0("L", 1:ncol(coordsmat))
  
  block <- blocks_by_fakecoord %>% 
    as.data.frame() %>% as.list() %>% interaction()
  blockdf <- data.frame(blocks_by_fakecoord %>% apply(2, as.numeric), block=as.numeric(block))
  
  if(ncol(coordsmat)==2){
    result <- cbind(fake_coords, blockdf) #%>% 
    #mutate(color = ((L1-1)*2+(L2-1)) %% 4)
    return(result)
  } else {
    result <- cbind(fake_coords, blockdf) #%>% 
    #mutate(color = 4*(L3 %% 2) + (((L1-1)*2+(L2-1)) %% 4))
    return(result)
  }
}

tessellation_axis_parallel_fix <- function(coordsmat, thresholds, n_threads){
  
  blocks_by_coord <- apply(part_axis_parallel_fixed(coordsmat, thresholds, n_threads), 2, factor)
  colnames(blocks_by_coord) <- paste0("L", 1:ncol(coordsmat))
  
  block <- blocks_by_coord %>% 
    as.data.frame() %>% as.list() %>% interaction()
  blockdf <- data.frame(blocks_by_coord %>% apply(2, as.numeric), block=as.numeric(block))
  result <- cbind(coordsmat, blockdf)# %>% 
  #mutate(color = (L1+L2) %% 2)
  
  if(ncol(coordsmat)==2){
    result <- cbind(coordsmat, blockdf) #%>% 
    #mutate(color = ((L1-1)*2+(L2-1)) %% 4)
    return(result)
  } else {
    result <- cbind(coordsmat, blockdf) #%>% 
    #mutate(color = 4*(L3 %% 2) + (((L1-1)*2+(L2-1)) %% 4))
    return(result)
  }
}


mesh_graph_build <- function(coords_blocking, Mv, verbose=TRUE, n_threads=1, debugdag=1){
  cbl <- coords_blocking %>% dplyr::select(-dplyr::contains("Var"))
  if("L3" %in% colnames(coords_blocking)){
    cbl %<>% 
      dplyr::group_by(.data$L1, .data$L2, .data$L3, .data$block) %>% 
      dplyr::summarize(na_which = sum(.data$na_which, na.rm=TRUE)/dplyr::n())#, color=unique(color))
  } else {
    cbl %<>% 
      dplyr::group_by(.data$L1, .data$L2, .data$block) %>% 
      dplyr::summarize(na_which = sum(.data$na_which, na.rm=TRUE)/dplyr::n())#, color=unique(color))
  }
  blocks_descr <- unique(cbl) %>% as.matrix()
  
  dag_both_axes <- TRUE
  if(debugdag==1){
    graphed <- mesh_graph_cpp(blocks_descr, Mv, verbose, dag_both_axes, n_threads)
  } else {
    graphed <- mesh_graph_cpp3(blocks_descr)
  }
  #
  
  
  block_ct_obs <- coords_blocking %>% 
    dplyr::group_by(.data$block) %>% 
    dplyr::summarise(block_ct_obs = sum(.data$na_which, na.rm=TRUE)) %>% 
    dplyr::arrange(.data$block) %$% 
    block_ct_obs# %>% `[`(order(block_names))
  
  graph_blanketed <- blanket(graphed$parents, graphed$children, graphed$names, block_ct_obs)
  groups <- coloring(graph_blanketed, graphed$names, block_ct_obs)
  
  blocks_descr %<>% 
    as.data.frame() %>% 
    dplyr::arrange(.data$block) %>% 
    cbind(groups) 
  groups <- blocks_descr$groups#[order(blocks_descr$block)]
  groups[groups == -1] <- max(groups)+1
  
  return(list(parents = graphed$parents,
              children = graphed$children,
              names = graphed$names,
              groups = groups))
}



mesh_graph_build_hypercube <- function(coords_blocking){
  cbl <- coords_blocking %>% dplyr::select(-dplyr::contains("Var"))
  n_axes <- ncol(cbl) - 2
  axes_names <- paste0("L", 1:n_axes)
  
  centroids <- coords_blocking %>% 
    dplyr::select(-!!(axes_names), -.data$na_which) %>%
    dplyr::group_by(.data$block) %>% 
    dplyr::summarise_all(mean) %>% 
    dplyr::arrange(.data$block) %>% 
    dplyr::select(-.data$block) %>% 
    as.matrix()
  
  cbl %<>% 
    dplyr::group_by(!!!rlang::syms(axes_names), .data$block) %>% 
    dplyr::summarise(na_which = sum(.data$na_which, na.rm=TRUE)/dplyr::n())
  
  dimen <- ncol(cbl) - 2
  u_cbl <- unique(cbl)
  nblocks <- nrow(u_cbl)
  
  ## CPP
  bucbl_ <- u_cbl %>% 
    dplyr::arrange(!!!rlang::syms(axes_names)) 
  na_which_ <- bucbl_$na_which  
  bucbl_ %<>% 
    dplyr::select(-.data$na_which) %>% as.matrix()
  bavail_  <- u_cbl %>% 
    dplyr::filter(.data$na_which > 0) %>% as.data.frame() %>% 
    dplyr::arrange(!!!rlang::syms(axes_names)) %>% 
    dplyr::select(-.data$na_which) 
  
  avblocks <- bavail_$block %>% sort()
  avcentroids <- centroids[avblocks,]
  
  bavail_ %<>% as.matrix()
  
  system.time({
    graphed <- mesh_graph_hyper(bucbl_, bavail_, na_which_, 
                                centroids, avcentroids, avblocks)
  })
  
  # continue
  block_ct_obs <- coords_blocking %>% 
    dplyr::group_by(.data$block) %>% 
    dplyr::summarise(block_ct_obs = sum(.data$na_which, na.rm=TRUE)) %>% 
    dplyr::arrange(.data$block) %$% 
    block_ct_obs# %>% `[`(order(block_names))
  
  graph_blanketed <- blanket(graphed$parents, graphed$children, graphed$names, block_ct_obs)
  groups <- coloring(graph_blanketed, graphed$names, block_ct_obs)
  
  bucbl_ %<>% as.data.frame() %>% 
    dplyr::arrange(.data$block) %>% 
    cbind(groups) 
  groups <- bucbl_$groups
  groups[groups == -1] <- max(groups)+1
  
  return(list(parents = graphed$parents,
              children = graphed$children,
              names = graphed$names,
              groups = groups))
}
