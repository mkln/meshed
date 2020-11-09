
mesh_graph_build <- function(coords_blocking, Mv, verbose=T){
  cbl <- coords_blocking %>% dplyr::select(-dplyr::contains("Var"))
  if("L3" %in% colnames(coords_blocking)){
    cbl %<>% dplyr::group_by(L1, L2, L3, block) %>% dplyr::summarize(na_which = sum(na_which, na.rm=T)/n())#, color=unique(color))
  } else {
    cbl %<>% dplyr::group_by(L1, L2, block) %>% dplyr::summarize(na_which = sum(na_which, na.rm=T)/n())#, color=unique(color))
  }
  blocks_descr <- unique(cbl) %>% as.matrix()
  
  graphed <- spmeshed:::mesh_graph_cpp(blocks_descr, Mv, verbose)
  
  block_ct_obs <- coords_blocking %>% dplyr::group_by(block) %>% 
    dplyr::summarise(block_ct_obs = sum(na_which, na.rm=T)) %>% 
    dplyr::arrange(block) %$% 
    block_ct_obs# %>% `[`(order(block_names))
  
  graph_blanketed <- spmeshed:::blanket(graphed$parents, graphed$children, graphed$names, block_ct_obs)
  groups <- spmeshed:::coloring(graph_blanketed, graphed$names, block_ct_obs)
  
  blocks_descr %<>% as.data.frame() %>% 
    dplyr::arrange(block) %>% 
    cbind(groups) 
  groups <- blocks_descr$groups#[order(blocks_descr$block)]
  groups[groups == -1] <- max(groups)+1
  
  list2env(graphed, environment())
  return(list(parents = parents,
              children = children,
              names = names,
              groups = groups))
}



mesh_graph_build_hypercube <- function(coords_blocking){
  cbl <- coords_blocking %>% dplyr::select(-dplyr::contains("Var"))
  n_axes <- ncol(cbl) - 2
  axes_names <- paste0("L", 1:n_axes)
  
  centroids <- coords_blocking %>% dplyr::select(-!!(axes_names), -na_which) %>%
    dplyr::group_by(block) %>% 
    dplyr::summarise_all(mean) %>% 
    dplyr::arrange(block) %>% 
    dplyr::select(-block) %>% 
    as.matrix()
  
  cbl %<>% dplyr::group_by(!!!rlang::syms(axes_names), block) %>% 
    dplyr::summarise(na_which = sum(na_which, na.rm=T)/n())
  
  dimen <- ncol(cbl) - 2
  u_cbl <- unique(cbl)
  nblocks <- nrow(u_cbl)
  
  ## CPP
  bucbl_ <- u_cbl %>% dplyr::arrange(!!!syms(axes_names)) 
  na_which_ <- bucbl_$na_which  
  bucbl_ %<>% dplyr::select(-na_which) %>% as.matrix()
  bavail_  <- u_cbl %>% dplyr::filter(na_which > 0) %>% as.data.frame() %>% 
    dplyr::arrange(!!!rlang::syms(axes_names)) %>% 
    dplyr::select(-na_which) 
  
  avblocks <- bavail_$block %>% sort()
  avcentroids <- centroids[avblocks,]
  
  bavail_ %<>% as.matrix()
  
  system.time({
    graphed <- spmeshed:::mesh_graph_hyper(bucbl_, bavail_, na_which_, 
                                centroids, avcentroids, avblocks)
  })
  
  # continue
  block_ct_obs <- coords_blocking %>% 
    dplyr::group_by(block) %>% 
    dplyr::summarise(block_ct_obs = sum(na_which, na.rm=T)) %>% 
    dplyr::arrange(block) %$% 
    block_ct_obs# %>% `[`(order(block_names))
  
  graph_blanketed <- spmeshed:::blanket(graphed$parents, graphed$children, graphed$names, block_ct_obs)
  groups <- spmeshed:::coloring(graph_blanketed, graphed$names, block_ct_obs)
  
  bucbl_ %<>% as.data.frame() %>% 
    dplyr::arrange(block) %>% 
    cbind(groups) 
  groups <- bucbl_$groups
  groups[groups == -1] <- max(groups)+1
  
  return(list(parents = graphed$parents,
              children = graphed$children,
              names = graphed$names,
              groups = groups))
}