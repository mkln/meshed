
library(tidyverse)
library(magrittr)

coords <- expand.grid(xx <- seq(0, 1, length.out=100), xx, seq(0, 1, length.out=30)) %>% 
  arrange(Var1, Var2, Var3) %>% as.matrix()
axis_partition <- c(100, 10, 30)

simdata <- rmeshedgp(coords, axis_partition, c(1, .5, 1))

colnames(test)[4] <- "w"
test %<>% as.data.frame()

for(tt in 1:length(unique(test$Var3))){
  plotted <- test %>% filter(Var3==unique(Var3)[tt]) %>% 
    ggplot(aes(Var1, Var2, fill=w)) + geom_raster() + scale_fill_viridis_c() +
    theme_void() +
    theme(legend.position="none") + ggtitle(tt)
  
  fname <- sprintf("~/spmeshed_files/plot_tests/%02d.png", tt)
  ggsave(plot=plotted, filename=fname, width=7, height=7)
}
system("convert -delay 25 -loop 0 -resize 350 ~/spmeshed_files/plot_tests/*.png ~/spmeshed_files/plot_tests/out_alt.gif")

