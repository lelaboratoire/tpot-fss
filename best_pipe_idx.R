library(tidyverse)
get_optimal_idx <- function(accu.df){
  opt.accu <- accu.df %>%
    dplyr::select(`Training CV Accuracy`) %>% 
    as.matrix %>% quantile(probs = 0.9)
  opt.idx <- which.min(abs(accu.df$`Training CV Accuracy`- opt.accu))
  print(accu.df[opt.idx, ])
}

accu.real.ds.dgm5 <- read_csv(paste0("RNASeq/accuracyDF.csv")) %>% dplyr::filter(subidx == 4)
accu.real.ds.dgm13 <- read_csv(paste0("RNASeq/accuracyDF.csv")) %>% dplyr::filter(subidx == 12)
accu.real.reg <- read_tsv(paste0("RNASeq/accuracies_reg/100_10032.tsv"))
accu.sim.reg <- read_tsv(paste0("simulation/accuracies_reg/100_100_33.tsv"))
accu.sim.ds <- read_csv('simulation/accuracyDF.csv')
lapply(list(accu.sim.reg, accu.sim.ds, accu.real.reg, 
            accu.real.ds.dgm5, accu.real.ds.dgm13), get_optimal_idx)
