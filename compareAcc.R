cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

best.sim <- read_csv('simulation/bestAccuracies.csv') %>% 
  reshape2::melt(id = 0) %>%
  mutate(dat = 'Simulated data')
best.rna <- read_csv('RNASeq/bestAccuracies.csv') %>% 
  reshape2::melt(id = 0) %>%
  mutate(dat = 'Real-world data')

best.acc <- rbind(best.sim, best.rna)
best.acc$dat <- factor(best.acc$dat, levels = unique(best.acc$dat))
  
pq <- best.acc %>% 
  ggplot(aes(y = value, x= variable, color = variable)) +
  facet_wrap(~ dat) +
  theme_bw() + geom_boxplot() +
  theme(strip.background = element_rect(fill="#fcfce6")) +
  scale_color_manual(values = cbPalette[c(8,6,3)])+
  guides(color = F) + labs(x = NULL, y = 'Accuracy')


pq
ggsave(pq, filename = 'compareAcc.svg', height = 3, width = 5)
 