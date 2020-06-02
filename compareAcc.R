library(tidyverse)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

best.sim <- read_csv('simulation/bestAccuracies.csv') %>% 
  reshape2::melt(id = 0) %>%
  mutate(dat = 'Simulated data')
best.rna <- read_csv('RNASeq/bestAccuracies.csv') %>% 
  reshape2::melt(id = 0) %>%
  mutate(dat = 'Real-world data')

best.acc <- rbind(best.sim, best.rna)
best.acc$dat <- factor(best.acc$dat, levels = unique(best.acc$dat))

levels(best.acc$variable) <- 
  dplyr::recode(levels(best.acc$variable), "TPOT-DS" = "TPOT-FSS")

pq <- best.acc %>% 
  ggplot(aes(y = value, x= variable, color = variable)) +
  facet_wrap(~ dat) +
  theme_bw() + 
  # theme(strip.background = element_rect(fill="#fcfce6")) +
  scale_color_manual(values = cbPalette[c(8,6,3)])+
  guides(color = F) + labs(x = NULL, y = 'Accuracy')


ggsave(pq, filename = 'compareAcc.svg', height = 3, width = 5)
ggsave(pq, filename = 'compareAcc.pdf', height = 3, width = 5)
ggsave(
  pq + geom_boxplot() +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank()), 
  filename = 'fig2.pdf', height = 2.3, width = 5)

library(ggdark)
pq_dark <- pq + 
  geom_boxplot(fill = '#111111') +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  dark_theme_gray() + 
  theme(
        plot.background = element_rect(fill = "#111111"),
        panel.background = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major = element_line(color = "grey30", size = 0.2),
        panel.grid.minor = element_line(color = "grey30", size = 0.2),
        legend.background = element_blank(),
        axis.ticks = element_blank(),
        legend.key = element_blank(),
        legend.position = c(0.815, 0.27))
ggsave(pq_dark, filename = 'dark_compare.svg', height = 3, width = 5)

simX <- best.sim %>% filter(variable == 'XGBoost') %>% pull(value)
simT <- best.sim %>% filter(variable == 'TPOT') %>% pull(value)
simDS <- best.sim %>% filter(variable == 'TPOT-FSS') %>% pull(value)
t.test(x = simX, y = simDS, 'less')
t.test(x = simT, y = simDS, 'less')
t.test(x = simT, y = simX, 'less')

rnaX <- best.rna %>% filter(variable == 'XGBoost') %>% pull(value)
rnaT <- best.rna %>% filter(variable == 'TPOT') %>% pull(value)
rnaDS <- best.rna %>% filter(variable == 'TPOT-FSS') %>% pull(value)
t.test(x = as.matrix(rnaX), y = as.matrix(rnaDS), 'less')
t.test(x = rnaT, y = rnaDS, 'less')
t.test(x = rnaX, y = rnaT, 'less')
