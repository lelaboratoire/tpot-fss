cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

bestAccu <- read_csv('bestAccuracies.csv')
p <- bestAccu %>% reshape2::melt(id = 0) %>%
  ggplot(aes(y = value, x= variable, color = variable)) +
  theme_bw() + geom_boxplot() +
  scale_color_manual(values = cbPalette[2:5])+
  guides(color = F) + labs(x = NULL, y = 'Accuracy')
p
ggsave(p, filename = 'compareAcc.svg')