library(tidyverse)
library(cowplot)

feat.imp <- read.csv("RNASeq/featureImp5.csv")[,-1]
feat.imp <- feat.imp[order(feat.imp$score, decreasing = T),]
# write.csv(feat.imp, file = "featureImportance5.csv")

feat.imp$feat <- factor(feat.imp$feat, levels = feat.imp$feat )
p <- feat.imp[1:20,] %>% 
  mutate(mod = "DGM-5") %>%
  ggplot(aes(x=score, y=fct_reorder(feat, score, identity))) + 
  geom_point(color = "grey40", stroke = 0) + theme_bw() +
  facet_grid(. ~ mod) +
  labs(y = NULL, x = NULL) + 
  scale_x_continuous(limits = c(0.0075, 0.05)) + 
  theme(strip.background = element_rect(fill="#fcfce6"))

p# ggsave(p, filename = "importanceFeatures5.svg", height = 4, width = 7)

feat.imp <- read.csv("RNASeq/featureImp13.csv")[,-1]
feat.imp <- feat.imp[order(feat.imp$score, decreasing = T),]
# write.csv(feat.imp, file = "featureImportance13.csv")

feat.imp$feat <- factor(feat.imp$feat, levels = feat.imp$feat )
q <- feat.imp[1:20,] %>% 
  mutate(mod = "DGM-13") %>%
  ggplot(aes(x=score, y=fct_reorder(feat, score, identity))) + 
  facet_grid(. ~ mod) +
  geom_point(color = "grey40", stroke = 0) + theme_bw() +
  scale_x_continuous(limits = c(0.02, 0.05)) +
  labs(x = NULL, y = NULL) +
  theme(strip.background = element_rect(fill="#fcfce6")) 
# ggsave(q, filename = "importanceFeatures12.svg", height = 4, width = 7)
q
pq <- plot_grid(p, q, labels=c('(a)', '(b)'))
pq
# ggsave(pq, filename = "importanceFeatures.svg", height = 4, width = 7)
# ggsave(pq, filename = "importanceFeatures.pdf", height = 4, width = 7)


library(ggdark)
pq_dark <- plot_grid(
  p +
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
      legend.position = c(0.815, 0.27)), 
  q +
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
      legend.position = c(0.815, 0.27)))
ggsave(pq_dark, filename = 'dark_imp_feats.svg', height = 4, width = 7)

