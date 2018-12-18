library(tidyverse)
library(cowplot)

feat.imp <- read.csv("featureImp5.csv")[,-1]
feat.imp <- feat.imp[order(feat.imp$score, decreasing = T),]
write.csv(feat.imp, file = "featureImportance5.csv")

feat.imp$feat <- factor(feat.imp$feat, levels = feat.imp$feat )
p <- feat.imp[1:20,] %>% 
  mutate(mod = "DGM-5") %>%
  ggplot(aes(x=score, y=fct_reorder(feat, score, fun=identity))) + 
  geom_point(color = "grey40", stroke = 0) + theme_bw() +
  facet_grid(. ~ mod) +
  labs(y = NULL, x = NULL) + 
  scale_x_continuous(limits = c(0.0075, 0.05)) + 
  theme(strip.background = element_rect(fill="#fcfce6"))

p# ggsave(p, filename = "importanceFeatures5.svg", height = 4, width = 7)

feat.imp <- read.csv("featureImp13.csv")[,-1]
feat.imp <- feat.imp[order(feat.imp$score, decreasing = T),]
write.csv(feat.imp, file = "featureImportance13.csv")

feat.imp$feat <- factor(feat.imp$feat, levels = feat.imp$feat )
q <- feat.imp[1:20,] %>% 
  mutate(mod = "DGM-13") %>%
  ggplot(aes(x=score, y=fct_reorder(feat, score, fun=identity))) + 
  facet_grid(. ~ mod) +
  geom_point(color = "grey40", stroke = 0) + theme_bw() +
  scale_x_continuous(limits = c(0.02, 0.05)) +
  labs(x = NULL, y = NULL) +
  theme(strip.background = element_rect(fill="#fcfce6"))
# ggsave(q, filename = "importanceFeatures12.svg", height = 4, width = 7)
q
pq <- plot_grid(p, q, labels="auto")
pq
ggsave(pq, filename = "importanceFeatures.svg", height = 4, width = 7)
