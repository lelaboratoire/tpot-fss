library(tidyverse)
library(cowplot)

feat.imp <- read.csv("featureImp5.csv")[,-1]
feat.imp <- feat.imp[order(feat.imp$score, decreasing = T),]
write.csv(feat.imp, file = "featureImportance5.csv")

feat.imp$feat <- factor(feat.imp$feat, levels = feat.imp$feat )
p <- ggplot(feat.imp[1:20,], aes(x=score, y=fct_reorder(feat, score, fun=identity))) + 
  geom_point(color = "grey40", stroke = 0) + theme_bw() +
  labs(x = "DGM-5", y = "") + 
  scale_x_continuous(limits = c(0.007, 0.04))
# ggsave(p, filename = "importanceFeatures5.svg", height = 4, width = 7)

feat.imp <- read.csv("featureImp13.csv")[,-1]
feat.imp <- feat.imp[order(feat.imp$score, decreasing = T),]
write.csv(feat.imp, file = "featureImportance13.csv")

feat.imp$feat <- factor(feat.imp$feat, levels = feat.imp$feat )
q <- ggplot(feat.imp[1:20,], aes(x=score, y=fct_reorder(feat, score, fun=identity))) + 
  geom_point(color = "grey40", stroke = 0) + theme_bw() +
  scale_x_continuous(limits = c(0.007, 0.04)) +
  labs(x = "DGM-13", y = "")  
# ggsave(q, filename = "importanceFeatures12.svg", height = 4, width = 7)

pq <- plot_grid(p, q, labels="AUTO")
ggsave(pq, filename = "importanceFeatures.svg", height = 4, width = 7)
