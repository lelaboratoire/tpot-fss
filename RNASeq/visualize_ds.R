rm(list = ls())

library(reshape2)
library(tidyverse)
library(ggthemes)
library(scales)
library(ggplot2)
library(ggforce)
# library(viridis)
# library(colorblindr)

n.iters <- 44
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

accu <- read_tsv(paste0("accuracies_ds/100_100_", n.iters - 1, ".tsv"))
colnames(accu)[1] <- "dataidx"
accu.melt <- melt(accu, id = "dataidx")

ggplot(accu.melt, aes(y = value, x = variable, color = variable)) + 
  geom_point() + geom_line(aes(group = dataidx), color = "grey") + 
  theme_bw() + labs(y = "Accuracy", x = "") +
  theme(legend.position = "None", legend.title = element_blank())

filenames <- list.files("pipelines_ds", pattern="*.py", full.names=TRUE)
files.short <- gsub("pipelines_ds/RNASeq_|.py|score_", "", filenames)
selected.sub <- data.frame(matrix(NA, nrow = length(filenames), ncol = 1), 
                           row.names = files.short)
colnames(selected.sub) <- "selectedSubsetID"

for (file in filenames){
  file.short <- gsub("pipelines_ds/RNASeq_|.py|score_", "", file)
  pipe <- read.delim(file, stringsAsFactors = F)
  pipe.idx <- grep("    DatasetSelector", pipe[,1])
  select.i <- gsub("\\, subset_list=module23.csv)\\,|    DatasetSelector\\(sel_subset=", "", pipe[pipe.idx, 1])
  selected.sub[file.short, 1] <- select.i
}
selected.sub$dataidx <- as.numeric(gsub("MDD", "", rownames(selected.sub)))

accu.subset <- merge(selected.sub, accu, by = "dataidx")
accu.subset$subidx <- as.numeric(accu.subset$selectedSubsetID)
accu.subset.sum <- 
  accu.subset %>% 
  group_by(subidx) %>% 
  summarise(avg.test = mean(`Testing Accuracy`))

p <- ggplot(accu.subset, aes(group = subidx)) + 
  geom_histogram(aes(x = subidx, fill = selectedSubsetID), alpha = 0.4, binwidth = 1) +
  geom_jitter(aes(x = subidx, y = (`Testing Accuracy`-0.4)*10/0.2, color = selectedSubsetID)) + 
  scale_y_continuous(sec.axis = sec_axis(~.*0.2/10+0.4, name = "Testing Accuracy")) +
  theme_bw() +   
  scale_x_continuous(breaks = seq(1, 23, 3), limits = c(1, 23), minor_breaks = 1:23) + 
  labs(x = "Subset ID", y = "Pipeline choice frequency") +
  guides(fill = FALSE) + guides(colour=FALSE)

p

accu.subset$subname <- as.factor(paste0("s", accu.subset$subidx))
accu.subset$subname <- factor(accu.subset$subname, levels = unique(accu.subset$subname))
q <- ggplot(accu.subset, aes(x = subname, y = `Testing Accuracy`)) + 
  geom_boxplot(color = "grey80") +
  # geom_violin(aes(x = subidx, y = `Testing Accuracy`, fill = selectedSubsetID), 
  #             alpha = 0.6, scale = "count", trim = T) +
  geom_jitter(size = 2, width = 0.1) + 
  theme_bw() + 
  # viridis::scale_fill_viridis(discrete = TRUE, option="D", reverse = TRUE) +
  # viridis::scale_color_viridis(discrete = TRUE, option="D", reverse = TRUE) +
  # scale_colour_tableau("Classic Color Blind") +
  # scale_fill_tableau("Classic Color Blind") +
  # scale_color_colorblind() +
  # scale_fill_colorblind() +
  # scale_fill_viridis_d(direction = -1, end = 0.5) +
  # scale_color_viridis_d(direction = -1, end = 0.5) +
  # scale_x_continuous(breaks = seq(1, 23, 3), limits = c(1, 23), minor_breaks = 1:23) + 
  labs(x = "Subset ID", y = "Testing Accuracy") +
  guides(fill = FALSE) + guides(colour=FALSE)

q

ggsave(q, filename = paste0("real_", n.iters, ".svg"), width = 5, height = 4, units = "in")






q <- ggplot(accu.subset, aes(x = subname, y = `Testing Accuracy`)) + 
  geom_boxplot(color = "grey70") +
  # geom_violin(aes(x = subidx, y = `Testing Accuracy`, fill = selectedSubsetID), 
  #             alpha = 0.6, scale = "count", trim = T) +
  # geom_jitter(size = 2, width = 0.1, height = 0) +
  # ggforce::geom_sina(size = 2) +
  ggbeeswarm::geom_beeswarm(cex = 2, size = 2, alpha = 0.7) +
  theme_bw() + 
  # viridis::scale_fill_viridis(discrete = TRUE, option="D", reverse = TRUE) +
  # viridis::scale_color_viridis(discrete = TRUE, option="D", reverse = TRUE) +
  # scale_colour_tableau("Classic Color Blind") +
  # scale_fill_tableau("Classic Color Blind") +
  # scale_color_colorblind() +
  # scale_fill_colorblind() +
  # scale_fill_viridis_d(direction = -1, end = 0.5) +
  # scale_color_viridis_d(direction = -1, end = 0.5) +
  # scale_x_continuous(breaks = seq(1, 23, 3), limits = c(1, 23), minor_breaks = 1:23) + 
  labs(x = "Subset ID", y = "Testing Accuracy") +
  guides(fill = FALSE) + guides(colour=FALSE)

q


