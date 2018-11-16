rm(list = ls())

check.packages <- function(pkg){
  # check.packages function: install and load multiple R packages.
  # Check to see if packages are installed. Install them if they are not, 
  # then load them into the R session.
  # https://gist.github.com/smithdanielle/9913897
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, repos = "http://cran.us.r-project.org", dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}
packages <- c("tidyverse", "reshape2", "ggbeeswarm", "viridis")
check.packages(packages) 

library(tidyverse)

n.iters <- 100
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", 
                "#CC79A7", "#c5679b", "#be548f")
accu <- read_tsv(paste0("accuracies_ds/100_100_", n.iters - 1, ".tsv"))
colnames(accu)[1] <- "dataidx"
accu.melt <- reshape2::melt(accu, id = "dataidx")

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
  summarise(avg.test = mean(`Testing Accuracy`), avg.train.CV = mean(`Training CV Accuracy`))
accu.subset$subname <- as.factor(paste0("DGM-", accu.subset$subidx+1))
accu.subset$subname <- factor(accu.subset$subname, levels = paste0("DGM-", sort(unique(accu.subset$subidx))+1))


q <- ggplot(accu.subset, aes(x = subname, y = `Testing Accuracy`, color = subname)) + 
  geom_boxplot(color = "grey40") +
  stat_summary(fun.data = function(x) c(y = 0.77, label = round(length(x)/n.iters, 2)), 
               geom = "text", fun.y = NULL, 
               position = position_dodge(width = 0.75)) +
  ggbeeswarm::geom_beeswarm(priority = "random", cex = 1.8, size = 1, alpha = 0.8) +
  theme_bw() + 
  viridis::scale_color_viridis(discrete = T) +
  labs(x = "Subset ID", y = "Testing Accuracy") +
  guides(fill = FALSE) + guides(colour=FALSE)

q

ggsave(q, filename = paste0("real_", n.iters, ".svg"), width = 5, height = 4, units = "in")

accu.sub.melt <- reshape2::melt(
  accu.subset[, c("Training CV Accuracy", "Testing Accuracy", "subname", "dataidx")],
  id = c("subname", "dataidx"))


ggplot(accu.sub.melt, aes(y = value, x = variable, group = subname, color = subname)) + 
  geom_point() + geom_line(aes(group = dataidx)) + 
  viridis::scale_color_viridis(discrete = T) +
  labs(color = "Subset") +
  theme_bw() + labs(y = "Accuracy", x = "") +
  theme(legend.position = c(0.15,0.28))



accu.subset$box <- accu.subset$subname %in% c("DGM-5", "DGM-13")
accu.subset$col <- accu.subset$subname %in% c("DGM-3", "DGM-5", "DGM-17")
q <- ggplot(accu.subset, aes(x = subname, y = `Testing Accuracy`, color = col)) + 
  stat_summary(fun.data = function(x) c(y = 0.77, label = length(x)), 
               geom = "text", fun.y = NULL, 
               position = position_dodge(width = 0.75)) +
  geom_boxplot(data = accu.subset[accu.subset$box == TRUE, ],  
               aes(x = subname, y = `Testing Accuracy`), color = "grey70") +
  ggbeeswarm::geom_beeswarm(priority = "random", cex = 1.6, size = 1.5, alpha = 0.8, stroke = 0) +
  theme_bw() + 
  annotate("text", x = 4.3, y = 0.45, size = 2.7, fontface = 'italic',
           label = "* Boxplots are drawn for subsets with more than three data points") +
  # viridis::scale_color_viridis(discrete = T, option = "E") +
  scale_color_manual(values = c(cbbPalette[6], cbbPalette[10])) +
  scale_y_continuous(labels = scales::percent, name = "Holdout accuracy") +
  labs(x = NULL) +
  guides(fill = FALSE) + guides(colour=FALSE)
q
ggsave(q, filename = paste0("real_", n.iters, ".svg"), width = 5, height = 3.5, units = "in")
    
