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
files.short <- gsub("pipelines_ds/|.py|score_", "", filenames)
selected.sub <- data.frame(matrix(NA, nrow = length(filenames), ncol = 1), 
                           row.names = files.short)
colnames(selected.sub) <- "selectedSubsetID"

for (file in filenames){
  file.short <- gsub("pipelines_ds/|.py|score_", "", file)
  pipe <- read.delim(file, stringsAsFactors = F)
  pipe.idx <- grep("    DatasetSelector", pipe[,1])
  select.i <- gsub("\\, subset_list=subsets.csv)\\,|    DatasetSelector\\(sel_subset=", "", pipe[pipe.idx, 1])
  selected.sub[file.short, 1] <- select.i
}
selected.sub$dataidx <- as.numeric(gsub("simulatedGenex", "", rownames(selected.sub)))


accu.subset <- merge(selected.sub, accu, by = "dataidx") %>%
  mutate(subidx = as.numeric(selectedSubsetID)) %>%
  mutate(subname = factor(as.factor(paste0("S[", subidx+1, "]")))) %>%
  arrange(desc(`Testing Accuracy`))

accu.subset.sum <- 
  accu.subset %>% 
  group_by(subidx) %>% 
  summarise(avg.test = mean(`Testing Accuracy`), avg.train.CV = mean(`Training CV Accuracy`))

    
write_csv(accu.subset, "accuracyDF.csv")
accu.subset$subname <- factor(accu.subset$subname, levels = paste0("S[", sort(unique(accu.subset$subidx))+1, "]"))

accu.sub.melt <- reshape2::melt(
  accu.subset[, c("Training CV Accuracy", "Testing Accuracy", "subname", "dataidx")],
  id = c("subname", "dataidx"))

# hacky stuff to interchange colors
accu.subset$box <- accu.subset$subname %in% c("S[1]", "S[5]", "S[15]")
accu.subset$col <- accu.subset$subname %in% c("S[1]", "S[3]", "S[5]", "S[8]", "S[12]", "S[16]", "S[18]")
q <- ggplot(accu.subset, aes(x = subname, y = `Testing Accuracy`, color = col)) +
  stat_summary(fun.data = function(x) c(y = 0.9, label = length(x)),
               geom = "text", fun.y = NULL,
               position = position_dodge(width = 0.75)) +
  geom_boxplot(data = accu.subset[accu.subset$box == TRUE, ],  
               outlier.size = NULL,
               aes(x = subname, y = `Testing Accuracy`), color = "grey70") +
  ggbeeswarm::geom_beeswarm(priority = "random", cex = 1.3, size = 1.6, alpha = 0.8, stroke = 0) +
  theme_bw() +
  annotate("text", x = 10.2, y = 0.35, size = 2.5, fontface = 'italic',
           label = "* Boxplots are drawn for subsets with more than three data points") +
  scale_color_manual(values = c("#d55e00", "#549735")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 2), name = "Holdout accuracy") +
  expand_limits(x = -0.85) +
  labs(x = NULL) +
  scale_x_discrete(labels = parse(text = levels(accu.subset$subname))) +
  guides(fill = FALSE) + guides(colour=FALSE)

q
# ggsave(q, filename = paste0("sim_", n.iters, ".svg"), width = 5.8, height = 2.8, units = "in")
ggsave(q, filename = paste0("sim_", n.iters, ".pdf"), width = 5.7, height = 2.7, units = "in") 
