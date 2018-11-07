rm(list = ls())

library(reshape2)
library(tidyverse)

accu <- read_tsv("accuracies_ds/100_100_23.tsv")
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
  geom_jitter(aes(x = subidx, y = (`Testing Accuracy`-0.4)*10/0.6, color = selectedSubsetID)) + 
  scale_y_continuous(sec.axis = sec_axis(~.*0.6/10+0.4, name = "Average Testing Accuracy")) +
  theme_bw() +   
  scale_x_continuous(breaks = seq(1, 23, 3), limits = c(1, 23), minor_breaks = 1:23) + 
  labs(x = "Subset ID", y = "Count") +
  guides(fill = FALSE) + guides(colour=FALSE)

p
