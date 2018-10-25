# ---------------------------------------------------# 
# 
# Simulate gene expression data to test TPOT template
# 
# ---------------------------------------------------# 
# 
# Subset sizes are simulated to match the distribution 
# of 23 module sizes in "Identification and replication of 
# RNA-Seq gene network modules associated with depression 
# severity" (Le et al. 2018). Data available at
#  https://github.com/insilico/DepressionGeneModules

library(privateEC)
library(fitdistrplus)
library(readr)
# setwd("~/tpot-ds/simulation")

set.seed(1618)
n.samples <- 400     # 200 samples in train/holdout/test
n.variables <- 5000   # 100 features
ndraws <- 100

label <- "class"
type <- "interactionErdos" # interaction effect simulatios
bias <- 0.8 # moderate effect size, smaller value is larger effect
pct.signals <- 0.04   # pct functional features
n.signals <- n.variables*pct.signals
verbose <- FALSE
data.sets <- createSimulation(num.samples = n.samples,
                              num.variables = n.variables,
                              pct.signals = pct.signals,
                              bias = bias,
                              pct.train = 1/2,
                              pct.holdout = 1/2,
                              sim.type = type,
                              save.file = NULL,
                              verbose = verbose)
save(data.sets, file = "data_inte_1618.Rdata")
# load("data_inte_1618.Rdata")
write_csv(data.sets$train, "simulatedGenex.csv")

# Fit a gamma distribution to the known 23 module sizes
mod23 <- read.csv("module23.csv")
plot(density(mod23$Subset.size))
fit.gamma <- fitdist(mod23$Subset.size, distr = "gamma", method = "mle")
(paras <- fit.gamma$estimate)
plot(fit.gamma)

# Sampling from the obtained gamma functions the subset sizes, 
# until reaches n.variables = 5000
subset.sizes <- ceiling(rgamma(ndraws, paras[1], paras[2]))
cum.sizes <- cumsum(subset.sizes)
# Patch beginning and end - last subset size to be > 100
cum.sizes <- c(0, cum.sizes[cum.sizes < (n.variables - 100)], n.variables) 
nsubs <- length(cum.sizes) - 1
all.features <- colnames(data.sets$train)
null.feats <- setdiff(all.features, c(data.sets$signal.names, "class"))
permuted.null.feats <- sample(null.feats, size = n.variables - n.signals, replace = F)

probs <- 1.618^(-(1:nsubs))
func.group <- sample(1:nsubs, size = n.signals, prob = probs, replace = T)
func.group.df <- data.frame(sigs = data.sets$signal.names, func.group, stringsAsFactors = F)

mysubset <- data.frame(matrix(NA, ncol = 3, nrow = nsubs))
colnames(mysubset) <- c("Subset", "Subset size", "Features")
feat.list <- list() 
for (i in 1:nsubs){
  null.features <- permuted.null.feats[(cum.sizes[i]+1):cum.sizes[i+1]]
  func.features <- func.group.df[func.group.df$func.group == i, "sigs"]
  features <- c(null.features, func.features)
  feat.list[[i]] <- features
  feat.names <- paste(features, collapse = ";") 
  mysubset[i, 1] <- i
  mysubset[i, 2] <- length(features)
  mysubset[i, 3] <- feat.names
}


write_csv(mysubset, "subsets.csv")
save(feat.list, file = "subsets.Rdata")
func.vars <- unlist(lapply(feat.list, function(feats) length(grep(pattern = "simvar", x = feats))))
signal.df <- data.frame(subset = 1:nsubs, pctSig = func.vars/mysubset[, 2])
signal.df <- signal.df[order(signal.df$pctSig, decreasing = T), ]
write_csv(signal.df, "signalPct.csv")
