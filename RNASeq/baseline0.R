library(tidyverse)
library(xgboost)
library(caret)
library(parallel)
library(doSNOW)


Xtrain <- as.matrix(read_csv("RNASeqMDD/Xtrain.csv"))
Xtest <- as.matrix(read_csv("RNASeqMDD/Xtest.csv"))
ytrain <- as.matrix(read_csv("RNASeqMDD/ytrain.csv", col_names = F))
ytest <- as.matrix(read_csv("RNASeqMDD/ytest.csv", col_names = F))
length(ytrain)
# 
# train.control <- trainControl(method = "adaptive_cv", 
#                               number = 10,
#                               repeats = 5,
#                               search = "grid")
# 
# # set parameter values to check
# tune.grid <- expand.grid(eta = 1,
#                          nrounds = c(20, 30, 40),
#                          max_depth = 2:4,
#                          min_child_weight = 1:3,
#                          colsample_bytree = 1,
#                          gamma = 0, 
#                          subsample = c(0.2, 0.4, 0.6))
# 
# # create parallel processes for modeling
# cl <- makeCluster(4, type = "SOCK")
# # register cluster
# t0 <- proc.time()
# set.seed(161308)
# registerDoSNOW(cl)
# # train model with training data
# caret.cv <- train(Xtrain, as.factor(ytrain),
#                   method = "xgbTree",
#                   metric = "Accuracy",
#                   tuneGrid = tune.grid,
#                   trControl = train.control)
# 
# stopCluster(cl)
# t1 <- proc.time()
# delt <- t1 - t0
# caret.cv
# 
# y.pred.bin <- predict(caret.cv, newdata = Xtest)
# (accuracy <- sum(y.pred.bin == ytest)/length(ytest))
# 


# train.control <- trainControl(method = "adaptive_cv", 
#                               number = 10,
#                               repeats = 5,
#                               search = "random")
# 
# # set parameter values to check
# tune.grid <- expand.grid(eta = c(0.8, 0.9, 1),
#                          gamma = 0, 
#                          nrounds = c(20, 40),
#                          max_depth = 2:5,
#                          min_child_weight = 1:2,
#                          colsample_bytree = 1,
#                          subsample = c(0.5, 0.6, 0.7))
# 
# # create parallel processes for modeling
# cl <- makeCluster(4, type = "SOCK")
# # register cluster
# t0 <- proc.time()
# 
# registerDoSNOW(cl)
# set.seed(161308)
# # train model with training data
# caret.cv <- train(Xtrain, as.factor(ytrain),
#                   method = "xgbTree",
#                   metric = "Accuracy",
#                   tuneGrid = tune.grid,
#                   trControl = train.control)
# 
# stopCluster(cl)
# t1 <- proc.time()
# delt <- t1 - t0
# caret.cv$results
# str(caret.cv)
# str(caret.cv$finalModel)
# y.pred.bin <- predict(caret.cv, newdata = Xtest)
# (accuracy <- sum(y.pred.bin == ytest)/length(ytest))


cl <- makeCluster(detectCores())
doParallel::registerDoParallel(cl)

set.seed(1613)
seeds <- vector(mode = "list", length = 51)
#(72 is the number of tuning parameters, 51 = number*repeats)
for(i in 1:50) seeds[[i]]<- sample.int(n=1000, 72)

#for the last model
seeds[[51]]<-sample.int(1000, 1)
train.control <- trainControl(method = "repeatedcv", 
                              number = 10,
                              repeats = 5,
                              search = "random",
                              seeds = seeds)

# set parameter values to check
tune.grid <- expand.grid(eta = c(0.8, 0.9, 1),
                         gamma = 0, 
                         nrounds = c(20, 40),
                         max_depth = 2:5,
                         min_child_weight = 1:2,
                         colsample_bytree = 1,
                         subsample = c(0.5, 0.6, 0.7))

# create parallel processes for modeling
# cl <- makeCluster(4, type = "SOCK")
# register cluster
t0 <- proc.time()


# train model with training data
# registerDoSNOW(cl)
caret.cv <- train(Xtrain, as.factor(ytrain),
                  method = "xgbTree",
                  metric = "Accuracy",
                  tuneGrid = tune.grid,
                  trControl = train.control)

stopCluster(cl)
t1 <- proc.time()
delt <- t1 - t0
# caret.cv$results
# str(caret.cv)
str(caret.cv$finalModel)
y.pred.bin <- predict(caret.cv, newdata = Xtest)
(accuracy <- sum(y.pred.bin == ytest)/length(ytest))


# set.seed(161308)
# my.rf <- ranger::ranger(class ~ ., data = dat.train)
# str(my.rf)
# y.pred <- predict(my.rf, data = Xtest)$predictions
# (accuracy <- sum(y.pred == ytest)/length(ytest))
caret.cv$finalModel$tuneValue

caret.cv$results[caret.cv$results$eta == 0.8 &
                   caret.cv$results$nrounds == 20 &
                   caret.cv$results$subsample == 0.6 &
                   caret.cv$results$max_depth == 2,]
max(caret.cv$resample$Accuracy)

