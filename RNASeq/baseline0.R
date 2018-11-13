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


train.control <- trainControl(method = "adaptive_cv", 
                              number = 10,
                              repeats = 5,
                              search = "random")

# set parameter values to check
tune.grid <- expand.grid(eta = c(0.8, 0.9, 1),
                         gamma = 0, 
                         nrounds = c(20, 40),
                         max_depth = 2:5,
                         min_child_weight = 1:2,
                         colsample_bytree = 1,
                         subsample = c(0.5, 0.6, 0.7))

# create parallel processes for modeling
cl <- makeCluster(4, type = "SOCK")
# register cluster
t0 <- proc.time()

registerDoSNOW(cl)
set.seed(161308)
# train model with training data
caret.cv <- train(Xtrain, as.factor(ytrain),
                  method = "xgbTree",
                  metric = "Accuracy",
                  tuneGrid = tune.grid,
                  trControl = train.control)

stopCluster(cl)
t1 <- proc.time()
delt <- t1 - t0
caret.cv$results
str(caret.cv)
str(caret.cv$finalModel)
y.pred.bin <- predict(caret.cv, newdata = Xtest)
(accuracy <- sum(y.pred.bin == ytest)/length(ytest))


# set.seed(161308)
# my.rf <- ranger::ranger(class ~ ., data = dat.train)
# str(my.rf)
# y.pred <- predict(my.rf, data = Xtest)$predictions
# (accuracy <- sum(y.pred == ytest)/length(ytest))




