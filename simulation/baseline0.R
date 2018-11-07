library(tidyverse)
library(xgboost)
library(caret)
library(parallel)
library(doSNOW)
library(ranger)
rm(list = ls())
load('data_inte_1618.Rdata')

dat.train <- data.sets$train
dat.test <- data.sets$holdout
Xtrain <- select(dat.train, - class)
ytrain <- dat.train$class
Xtest <- select(dat.test, - class)
ytest <- dat.test$class
# length(ytrain)

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

max(caret.cv$resample$Accuracy)
caret.cv$results[which.max(caret.cv$results$Accuracy),]
caret.cv$finalModel$tuneValue
caret.cv$results[caret.cv$results$eta == 0.8 & 
                   caret.cv$results$subsample == 0.7 &
                   caret.cv$results$max_depth == 3,]
