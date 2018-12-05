library(tidyverse)
library(xgboost)
library(caret)
library(parallel)
library(ranger)
rm(list = ls())

# load('data_inte_1618.Rdata')

# dat.train <- data.sets$train
# dat.test <- data.sets$holdout
# Xtrain <- select(dat.train, - class)
# ytrain <- dat.train$class
# Xtest <- select(dat.test, - class)
# ytest <- dat.test$class
# length(ytrain)
Xtrain <- as.matrix(read_csv("simdat/Xtrain.csv"))
Xtest <- as.matrix(read_csv("simdat/Xtest.csv"))
ytrain <- as.matrix(read_csv("simdat/ytrain.csv", col_names = F))
ytest <- as.matrix(read_csv("simdat/ytest.csv", col_names = F))

cl <- makeCluster(detectCores())
doParallel::registerDoParallel(cl)

set.seed(1618)
# seeds <- vector(mode = "list", length = 51)
# (72 is the number of tuning parameters, 51 = number*repeats + 1)
# for(i in 1:50) seeds[[i]]<- sample.int(n=1000, 72)

#for the last model
# seeds[[51]]<-sample.int(1000, 1)


train.control <- trainControl(method = "repeatedcv", 
                              number = 10,
                              repeats = 5,
                              search = "random"
                              # , seeds = seeds
                              )

# set parameter values to check
tune.grid <- expand.grid(eta = c(0.7, 0.8, 0.9, 1),
                         gamma = c(0, 0.1, 0.2), 
                         nrounds = c(20, 40),
                         max_depth = 2:5,
                         min_child_weight = 1:5,
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

max(caret.cv$resample$Accuracy)
caret.cv$results[which.max(caret.cv$results$Accuracy),]
# caret.cv$finalModel$tuneValue
caret.cv$results[caret.cv$results$eta == 1 &
                   caret.cv$results$nrounds == 20 &
                   caret.cv$results$subsample == 0.9 &
                   caret.cv$results$max_depth == 4 &
                   caret.cv$results$gamma == 0 &
                   caret.cv$results$min_child_weight == 1,]

# set.seed(1618)
# my.rf <- ranger::ranger(class ~ ., data = dat.train)
# str(my.rf)
# y.pred <- predict(my.rf, data = Xtest)$predictions
# (accuracy <- sum(y.pred == ytest)/length(ytest))

save.image(file = 'bestSimResult.Rdata')
