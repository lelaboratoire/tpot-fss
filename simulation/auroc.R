library(tidyverse)
load("bestSimResult.Rdata")
y.pred <- predict(caret.cv, newdata = Xtest, type = "prob")[, 2]
readr::write_csv(data.frame(y.pred, ytest), "XGBoostPredSim.csv")
predXGBoostSim <- readr::read_csv("XGBoostPredSim.csv")
predTpotDsSim <- readr::read_csv("predictionsBest.csv")

length(y.pred)
myrocX = pROC::roc(predXGBoostSim$ytest, predXGBoostSim$y.pred)
myrocT = pROC::roc(predTpotDsSim$y, predTpotDsSim$ypred)
myrocT$auc

df = data.frame(Specificity=c(myrocX$specificities, myrocT$specificities), 
                Sensitivity=c(myrocX$sensitivities, myrocT$sensitivities),
                Type = c(rep("XGBoost", length(y.pred) + 1), 
                         rep("TPOT-DS", length(myrocT$sensitivities))))

ggplot(data = df, aes(x = Specificity, y = Sensitivity, color = Type))+
  geom_abline(intercept = 1, slope = 1, color='grey', linetype = 2)+
  geom_path()+
  scale_x_reverse(labels = scales::percent, limit = c(1, 0)) +
  scale_y_continuous(labels = scales::percent) +
  annotate("text", x = 0.30, y = 0.20, label = paste0('AUC: ', round(myrocX$auc*100,1), '%'), size = 5)+
  ylab('Sensitivity')+
  xlab('Specificity')
