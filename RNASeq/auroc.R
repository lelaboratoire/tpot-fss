library(tidyverse)
library(plotROC)
library(pROC)
# install.packages("directlabels", repo="http://r-forge.r-project.org")
# library(directlabels)
# install.packages("devtools")
# library(devtools)
# install_github("wgaul/wgutil")
library("wgutil")
data(cbPalette)
pred5 <- read_csv("predictions5.csv")
predStandard <- read_csv("predictionsTPOTStand.csv")

load("bestRealResult.Rdata")

y.pred <- predict(caret.cv, newdata = Xtest, type = "prob")[, 2]
write_csv(data.frame(ypred = y.pred, y = as.numeric(ytest)), "XGBoostPredSim.csv")
predXGBoostSim <- read_csv("XGBoostPredSim.csv")
# predTpotDsSim <- read_csv("predictionsTpot.csv")


myroc = pROC::roc(pred5$y, pred5$ypred)
plot.roc(myroc, print.auc = TRUE, col='red')

myrocX = pROC::roc(predXGBoostSim$y, predXGBoostSim$ypred)
myrocT5 = pROC::roc(pred5$y, pred5$ypred)
myrocTStandard = pROC::roc(predStandard$y, predStandard$ypred)

myrocX$auc

df = data.frame(Specificity=c(myrocX$specificities, myrocT5$specificities, myrocTStandard$specificities), 
                Sensitivity=c(myrocX$sensitivities, myrocT5$sensitivities, myrocTStandard$sensitivities),
                Type = c(rep("XGBoost", length(myrocX$specificities)), 
                         rep("TPOT-DS", length(myrocT5$sensitivities)),
                         rep("TPOT", length(myrocTStandard$sensitivities))))

# df = data.frame(Specificity=myroc$specificities, Sensitivity=myroc$sensitivities)
# ggplot(data = df, aes(x = Specificity, y = Sensitivity))+
#   geom_step(color='red', size=2, direction = "hv")+
#   scale_x_reverse()+
#   geom_abline(intercept = 100, slope = 1, color='grey')+
#   # annotate("text", x = 30, y = 20, label = paste0('AUC: ', round(roc$auc,1), '%'), size = 8)+
#   ylab('Sensitivity (%)')+
#   xlab('Specificity (%)')


auc.plot <- ggplot(data = df, aes(x = Specificity, y = Sensitivity, color = Type))+
  geom_abline(intercept = 1, slope = 1, color='grey', linetype = 2)+
  geom_path(size = 1)+
  # geom_dl() +
  scale_colour_manual(values=cbPalette) +
  scale_x_reverse(labels = scales::percent, limit = c(1, 0), name = 'Specificity') +
  scale_y_continuous(labels = scales::percent, name = 'Sensitivity') +
  annotate("text", x = 0.5, y = 0.25, label = paste0('TPOT AUC: ', round(myrocTStandard$auc, 3)), color = cbPalette[1])+
  annotate("text", x = 0.63, y = 0.9, label = paste0('TPOT-DS AUC: ', round(myrocT5$auc, 3)), color = cbPalette[2])+
  annotate("text", x = 0.3, y = 0.6, label = paste0('XGBoost AUC: ', round(myrocX$auc, 3)), color = cbPalette[3]) +
  guides(color=FALSE)
auc.plot
ggsave(auc.plot, filename = "aucRNASeq.svg", height = 5, width = 5)
# auc.plot + geom_dl(method="first.qp")
# direct.label(auc.plot)
# install.packages("ggproto")
# library(ggplot2)
# giris <- ggplot(iris,aes(Petal.Length,Sepal.Length))+
#   geom_point(aes(shape=Species))
# giris.labeled <- giris+
#   geom_dl(aes(label=Species),method="smart.grid")+
#   scale_shape_manual(values=c(setosa=1,virginica=6,versicolor=3),
#                      guide="none")

