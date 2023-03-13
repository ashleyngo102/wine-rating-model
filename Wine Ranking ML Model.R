library(rpart)
library(rpart.plot)
library(randomForest)
library(class)
library(e1071)
library(stats)
library(FNN)
library(forecast)


#First we load the full datasets (white and red wine sets) into R
fullWhiteSet = read.csv("whiteSet.csv", header=TRUE)
fullRedSet = read.csv("redSet.csv", header=TRUE)

#Now we evaluate the contents of the dataset to see how best to clean the data
summary(fullWhiteSet)
summary(fullRedSet)

#To remove statistically insignificant features, we run linear regressions on each data set and remove features with P values greater than 0.05
regModelWhite = lm(quality ~., data=fullWhiteSet)
regModelRed = lm(quality~., data=fullRedSet)
summary(regModelWhite)
summary(regModelRed)


regWhiteSet = subset(fullWhiteSet, select = c(-citric.acid, -chlorides, -total.sulfur.dioxide))
regRedSet = subset(fullRedSet, select = c(-fixed.acidity, -citric.acid, -residual.sugar, -density))

#The above feature selection process assumes that the relationship between quality and all other features is linear, which is not necessarily the case. As such, we will also run our models on a decision tree which makes no such assumption to perform feature selection
#First we build our tree models off of the full data set (as we are not predicting quite yet) and prune to prevent overfitting
treeModelWhite = rpart(quality~., data=fullWhiteSet)
summary(treeModelWhite)
treeModelWhite = prune(treeModelWhite, cp=0.018)

treeModelRed = rpart(quality~., data=fullRedSet)
summary(treeModelRed)
treeModelRed = prune(treeModelRed, cp=0.015)

#Then we plot each of our decision trees and any features that do not show up can be deemed statistically insignificant, so we discard them from our full datasets and store the remaining subset into new variables
plot(treeModelWhite, margin=0.1)
text(treeModelWhite)
treeWhiteSet = fullWhiteSet[,c("alcohol", "volatile.acidity", "free.sulfur.dioxide", "quality")]

plot(treeModelRed, margin=0.1)
text(treeModelRed)
treeRedSet = fullRedSet[,c("alcohol", "volatile.acidity", "sulphates", "quality")]

#Now that we have full datasets for both red and white wines, as well as datasets with features removed guided by linear regression and decision tree models, we can run classification models on all 3 types of datasets to see which assortment of features provides the highest prediction accuracy.  