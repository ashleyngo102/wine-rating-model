library(rpart)
library(rpart.plot)
library(randomForest)
library(class)
library(e1071)
library(stats)
library(FNN)
library(forecast)
library(neuralnet)
library(ggplot2)
library(corrplot)

AllWhiteRegErrors = c()
AllWhiteTreeErrors = c()
AllWhiteForestErrors = c()
AllWhiteKnnErrors = c()
AllWhiteSvrErrors = c()
AllWhiteNeuralErrors = c()

AllRedRegErrors = c()
AllRedTreeErrors = c()
AllRedForestErrors = c()
AllRedKnnErrors = c()
AllRedSvrErrors = c()
AllRedNeuralErrors = c()


#First we load the full datasets (white and red wine sets) into R
fullWhiteSet = read.csv("whiteSet.csv", header=TRUE)
fullRedSet = read.csv("redSet.csv", header=TRUE)

#Now we evaluate the contents of the dataset to see how best to clean the data
summary(fullWhiteSet)
summary(fullRedSet)

p <- fullRedSet[0,1:11]

#visualize red wine
ggplot(fullRedSet, aes (x = quality, y = density)) + geom_boxplot(outlier.colour = "red", outlier.shape=18,outlier.size=2, notch = TRUE)
ggplot(data = fullRedSet , mapping = aes(x = quality , y = density)) +
  geom_boxplot () 
ggplot(data = fullRedSet) +
  geom_point(mapping = aes(x = quality , y = density)) +  coord_flip()
corrplot(cor(fullRedSet[,1:12]), method = "number")

#visualize white wine 
ggplot(fullWhiteSet, aes (x = quality, y = density)) + geom_boxplot(outlier.colour = "red", outlier.shape=18,outlier.size=2, notch = TRUE)
ggplot(data = fullWhiteSet , mapping = aes(x = quality , y = )) + geom_boxplot () 
ggplot(data = fullWhiteSet) +
  geom_point(mapping = aes(x = quality , y = density)) +  coord_flip()
corrplot(cor(fullWhiteSet[,1:12]), method = "number")

#Visualize the data
ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=alcohol,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=alcohol,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=fixed.acidity,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=fixed.acidity,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=volatile.acidity,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=volatile.acidity,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=citric.acid,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=citric.acid,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=residual.sugar,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=residual.sugar,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=chlorides,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=chlorides,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=free.sulfur.dioxide,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=free.sulfur.dioxide,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=total.sulfur.dioxide,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=total.sulfur.dioxide,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=density,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=density,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=pH,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=pH,y=quality))

ggplot(data = fullWhiteSet) + 
  geom_point(mapping = aes(x=sulphates,y=quality))

ggplot(data = fullRedSet) + 
  geom_point(mapping = aes(x=sulphates,y=quality))

round(cor(fullWhiteSet),2)
round(cor(fullRedSet),2)

#Let us split the datasets into testing and training sets
for(fold in 1:5)
{
  num_samplesWhite = dim(fullWhiteSet)[1]
  sampling.rate = 0.80
  trainingWhite = sample(1:num_samplesWhite, sampling.rate * num_samplesWhite, replace=FALSE)
  trainingSetWhite = fullWhiteSet[trainingWhite,]
  testingWhite = setdiff(1:num_samplesWhite, trainingWhite)
  testingSetWhite = fullWhiteSet[testingWhite,]
  sizeTestSetWhite = dim(testingSetWhite)[1]
  
  num_samplesRed = dim(fullRedSet)[1]
  sampling.rate = 0.80
  trainingRed = sample(1:num_samplesRed, sampling.rate * num_samplesRed, replace=FALSE)
  trainingSetRed = fullRedSet[trainingRed,]
  testingRed = setdiff(1:num_samplesRed, trainingRed)
  testingSetRed = fullRedSet[testingRed,]
  sizeTestSetRed = dim(testingSetRed)[1]
  
  
  #--------------------LINEAR REGRESSION----------------------#
  
  #To identify statistically insignificant features, we run linear regressions on each full data set and remove features with P values greater than 0.05
  whiteRegModel = lm(quality ~., data=fullWhiteSet)
  redRegModel = lm(quality~., data=fullRedSet)
  summary(whiteRegModel)
  summary(redRegModel)
  
  whiteRegModelAdjusted = lm(quality ~ fixed.acidity + volatile.acidity + residual.sugar + free.sulfur.dioxide + density + pH + sulphates + alcohol, data=trainingSetWhite)
  redRegModelAdjusted = lm(quality~volatile.acidity + chlorides + total.sulfur.dioxide + pH + sulphates + alcohol, data=trainingSetRed)
  
  whiteRegPredictions = predict(whiteRegModelAdjusted, testingSetWhite)
  redRegPredictions = predict(redRegModelAdjusted, testingSetRed)
  
  whiteRegError = whiteRegPredictions - testingSetWhite$quality
  redRegError = redRegPredictions - testingSetRed$quality
  
  whiteRegMSE = mean(whiteRegError^2)
  redRegMSE = mean(redRegError^2)
  
  AllWhiteRegErrors[fold] = whiteRegMSE
  AllRedRegErrors[fold] = redRegMSE
  
  
  #----------------------DECISION TREE------------------------#
  
  #Full White Set
  #Create decision tree model
  whiteTreeModel=rpart(quality~., data=trainingSetWhite)
  plot(whiteTreeModel, margin=0.1)
  text(whiteTreeModel)
  rpart.plot(whiteTreeModel)
  plotcp(whiteTreeModel)
  
  #Prune the tree
  pruned_whiteTreeModel=prune(whiteTreeModel, cp=0.024)
  plot(pruned_whiteTreeModel, margin=0.1)
  text(pruned_whiteTreeModel)
  rpart.plot(pruned_whiteTreeModel)
  predictedLabels=predict(pruned_whiteTreeModel, testingSetWhite)
  whiteTreeError=predictedLabels - testingSetWhite$quality
  whiteTreeMSE = mean(whiteTreeError^2)
  
  #Full Red Set
  #Create decision tree model
  redTreeModel=rpart(quality~., data=trainingSetRed)
  plot(redTreeModel, margin=0.1)
  text(redTreeModel)
  rpart.plot(redTreeModel)
  plotcp(redTreeModel)
  
  #Prune the tree
  pruned_redTreeModel=prune(redTreeModel, cp=0.025)
  plot(pruned_redTreeModel, margin=0.1)
  text(pruned_redTreeModel)
  rpart.plot(pruned_redTreeModel)
  predictedLabels=predict(pruned_redTreeModel, testingSetRed)
  redTreeError=predictedLabels - testingSetRed$quality
  redTreeMSE = mean(redTreeError^2)
  
  AllWhiteTreeErrors[fold] = whiteTreeMSE
  AllRedTreeErrors[fold] = redTreeMSE
  
  #----------------------RANDOM FOREST------------------------#
  
  #Now we run our random forest models 
  whiteForestModel = randomForest(quality ~., data = trainingSetWhite)
  plot(whiteForestModel)
  #Prune forest model
  whiteForestModel = randomForest(quality ~., data = trainingSetWhite, ntree=200 )
  
  redForestModel = randomForest(quality ~ .,  data = trainingSetRed)
  plot(redForestModel)
  #Prune forest model
  redForestModel = randomForest(quality ~ .,  data = trainingSetRed, ntree=200)
  
  #Now we will evaluate the models using the testing data
  predictions.White = predict(whiteForestModel, testingSetWhite)
  whiteForestError = predictions.White - testingSetWhite$quality
  whiteForestMSE = mean(whiteForestError^2)
  
  predictions.Red = predict(redForestModel, testingSetRed)
  redForestError = predictions.Red - testingSetRed$quality
  redForestMSE = mean(redForestError^2)
  
  AllWhiteForestErrors[fold] = whiteForestMSE
  AllRedForestErrors[fold] = redForestMSE
  
  #------------------SUPPORT VECTOR MACHINE-------------------#
  
  #SVR Model
  svrModelWhite = svm(quality~., trainingSetWhite)
  svrModelRed = svm(quality~., trainingSetRed)
  svrWhitePredictions = predict(svrModelWhite, testingSetWhite)
  svrRedPredictions = predict(svrModelRed, testingSetRed)
  
  whiteSvrError = svrWhitePredictions - testingSetWhite$quality
  redSvrError = svrRedPredictions - testingSetRed$quality
  
  whiteSvrMSE = mean(whiteSvrError^2)
  redSvrMSE = mean(redSvrError^2)
  
  AllWhiteSvrErrors[fold] = whiteSvrMSE
  AllRedSvrErrors[fold] = redSvrMSE
  
  #-------------------K NEAREST NEIGHBOURS--------------------#
  
  #Normalize the test and training data
  scaledWhiteTrainingSet = trainingSetWhite
  scaledWhiteTestingSet = testingSetWhite
  scaledRedTrainingSet = trainingSetRed
  scaledRedTestingSet = testingSetRed
  scaledWhiteTrainingSet[,1:11] = scale(trainingSetWhite[,1:11])
  scaledWhiteTestingSet[,1:11] = scale(testingSetWhite[,1:11])
  scaledRedTrainingSet[,1:11] = scale(trainingSetRed[,1:11])
  scaledRedTestingSet[,1:11] = scale(testingSetRed[,1:11])
  
  
  #Red Wine Features & Labels
  redTrainingFeatures <- subset(scaledRedTrainingSet, select=c(-quality)) # Get the labels of the training set
  redTrainingLabels <- scaledRedTrainingSet$quality
  #Get the features of the testing set
  redTestingFeatures <- subset(scaledRedTestingSet, select=c(-quality))
  
  #White Wine Features & Labels
  whiteTrainingFeatures <- subset(scaledWhiteTrainingSet, select=c(-quality)) # Get the labels of the training set
  whiteTrainingLabels <- scaledWhiteTrainingSet$quality
  #Get the features of the testing set
  whiteTestingFeatures <- subset(scaledWhiteTestingSet, select=c(-quality))
  
  #Red Wine KNN Model
  redKnnPredictions = knn.reg(redTrainingFeatures, redTestingFeatures, redTrainingLabels, k = 10)
  
  redKnnError = redKnnPredictions$pred - scaledRedTestingSet$quality
  redKnnMSE = mean(redKnnError^2)
  
  #White Wine KNN Model
  whiteKnnPredictions = knn.reg(whiteTrainingFeatures, whiteTestingFeatures, whiteTrainingLabels, k=10)
  summary(whiteKnnPredictions)
  
  whiteKnnError = whiteKnnPredictions$pred - scaledWhiteTestingSet$quality
  whiteKnnMSE = mean(whiteKnnError^2)
  
  AllWhiteKnnErrors[fold] = whiteKnnMSE
  AllRedKnnErrors[fold] = redKnnMSE
  
}

#---------------------NEURAL NETWORKS-----------------------#

#Build and Test Red Neural Net Model
redNeuralModel <- neuralnet(quality~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=scaledRedTrainingSet, hidden=c(2,2), linear.output = TRUE, stepmax = 1e7)
redNeuralModel$result.matrix
plot(redNeuralModel)
redPredictedLabels <- compute(redNeuralModel,scaledRedTestingSet[,1:11])
redNeuralResults <- data.frame(actual = scaledRedTestingSet[,12], prediction = redPredictedLabels$net.result)
redNeuralResults
redNeuralError = redNeuralResults$actual - redNeuralResults$prediction
redNeuralMSE = mean(redNeuralError^2)

#Build and Test White Neural Net Model
whiteNeuralModel <- neuralnet(quality~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=scaledWhiteTrainingSet, hidden=c(2,2), linear.output = TRUE, stepmax = 1e7)
whiteNeuralModel$result.matrix
plot(whiteNeuralModel)
whitePredictedLabels <- compute(whiteNeuralModel,scaledWhiteTestingSet[,1:11])
whiteNeuralResults <- data.frame(actual = scaledWhiteTestingSet[,12], prediction = whitePredictedLabels$net.result)
whiteNeuralResults
whiteNeuralError = whiteNeuralResults$actual - whiteNeuralResults$prediction
whiteNeuralMSE = mean(whiteNeuralError^2)

#---------------------COMPARE MSE VALUES--------------------#
mean(AllWhiteRegErrors)
mean(AllWhiteTreeErrors)
mean(AllWhiteForestErrors)
mean(AllWhiteKnnErrors)
mean(AllWhiteSvrErrors)
whiteNeuralMSE

mean(AllRedRegErrors)
mean(AllRedTreeErrors)
mean(AllRedForestErrors)
mean(AllRedKnnErrors)
mean(AllRedSvrErrors)
redNeuralMSE

#Edge case analysis, check how random forest MSE differs on a merged dataset instead
updatedWhiteSet = fullWhiteSet
updatedRedSet = fullRedSet
updatedWhiteSet$RW = 1
updatedRedSet$RW = 0
updatedWhiteSet = updatedWhiteSet[,c(1,2,3,4,5,6,7,8,9,10,11,13,12)]
updatedRedSet = updatedRedSet[,c(1,2,3,4,5,6,7,8,9,10,11,13,12)]
mergedDataSet = rbind(updatedWhiteSet, updatedRedSet)

AllCombinedErrors = c()

for(fold2 in 1:5)
{
  num_samplesMerged = dim(mergedDataSet)[1]
  trainingMerged = sample(1:num_samplesMerged, sampling.rate * num_samplesMerged, replace=FALSE)
  trainingSetMerged = mergedDataSet[trainingMerged,]
  testingMerged = setdiff(1:num_samplesMerged, trainingMerged)
  testingSetMerged = mergedDataSet[testingMerged,]
  sizeMergedSet = dim(testingSetMerged)[1]
  mergedRandomForest = randomForest(quality ~., data = trainingSetMerged)
  mergedRandomPredictions = predict(mergedRandomForest, testingSetMerged)
  mergedError = mergedRandomPredictions - testingSetMerged$quality
  mergedMSE = mean(mergedError^2)
  AllCombinedErrors[fold2] = mergedMSE
}

mean(AllCombinedErrors)


