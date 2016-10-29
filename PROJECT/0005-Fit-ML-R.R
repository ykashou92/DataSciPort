setwd("f://aquarius.fstruct//hobby//data science")
library(caret) #For training datasets and applying machine learning & genetic algorithms
library(ggplot2) #For awesome plotting
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(e1071)
set.seed(111)

# We will use url0 for the training dataset and url1 for the testing dataset
url0 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" 
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Download and Parition dataset
train <- read.csv(url(url0))
test <- read.csv(url(url1))

#Partition the training data into 60% training and 40% validation and check dimensions.
trainIndex <- createDataPartition(train$classe, p = .60, list = FALSE)
trainData <- train[ trainIndex, ]
validData <- train[-trainIndex, ]
dim(trainData)
dim(validData)

# Both return 160 variables, many of which are filled with NA records
# If over 90% of a variable is filled with NA then it is omitted from the training and test datasets
trainData <- trainData[, colMeans(is.na(trainData)) < 0.1]
dim(trainData)
validData <- validData[, colMeans(is.na(validData)) < 0.1]
dim(validData)

# We can remove the first five colums which are ID columns, as well as the timestamp as we do not need it in this analysis.
trainData <- trainData[, -(1:5)]
validData <- validData[, -(1:5)]
dim(trainData)
dim(validData)

# We can also remove all variables with nearly zero variance
near.zero.var <- nearZeroVar(trainData)
trainData <- trainData[, -near.zero.var]
validData <- validData[, -near.zero.var]
dim(trainData)
dim(validData)

# `trainData` and `validData` now have 54 variables, down from 160

# Successful Decision Tree
mod.train.dt <- train(classe ~ ., method = "rpart", data = trainData)
mod.predict.dt <- predict(mod.train.dt, validData)
cm.dt <- confusionMatrix(mod.predict.dt, validData$classe)
print(mod.train.dt$finalModel)
fancyRpartPlot(mod.train.dt$finalModel,cex=.5,under.cex=1,shadow.offset=0)

# Successful Random Forest
mod.train.rf <- randomForest(classe ~ ., data = trainData, mtry = 3, ntree = 200, do.trace = 25)
mod.predict.rf <- predict(mod.train.rf, validData)
cm.rf <- confusionMatrix(mod.predict.rf, validData$classe)
# Variable Importance According to random Forest
imp.rf <- importance(mod.train.rf)
varImpPlot(mod.train.rf, n.var = 15, sort = TRUE, main = "Variable Importance", lcolor = "blue", bg = "purple")

# Successful Support Vector Machine
mod.train.svm <- svm(classe ~ ., data = trainData)
mod.predict.svm <- predict(mod.train.svm, validData)
cm.svm <- confusionMatrix(mod.predict.svm, validData$classe)

# Comparing Accuracies
a.dt <- cm.dt$overall[1]
a.rf <- cm.rf$overall[1]
a.svm <- cm.svm$overall[1]

cm.dataframe <- data.frame(Algorithm = c("Decision Tree", "Random Forest", "Support Vector Machine"), Index = c("dt", "rf", "svm"), Accuracy = c(a.dt, a.rf, a.svm))
cm.dataframe <- arrange(cm.dataframe, desc(Accuracy))
cm.dataframe

# In sample Error Rate
InSampError.rf <- (1 - 0.994)*100
InSampError.rf

# Out of sample Error Rate
print(mod.train.rf)

# Print Final Prediction Results of the Algorithm with Highest Accuracy (Random Forest)
fp.rf <- predict(mod.train.rf, newdata=test)
fp.rf