# Gregory Johnson
# Final Project

rm(list=ls(all=TRUE))
options(warn=-1)
set.seed(1234)

# For predicting how the present state of the environment will change

# Get data from the csv
climData <- read.csv("1-1994_3-2017_Climate_Full.csv")
climData <- climData[1:21]
numRows = nrow(climData)
numCols = ncol(climData)
myRows <- c(1:numRows)

# Line plot of all sets of data (columns)
par(mfrow=c(3,3))
for(i in 4:numCols) { # skip index [1,2,3] because plotting would be pointless
  heading = paste("",colnames(climData)[i])
  temp <- as.vector(as.matrix(climData[i]))
  plot(myRows, temp, main = heading, xlab="Month # in Data", ylab=colnames(climData)[i])
  lines(myRows, temp, type="l")
}

# For getting max, min and standard deviation of each column
colMax <- function(data) sapply(data, max, na.rm = TRUE)
colMin <- function(data) sapply(data, min, na.rm = TRUE)
colSd <- function(data) sapply(data, sd, na.rm = TRUE)

colMax(climData[4:numCols])
colMin(climData[4:numCols])
colSd(climData[4:numCols])

# Create test and training groups
#   Take every 9th and put in test dataset
#   Put the rest in training dataset
range = 1:numRows
multsOf = (range %% 18) == 1 #its equal to 1 because we start by taking the first element. Ex: 1, 10, 19, etc.
clippedData <- climData[,3:21]
getMonth <- function(x) substr(x, 5, length(x))
clippedData <- data.frame(apply(clippedData[1],2,getMonth), clippedData[2:19])
clippedData$YearMonth <- as.numeric(clippedData$YearMonth)
test<-clippedData[multsOf,]
training<-clippedData[!multsOf,]

# K nearest neighbor------------------------------------------
library(class)
# Used to measure the performance of knn 
checkPerformance <- function(testWithAnswer, knnResults) {
  num <- nrow(testWithAnswer)
  count = 0
  for(i in 1:num) {
    if(testWithAnswer[i,19] == knnResults[i]) {
      count = count + 1
    }
  }
  return (count/num)
}

for(n in 1:10) {
  myResults <- knn(training[,-19],test[,-19],training[,19],k=n)
  print(paste("K =",n,":",checkPerformance(test, myResults)), quote=F)
}
'
[1] K = 1 : 0.3125
[1] K = 2 : 0.3125
[1] K = 3 : 0.5
[1] K = 4 : 0.375
[1] K = 5 : 0.4375
[1] K = 6 : 0.4375
[1] K = 7 : 0.4375
[1] K = 8 : 0.3125
[1] K = 9 : 0.4375
[1] K = 10 : 0.375
'
# K nearest neighbor averages about 44% (Between k=4 and k=5) correct on predictions using the current training and testing datasets

# Artificial neural network-------------------------------------------------------------------
# Create vector of column max and min values
maxs <- apply(clippedData[,2:18], 2, max)
mins <- apply(clippedData[,2:18], 2, min)

# Use scale() and convert the resulting matrix to a data frame
scaled.data <- as.data.frame(scale(clippedData[,2:18] ,center = mins, scale = maxs - mins))
print(head(scaled.data,5))

library(caTools)

# Add the wetness indexes to the scaled dataset
YearMonth <- climData$YearMonth
WetnessIndex <- clippedData$Wet.Dry.Period
data = cbind(WetnessIndex,scaled.data)
rownames(data) <- YearMonth # name the rows so that they're easy to identify

# Creating split of the data
split = sample.split(data$WetnessIndex, SplitRatio = 0.75)

# Split into trainging and test sets
train = subset(data, split == TRUE)
test = subset(data, split == FALSE)

feats <- names(scaled.data)

# Concatenate strings
f <- paste(feats, collapse=' + ')
f <- paste('WetnessIndex ~', f)

# Convert to formula
f <- as.formula(f)
f

# neuralnet creation
library(neuralnet)
nn <- neuralnet(f,train,hidden=10,linear.output=FALSE)
plot(nn)

# Compute Predictions off Test Set
pr.nn <- compute(nn,test[-1])

pr.nn_ <- pr.nn$net.result*(max(data$WetnessIndex)-min(data$WetnessIndex))+min(data$WetnessIndex)
pr.nn_ <- sapply(pr.nn_,round,digits=0)
test.r <- (test$WetnessIndex)*(max(data$WetnessIndex)-min(data$WetnessIndex))+min(data$WetnessIndex)

MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test)

par(mfrow=c(1,1))
plot(test$WetnessIndex,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
lmRes <- lm(pr.nn_ ~ test$WetnessIndex)
print(paste("Success Rate: ", (1 - abs(coef(lmRes)[2] - 1)) * 100, "%")) #slope
abline(lmRes)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

table(test$WetnessIndex,pr.nn_)

# Read in the data
library(e1071)
library(randomForest)
nbData <- clippedData
nbData[] <- lapply(nbData, factor)

# Shows the categories ("factors") of the data
ls.str(nbData)

# Naïve Bayes
model <- naiveBayes(Wet.Dry.Period ~ ., data = nbData)
myPredict <- predict(model, nbData[,-19])
table(myPredict, nbData$Wet.Dry.Period)

# Random Forest does not work because there are too many levels within the factors

# Decision tree-----------------------------------------------------------------------
library(party)

input.data <- clippedData
target <- Wet.Dry.Period ~ .
output.tree <- ctree(target, data = input.data)
plot(output.tree)
predDT <- predict(output.tree)
table(predDT, input.data$Wet.Dry.Period)
print(paste("DT success =",checkPerformance(input.data, predDT)))

# Classification and Regression--------------------------------------------------
library(caret)
lm_model <- train(Wet.Dry.Period ~ ., data = clippedData, method = "lm") # linear regression model
summary(lm_model)

# Linear Discriminant Analysis------------------------------------------------- Good results
library(MASS)
fit <- lda(Wet.Dry.Period ~ ., data=clippedData, 
           na.action="na.omit", CV=TRUE)
plot(clippedData$Wet.Dry.Period,fit$class, pch = 16, col='blue', main="Real vs Predicted LDA")
abline(lm(fit$class ~ clippedData$Wet.Dry.Period))
 
# Assess the accuracy of the prediction percent correct
ct <- table(clippedData$Wet.Dry.Period, fit$class)
diag(prop.table(ct, 1))
# total percent correct
print(paste("Percent correct from data: ", sum(diag(prop.table(ct))) * 100, "%"))
print(paste("Success Rate: ", (1 - abs(coef(lm(fit$class ~ clippedData$Wet.Dry.Period))[2] - 1)) * 100, "%")) #slope
