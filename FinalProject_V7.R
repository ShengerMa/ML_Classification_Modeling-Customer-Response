#**********************
# 1.Importing Libraries
#**********************

if(!require(ggplot2)){install.packages('ggplot2'); require(ggplot2)}
if(!require(lattice)){install.packages('lattice'); require(lattice)}
if(!require(caret)){install.packages('caret'); require(caret)}
if(!require(RSQLite)){install.packages('RSQLite'); require(RSQLite)}
if(!require(sqldf)){install.packages('sqldf'); require(sqldf)}
if(!require(caTools)){install.packages('caTools'); require(caTools)} # visualize ROC curve
if(!require(mlbench)){install.packages('mlbench'); require(mlbench)}
if(!require(pillar)){install.packages('pillar'); require(pillar)}
if(!require(e1071)){install.packages('e1071'); require(e1071)} #SVM
if(!require(plyr)){install.packages('plyr'); require(plyr)}
if(!require(dplyr)){install.packages('dplyr'); require(dplyr)} #tidy verse
if(!require(tidyverse)){install.packages('tidyverse'); require(tidyverse)} #tidy verse
if(!require(gridExtra)){install.packages('gridExtra'); require(gridExtra)} #sub plots
if(!require(reshape2)){install.packages('reshape2'); require(reshape2)}
if(!require(corrplot)){install.packages('corrplot'); require(corrplot)}
if(!require(caretEnsemble)){install.packages('caretEnsemble'); require(caretEnsemble)}
if(!require(rpart.plot)){install.packages('rpart.plot'); require(rpart.plot)}
if(!require(pROC)){install.packages('pROC'); require(pROC)}
install.packages("/Users/chenyuejiang/Downloads/DMwR_0.4.1.tar.gz", repos = NULL, type = "source")
install.packages(c("xts","quantmod","ROCR"))
library(DMwR) #smote
#**********************
# 2.Loading Data
#**********************

data_ori<- read.csv('CustomerResponse.csv')
head(data_ori)
dim(data_ori)
str(data_ori)
summary(data_ori)
data_ori$response<-as.factor(data_ori$response)
#**********************
# 3.Data Cleaning
#**********************

#*3.1 Feature Engineering
#Reduce the redundant information
summary(data_ori$response)
data_ori$campaign <- as.factor(data_ori$campaign)
dmy <- dummyVars("~ campaign",data = data_ori)
campaign_trsf <- data.frame(predict(dmy,newdata = data_ori))
str(data_ori)
data_cleaning<- cbind(data_ori[,-2],campaign_trsf)

data_with_response <- subset(data_cleaning,response == 1)
data_without_response <- subset(data_cleaning,response == 0)
str(data_with_response)

remove_duplicate <- function(dataset){
  df_grp_cstmid <- dataset %>% group_by(customer_id) %>% 
    summarise(campaign_1 = sum(campaign.1),
              campaign_2 = sum(campaign.2),
              campaign_3 = sum(campaign.3),
              campaign_4 = sum(campaign.4))
  duplicated_cols <- c("Rowid","campaign.1","campaign.2","campaign.3","campaign.4")
  dataset<- dataset[,!names(dataset) %in% duplicated_cols]
  df <- sqldf("SELECT DISTINCT*
              FROM df_grp_cstmid
              LEFT JOIN dataset
              USING (customer_id)")
  return (df)
}

df1 <- remove_duplicate(data_with_response)
df2 <- remove_duplicate(data_without_response)

data_cleaning<- rbind(df1,df2)
summary(data_cleaning$response)
str(data_cleaning)
# transform age_youngest child to binary variable (0,1)
data_cleaning$has_child<-ifelse(data_cleaning$age_youngest_child ==0,0,1)

#*3.2 check for missing values
colSums(is.na(data_cleaning))
col_with_missing_values<-colnames(data_cleaning)[colSums(is.na(data_cleaning))>0]
print(col_with_missing_values)
# we can find most of the data of Column 'product_id', 'purchase', 'purchase_date' is missing values

#*3.3 Drop unnecessary features 
drop <- c("customer_id","response_date","purchase_date","product_id","age_youngest_child")
data = data_cleaning[,!(names(data_cleaning) %in% drop)]
names(data)
#**********************
# 4.EDA
#**********************

#4.1  double check numerical features for outliers 
str(data)
categorical_features <- c("response","campaign_1","campaign_2","campaign_3","campaign_4","has_child","marital","gender","branch")
###################
###################
###################
numeric_var <- data[,names(data)%in%categorical_features==FALSE]
outlier_check = boxplot(scale(numeric_var[,1:ncol(numeric_var)]),main="Boxplot of numerical features",xlab="Features",ylab="Values",col="blue",horizontal=TRUE,las=1,notch=TRUE,cex.axis=0.7,cex.lab=0.7,cex.main=0.8)
# see appendix for the plots of all the features

#4.2 perform chi-test for categorical variables
categorical_cols <- c("branch","gender","marital")
categorical_var = data[,names(data)%in%categorical_cols]
variable_name = c()
p_value = c()
for (x in names(categorical_var)){
  chi_test = chisq.test(data$response,data[[x]])
  variable_name = append(variable_name,x)
  p_value = append(p_value,chi_test$p.value)
}
ctgr_pValue = data.frame(column_name = variable_name,p_Value = p_value)

#4,3 plot the p value of each categorical variable against response and visualize the variable with p value < 0.05
plot_chi_test <- ggplot(ctgr_pValue, aes(x=column_name, y=p_Value)) + 
  geom_point() + 
  geom_hline(yintercept = 0.05, color = "red")+
  labs(y="p value", 
       x="categorical variable", 
       title="p value of each categorical variable against response")

#4.4 t-test for numerical variables
numerical_var = data[,!names(data)%in%c("gender","marital","branch","campaign_1","campaign_2","campaign_3","campaign_4","response")]
names(numerical_var)
variable_name = c()
p_value = c()
for (x in names(numerical_var)){
  t_test = t.test(c(as.numeric(data$response)),data[[x]])
  variable_name = append(variable_name,x)
  p_value = append(p_value,t_test$p.value)
}
nmr_pValue= data.frame(column_name = variable_name,p_Value = p_value)
#4.5 plot the p value of each numerical variable against response and visualize the variable with p value < 0.05
plot_t_test <- ggplot(nmr_pValue, aes(x=column_name, y=p_Value)) + 
  geom_point() + 
  geom_hline(yintercept = 0.01, color = "red")+
  labs(y="p value", 
       x="numarical value", 
       title="p value of each numerical variable against response")+ theme(axis.text.x = element_text(angle = 90, hjust = 1))

#4.6 correlation between numerical features
str(numerical_var)
M <- cor(numerical_var)
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}
  #4.6.1 matrix of the p-value of the correlation
p.mat <- cor.mtest(numerical_var)
head(p.mat[, 1:5])


col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(M, method="color", col=col(200),  
         type="lower", order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.pos = "ld",tl.srt=1, #Text label color and rotation
         # Combine with significance
         p.mat = p.mat, sig.level = 0.01, insig = "blank",
          # hide correlation coefficient on the principal diagonal
        diag=F,
         # Add title
         title="Correlation between numerical features", 
          # Add color legend
          addCoefasPercent = F, # Show correlation coefficient as percentage
          tl.cex = 0.7, # Text label size
          addgrid.col = "grey",number.cex = 0.7) # Add grid to make it more clear

#4.7 visualize variables that were tested insignificant in previous chi-test and t-test
plot_branch = ggplot(data = data, aes(x = branch, fill = response)) +
    geom_bar(position = "fill") + ylab("proportion") +
    theme_gray() +theme(plot.title = element_text(size=22),axis.text.x= element_text(size=15,angle = 45),
                            axis.text.y= element_text(size=15), axis.title=element_text(size=18))+
    ggtitle("Branch Correlation with Response")

plot_marital = ggplot(data = data, aes(x = marital, fill = response)) +
  geom_bar(position = "fill") + ylab("proportion") +
  theme_gray() +theme(plot.title = element_text(size=22),axis.text.x= element_text(size=15),
                      axis.text.y= element_text(size=15), axis.title=element_text(size=18))+
  ggtitle("Marital Correlation with Response")+scale_fill_manual(values=c("grey", "#173F74"))

plot_gender = ggplot(data = data, aes(x = gender, fill = response,)) +
  geom_bar(position = "fill") + ylab("proportion") +
  theme_grey() +theme(plot.title = element_text(size=22),axis.text.x= element_text(size=15),
                      axis.text.y= element_text(size=15), axis.title=element_text(size=18))+
  ggtitle("Gender Correlation with Response")+scale_fill_manual(values=c("grey", "#173F74"))

plot_loan_accounts = ggplot(data, aes(x=loan_accounts, color=response,fill=response)) +
  geom_histogram(alpha=0.2,position="identity",binwidth=1)+
  labs(x="Loan account",
       y="Count", 
       title="Distribution of Loan Accounts to response")+  
  theme_gray()+
  theme(plot.title = element_text(size=15)
        ,axis.text.x= element_text(size=15),
        axis.text.y= element_text(size=15),
        axis.title=element_text(size=18))
# we only plot noticable features here
plot_purchase = ggplot(data, aes(x=purchase,color=response,fill=response)) +
  geom_histogram(alpha=1,position="identity",binwidth=1)+
labs(x="Purchase",
         y="Count", 
       title="Distribution of Purchase for different Response")+  
theme_grey()+
theme(plot.title = element_text(size=15)
      ,axis.text.x= element_text(size=15),
       axis.text.y= element_text(size=15),
        axis.title=element_text(size=18))
###################
#histogram or bar plot
###################
  # Since there are only two values in the response variable, we can use bar plot
purchase_plot = ggplot(data = data, aes(x = purchase, fill = response,)) +
  geom_bar(position = "identity") + ylab("Count") +
  theme_grey() +theme(plot.title = element_text(size=22),axis.text.x= element_text(size=15),
                      axis.text.y= element_text(size=15), axis.title=element_text(size=18))+
  ggtitle("Purchase Correlation with Response")+scale_fill_manual(values=c("grey", "#173F74"))

#**********************
# 5.Pre-Processing 
#**********************
#*

#5.1 drop features according to EDA results
str(data)
drop_col = c("loan_accounts","gender","marital","purchase","branch")
data = data[,!names(data)%in%drop_col]
#relocate column
data<-data %>% relocate('response')
str(data)

#5.2 transform the response column to allow for classProbs
levels(data$response) <- make.names(levels(factor(data$response)))
str(data)
# 5.3 preprocessing - standardization and dimension reduction (PCA)
###################
# with or without PCA
###################
mypreProcess = c("center","scale","pca")

#5.4 split data into train and test
set.seed(123)
data_smote = SMOTE(response~.,data,perc.over=200,perc.under=150)
index_smote = createDataPartition(data_smote$response,p=0.8,list=F)
summary(data_smote$response)
train = data_smote[index_smote,]
test = data_smote[-index_smote,]
summary(train$response)
summary(test$response)
###################
# downsample
###################
data_down = downSample(data[,-1],data$response,list=FALSE,yname="response")
summary(data_down)
index_down = createDataPartition(data_down$response,p=0.8,list=F)
train = data_down[index_down,]
summary(train$response)
test = data_down[-index_down,]
summary(test$response)
###################
# direct split into train and test
###################
index = createDataPartition(data$response,p=0.8,list=F)
train = data[index,]
summary(train$response)
test = data[-index,]
summary(test$response)
#**********************
# 6.Modeling
#**********************
#*

#6.1 set random seed
set.seed(123)
#6.2 define the control
TControl <- trainControl(method = "cv",number = 10,summaryFunction = twoClassSummary,classProbs = TRUE,index = createFolds(train$response, k = 5))
#6.3 generate the structure of the report
report <- data.frame(Model=character(), Acc.Train=numeric(), Kappa.Train = numeric(),Acc.Test=numeric(),Kappa.Test = numeric())
balanced_accuracy_test = data.frame(Model=character(), balanced_accuracy_test=numeric())
specificity = data.frame(Model=character(), Specificity_train=numeric(),Specificity_test=numeric())
sensitivity = data.frame(Model=character(), Sensitivity_train=numeric(),Sensitivity_test=numeric())
F1_score = data.frame(Model=character(), F1_score_train=numeric(),F1_score_test=numeric())
#******************
#6.4 KNN 
#******************
knnGrid <- expand.grid(k = c(seq(3,25,2)))
knnmodel <- train(response~.,data=train, method="knn", tuneGrid=knnGrid, trControl=TControl,preProcess = mypreProcess)

prediction.train <-predict(knnmodel,train[,-1],type = "raw")
prediction.test <-predict(knnmodel,test[,-1],type = "raw")
acctr <- confusionMatrix(prediction.train,train[,1])
acctr$table
acctr$overall['Accuracy']
accte <-confusionMatrix(prediction.test,test$response)
accte$byClass['Balanced Accuracy']
accte$table
accte$overall['Accuracy']
# report
report<-rbind(report, data.frame(Model="k-NN", Acc.Train=acctr$overall['Accuracy'], Kappa.Train = acctr$overall['Kappa'],Acc.Test=accte$overall['Accuracy'],Kappa.Test = accte$overall['Kappa']))
# balanced accuracy
balanced_accuracy_test = rbind(balanced_accuracy_test, data.frame(Model="k-NN", balanced_accuracy_test=accte$byClass['Balanced Accuracy']))
# specificity
specificity = rbind(specificity, data.frame(Model="k-NN", Specificity_train=acctr$byClass['Specificity'],Specificity_test=accte$byClass['Specificity']))
# sensitivity
sensitivity = rbind(sensitivity, data.frame(Model="k-NN", Sensitivity_train=acctr$byClass['Sensitivity'],Sensitivity_test=accte$byClass['Sensitivity']))
# F1 score
F1_score = rbind(F1_score, data.frame(Model="k-NN", F1_score_train=acctr$byClass['F1'],F1_score_test=accte$byClass['F1']))

# plot(knnmodel)
colAUC(as.numeric(prediction.test),test$response,plotROC=TRUE)
colAUC(as.numeric(prediction.train),train$response,plotROC=TRUE)

#**********************
#6.5 C5.0
#**********************
set.seed(123)
c50Grid <- expand.grid(trials = c(1:5),
                       model = c("tree", "rules"),
                       winnow = FALSE)
c5model <- train(response ~., data=train, method="C5.0", tuneGrid=c50Grid, trControl=TControl,preProcess = mypreProcess)

prediction.train <- predict(c5model, train[,-1], type="raw")
prediction.test <- predict(c5model, test[,-1], type="raw")
acctr <- confusionMatrix(prediction.train, train[,1])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,1])
accte$table
# report
report <- rbind(report, data.frame(Model="C5.0", Acc.Train=acctr$overall['Accuracy'], Kappa.Train = acctr$overall['Kappa'],Acc.Test=accte$overall['Accuracy'],Kappa.Test = accte$overall['Kappa']))
# balanced accuracy
balanced_accuracy_test = rbind(balanced_accuracy_test, data.frame(Model="C5.0", balanced_accuracy_test=accte$byClass['Balanced Accuracy']))
# specificity
specificity = rbind(specificity, data.frame(Model="C5.0", Specificity_train=acctr$byClass['Specificity'],Specificity_test=accte$byClass['Specificity']))
# sensitivity
sensitivity = rbind(sensitivity, data.frame(Model="C5.0", Sensitivity_train=acctr$byClass['Sensitivity'],Sensitivity_test=accte$byClass['Sensitivity']))
# F1 score
F1_score = rbind(F1_score, data.frame(Model="C5.0", F1_score_train=acctr$byClass['F1'],F1_score_test=accte$byClass['F1']))

# plot(c5model) 
colAUC(as.numeric(prediction.test),test$response,plotROC=TRUE) 
colAUC(as.numeric(prediction.train),train$response,plotROC = TRUE)

install.packages('partykit')
library(partykit)
library(C50)
names(train)
C50Model = C5.0(response~income + campaign_3+months_current_account+campaign_4+campaign_2+average.balance.feed.index+campaign_1+months_customer+number_transactions+number_products,data=train,trials=10)
plot(C50Model,main="C5.0 Tree",subtree = 22,typr = "simple",cex = 0.8,shrink = 0.8)
#**********************
#6.6 Cart
#**********************
set.seed(123)
cartmodel <- train(response ~., data=train, method="rpart", trControl=TControl,preProcess=c("center","scale"))
cartmodel 

prediction.train <- predict(cartmodel, train[,-1], type="raw")
prediction.test <- predict(cartmodel, test[,-1], type="raw")
acctr <- confusionMatrix(prediction.train, train[,1])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,1])
accte$table
accte$overall['Accuracy']
# report
report <- rbind(report, data.frame(Model="CART", Acc.Train=acctr$overall['Accuracy'], Kappa.Train = acctr$overall['Kappa'],Acc.Test=accte$overall['Accuracy'],Kappa.Test = accte$overall['Kappa']))
# balanced accuracy
balanced_accuracy_test = rbind(balanced_accuracy_test, data.frame(Model="CART", balanced_accuracy_test=accte$byClass['Balanced Accuracy']))
# specificity
specificity = rbind(specificity, data.frame(Model="CART", Specificity_train=acctr$byClass['Specificity'],Specificity_test=accte$byClass['Specificity']))
# sensitivity
sensitivity = rbind(sensitivity, data.frame(Model="CART", Sensitivity_train=acctr$byClass['Sensitivity'],Sensitivity_test=accte$byClass['Sensitivity']))
# F1 score
F1_score = rbind(F1_score, data.frame(Model="CART", F1_score_train=acctr$byClass['F1'],F1_score_test=accte$byClass['F1']))

# rpart.plot(cartmodel$finalModel, type=4, extra=1, under=TRUE, faclen=0)
colAUC(as.numeric(prediction.test),test$response,plotROC=TRUE) 
colAUC(as.numeric(prediction.train),train$response,plotROC = TRUE)
#**********************
#6.7 Random Forest
#**********************
besta <- 0
bestn <- 0
bestm <- 0
set.seed(123)
for (maxnodes in c(8, 16, 24, 100)) {
  set.seed(123)
  rfGrid <- expand.grid(mtry = c(seq(2, 10)))
  rformodel <- train(response ~., data=train, method="rf", tuneGrid=rfGrid, trControl=TControl, maxnodes=maxnodes,preProcess=mypreProcess)
  cat("\n\n maxnodes=", maxnodes, "  ROC=", max(rformodel$results$ROC), "  mtry=", rformodel$bestTune$mtry)
  if (max(rformodel$results$ROC)>besta) {
    bestn <- maxnodes
    bestm <- rformodel$bestTune$mtry
    besta <- max(rformodel$results$ROC)
  }
}


###########################
rfGrid <- expand.grid(mtry = bestm)
rformodel <- train(response ~., data=train, method="rf", tuneGrid=rfGrid, trControl=TControl, maxnodes=bestn,preProcess=mypreProcess)
rformodel
library(randomForest)
rf_model = randomForest(response ~., data=train, mtry=bestm, maxnodes=bestn, importance=TRUE, ntree=500)
hist(treesize(rf_model),
     main = "No. of Nodes for the Trees",
     col = "green")
#Variable Importance
varImpPlot(rf_model, main="Variable Importance", n.var=10, cex=0.8, col="black", sort=TRUE, pch=19, cex.axis=0.8, cex.lab=0.8, cex.main=0.8, cex.sub=0.8)
importance(rf_model)
#MeanDecreaseGini
###########################

prediction.train <- predict(rformodel, train[,-1], type="raw")
prediction.test <- predict(rformodel, test[,-1], type="raw")

acctr <- confusionMatrix(prediction.train, train[,1])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,1])
accte$table
accte$overall['Accuracy']
# report
report <- rbind(report, data.frame(Model="Random Forest", Acc.Train=acctr$overall['Accuracy'], Kappa.Train = acctr$overall['Kappa'],Acc.Test=accte$overall['Accuracy'],Kappa.Test = accte$overall['Kappa']))
# balanced accuracy
balanced_accuracy_test = rbind(balanced_accuracy_test, data.frame(Model="Random Forest", balanced_accuracy_test=accte$byClass['Balanced Accuracy']))
# specificity
specificity = rbind(specificity, data.frame(Model="Random Forest", Specificity_train=acctr$byClass['Specificity'],Specificity_test=accte$byClass['Specificity']))
# sensitivity
sensitivity = rbind(sensitivity, data.frame(Model="Random Forest", Sensitivity_train=acctr$byClass['Sensitivity'],Sensitivity_test=accte$byClass['Sensitivity']))
# F1 score
F1_score = rbind(F1_score, data.frame(Model="Random Forest", F1_score_train=acctr$byClass['F1'],F1_score_test=accte$byClass['F1']))

# plot(rformodel$finalModel)
colAUC(as.numeric(prediction.test),test$response,plotROC=TRUE)
colAUC(as.numeric(prediction.train),train$response,plotROC = TRUE)
#**********************
#6.8 Logistic Regression
#**********************
set.seed(123)
glmGrid = expand.grid(alpha=0:1,lambda = seq(0.0001,0.1,length = 10))
lrmodel <- train(response~., data=train, method="glmnet", trControl=TControl,tuneGrid=glmGrid,preProcess=mypreProcess)
lrmodel

prediction.train <- predict(lrmodel, train[,-1], type="raw")
prediction.test <- predict(lrmodel, test[,-1], type="raw")
acctr <- confusionMatrix(prediction.train, train[,1])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,1])
accte$table
accte$overall['Accuracy']
# report
report <- rbind(report, data.frame(Model="Logistic Regression", Acc.Train=acctr$overall['Accuracy'], Kappa.Train = acctr$overall['Kappa'],Acc.Test=accte$overall['Accuracy'],Kappa.Test = accte$overall['Kappa']))
# balanced accuracy
balanced_accuracy_test = rbind(balanced_accuracy_test, data.frame(Model="Logistic Regression", balanced_accuracy_test=accte$byClass['Balanced Accuracy']))
# specificity
specificity = rbind(specificity, data.frame(Model="Logistic Regression", Specificity_train=acctr$byClass['Specificity'],Specificity_test=accte$byClass['Specificity']))
# sensitivity
sensitivity = rbind(sensitivity, data.frame(Model="Logistic Regression", Sensitivity_train=acctr$byClass['Sensitivity'],Sensitivity_test=accte$byClass['Sensitivity']))
# F1 score
F1_score = rbind(F1_score, data.frame(Model="Logistic Regression", F1_score_train=acctr$byClass['F1'],F1_score_test=accte$byClass['F1']))

#3plot(lrmodel)
# plot(lrmodel$finalModel)
colAUC(as.numeric(prediction.test),test$response,plotROC=TRUE) 
colAUC(as.numeric(prediction.train),train$response,plotROC = TRUE)
#**********************
#6.9   Neural Network
#**********************
set.seed(123)
nnmodel <- train(response~., data=train, method="nnet", trControl=TControl, trace = FALSE,preProcess=mypreProcess)
nnmodel

prediction.train <- predict(nnmodel, train[,-1], type="raw")
prediction.test <- predict(nnmodel, test[,-1], type="raw")
acctr <- confusionMatrix(prediction.train, train[,1])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,1])
accte$table
accte$overall['Accuracy']
# report
report <- rbind(report, data.frame(Model="Neural Network", Acc.Train=acctr$overall['Accuracy'], Kappa.Train = acctr$overall['Kappa'],Acc.Test=accte$overall['Accuracy'],Kappa.Test = accte$overall['Kappa']))
# balanced accuracy
balanced_accuracy_test = rbind(balanced_accuracy_test, data.frame(Model="Neural Network", balanced_accuracy_test=accte$byClass['Balanced Accuracy']))
# specificity
specificity = rbind(specificity, data.frame(Model="Neural Network", Specificity_train=acctr$byClass['Specificity'],Specificity_test=accte$byClass['Specificity']))
# sensitivity
sensitivity = rbind(sensitivity, data.frame(Model="Neural Network", Sensitivity_train=acctr$byClass['Sensitivity'],Sensitivity_test=accte$byClass['Sensitivity']))
# F1 score
F1_score = rbind(F1_score, data.frame(Model="Neural Network", F1_score_train=acctr$byClass['F1'],F1_score_test=accte$byClass['F1']))

# plot(nnmodel)
# print(nnmodel$finalModel)
colAUC(as.numeric(prediction.test),test$response,plotROC=TRUE) 
colAUC(as.numeric(prediction.train),train$response,plotROC = TRUE)
#************************
#6.10 Support Vector Machine
#************************
#6.10.1 linear
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
set.seed(123)
svmmodel.l <- train(response ~., data=train, method="svmLinear", trControl=TControl, tuneGrid = grid,preProcess=mypreProcess)
svmmodel.l

prediction.train <- predict(svmmodel.l, train[,-1], type="raw")
prediction.test <- predict(svmmodel.l, test[,-1], type="raw")
acctr <- confusionMatrix(prediction.train, train[,1])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,1])
accte$table
accte$overall['Accuracy']
# report
report <- rbind(report, data.frame(Model="SVM (Linear)", Acc.Train=acctr$overall['Accuracy'], Kappa.Train = acctr$overall['Kappa'],Acc.Test=accte$overall['Accuracy'],Kappa.Test = accte$overall['Kappa']))
# balanced accuracy
balanced_accuracy_test = rbind(balanced_accuracy_test, data.frame(Model="SVM (Linear)", balanced_accuracy_test=accte$byClass['Balanced Accuracy']))
# specificity
specificity = rbind(specificity, data.frame(Model="SVM (Linear)", Specificity_train=acctr$byClass['Specificity'],Specificity_test=accte$byClass['Specificity']))
# sensitivity
sensitivity = rbind(sensitivity, data.frame(Model="SVM (Linear)", Sensitivity_train=acctr$byClass['Sensitivity'],Sensitivity_test=accte$byClass['Sensitivity']))
# F1 score
F1_score = rbind(F1_score, data.frame(Model="SVM (Linear)", F1_score_train=acctr$byClass['F1'],F1_score_test=accte$byClass['F1']))

colAUC(as.numeric(prediction.test),test$response,plotROC=TRUE) 
colAUC(as.numeric(prediction.train),train$response,plotROC = TRUE)

#6.10.2 radial
grid <- expand.grid(sigma = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9),
                    C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
set.seed(123)
svmmodel.r <- train(response ~., data=train, method="svmRadial", trControl=TControl, tuneGrid = grid,preProcess=mypreProcess)
svmmodel.r
prediction.train <- predict(svmmodel.r, train[,-1], type="raw")
prediction.test <- predict(svmmodel.r, test[,-1], type="raw")
acctr <- confusionMatrix(prediction.train, train[,1])
acctr$table
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,1])
accte$table
accte$overall['Accuracy']
# report
report <- rbind(report, data.frame(Model="SVM (Radial)", Acc.Train=acctr$overall['Accuracy'], Kappa.Train = acctr$overall['Kappa'],Acc.Test=accte$overall['Accuracy'],Kappa.Test = accte$overall['Kappa']))
# balanced accuracy
balanced_accuracy_test = rbind(balanced_accuracy_test, data.frame(Model="SVM (Radial)", balanced_accuracy_test=accte$byClass['Balanced Accuracy']))
# specificity
specificity = rbind(specificity, data.frame(Model="SVM (Radial)", Specificity_train=acctr$byClass['Specificity'],Specificity_test=accte$byClass['Specificity']))
# sensitivity
sensitivity = rbind(sensitivity, data.frame(Model="SVM (Radial)", Sensitivity_train=acctr$byClass['Sensitivity'],Sensitivity_test=accte$byClass['Sensitivity']))
# F1 score
F1_score = rbind(F1_score, data.frame(Model="SVM (Radial)", F1_score_train=acctr$byClass['F1'],F1_score_test=accte$byClass['F1']))

colAUC(as.numeric(prediction.test),test$response,plotROC=TRUE) 
colAUC(as.numeric(prediction.train),train$response,plotROC = TRUE)
#**************
#7 Final Report
#**************

#7.1 show training results
results <- resamples(list(KNN=knnmodel, C5.0=c5model, CART=cartmodel,
                          RFor=rformodel, LogReg=lrmodel, NeuNet=nnmodel,  
                          SVM.L=svmmodel.l, SVM.R=svmmodel.r))
names(results)
dotplot(results)

#7.2 print report
report
  # 7.2.1 visualize the test accuracy of all models
ggplot(report, aes(x=Model, y=Acc.Test))+
    geom_point(col="blue", size=5)+
    labs(x="Model",
         y="Model Accuracy", 
       title="Report")
  #7.2.2 visualize the train accuracy of all models
ggplot(report, aes(x=Model, y=Acc.Train))+
    geom_point(col="blue", size=5)+
    labs(x="Model",
         y="Model Accuracy", 
       title="Report")
  #7.2.3 visualize the test kappa of all models
ggplot(report, aes(x=Model, y=Kappa.Test))+
    geom_point(col="blue", size=5)+
    labs(x="Model",
         y="Model Kappa", 
       title="Report")

  #7.2.4 visualize the train kappa of all models
ggplot(report, aes(x=Model, y=Kappa.Train))+
    geom_point(col="blue", size=5)+
    labs(x="Model",
         y="Model Kappa", 
       title="Report")
  #7.2.5 plot accuracy and kappa of all models on the same plot
  ggplot(report, aes(x=Model, y=Acc.Test, group=1))+
    geom_point(col="#00009e", size=2)+
    geom_line(col="#00009e", size=1)+
    labs(x="Model",
         y="Model Accuracy or Kappa", 
       title="Model Evaluation")+
    geom_point(aes(x=Model, y=Kappa.Test, group=1), col="#e35125", size=2)+
    geom_line(aes(x=Model, y=Kappa.Test, group=1), col="#e35125", size=1)+
    geom_hline(yintercept = 0.5, color = "red",linetype = "dashed")+ # 0.5 as an indicator of random guessing
    scale_y_continuous(limits=c(0,1))

  

#**********************
# 8.Evaluation
#**********************
algorithemList = c("knn","C5.0","rpart","rf","glmnet","nnet","svmLinear","svmPoly")
models= caretList(response~., data=train, trControl=TControl, methodList=algorithemList,preProcess=mypreProcess)
results <- resamples(models)
summary(results)
dotplot(results)
modelCor(results)
splom(results,col= "blue",cex=1.5, pch=19, bg="white",cex.axis=1.5, cex.lab=1.5, cex.main=1.5, cex.sub=1.5)
balanced_accuracy_test
# plot the balanced accuracy test
ggplot(balanced_accuracy_test) + 
  geom_(aes(x=Model, y=balanced_accuracy_test), stat="identity", fill="blue")+
  labs(x="Model",
       y="Balanced Accuracy Test", 
       title="Balanced Accuracy Test")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#**********************
# Appendix 
#**********************
# check outliers for all the features in the dataset
plot_age = ggplot(data, aes(x=age, color=response,fill=response)) +
    geom_histogram(alpha=0.2,position="identity",binwidth=1)+
    labs(x="Age",y="Count", title="Distribution of Age for different Response")+  
    theme_bw()+
    theme(plot.title = element_text(size=9),axis.text.x= element_text(size=15),axis.text.y= element_text(size=15),
        axis.title=element_text(size=9))

plot_income = ggplot(data, aes(x= income, color=response,fill=response)) +
  geom_histogram(alpha=0.2,position="identity",binwidth=1000)+
labs(x=" Income",
         y="Count", 
       title="Distribution of Income for different Response")+  
theme_bw()+
theme(plot.title = element_text(size=9)
      ,axis.text.x= element_text(size=9),
       axis.text.y= element_text(size=9),
        axis.title=element_text(size=9))

plot_debt_equity = ggplot(data, aes(x=debt_equity, color=response,fill=response)) +
  geom_histogram(alpha=0.2,position="identity",binwidth=1)+
labs(x="Debt to Equity",
         y="Count", 
       title="Distribution of Debt to Equity for different Response")+  
theme_bw()+
theme(plot.title = element_text(size=9)
      ,axis.text.x= element_text(size=9),
       axis.text.y= element_text(size=9),
        axis.title=element_text(size=9))

plot_average_balance_feed_index = ggplot(data, aes(x=average.balance.feed.index, color=response,fill=response)) +
  geom_histogram(alpha=0.2,position="identity",binwidth=200)+
labs(x="Average Balance Feed Index",
         y="Proportion", 
       title="Distribution of Average Balance Feed Index for different Response")+  
theme_bw()+
theme(plot.title = element_text(size=9)
      ,axis.text.x= element_text(size=9),
       axis.text.y= element_text(size=9),
        axis.title=element_text(size=9))

plot_household_debt_to_equity_ratio = ggplot(data, aes(x=household_debt_to_equity_ratio, color=response,fill=response)) +
  geom_histogram(alpha=0.2,position="identity",binwidth=5)+
labs(x="Household debt to equity ratio",
         y="Count", 
       title="Distribution of Household debt to equity ratio for different Response")+  
theme_bw()+
theme(plot.title = element_text(size=9)
      ,axis.text.x= element_text(size=9),
       axis.text.y= element_text(size=9),
        axis.title=element_text(size=9))

plot_non_worker_percentage = ggplot(data, aes(x=non_worker_percentage, color=response,fill=response)) +
  geom_histogram(alpha=0.2,position="identity",binwidth=0.5)+
labs(x="Non Worker percentage",
         y="Count", 
       title="Distribution of Non Worker percentage against response")+  
theme_bw()+
theme(plot.title = element_text(size=9)
      ,axis.text.x= element_text(size=9),
       axis.text.y= element_text(size=9),
        axis.title=element_text(size=9))

plot_white_collar_percentage = ggplot(data, aes(x=white_collar_percentage, color=response,fill=response)) +
  geom_histogram(alpha=0.2,position="identity",binwidth=0.5)+
labs(x="White Collar Percentage",
         y="Count", 
       title="Distribution of White Collar Percentage against response")+  
theme_bw()+
theme(plot.title = element_text(size=9)
      ,axis.text.x= element_text(size=9),
       axis.text.y= element_text(size=9),
        axis.title=element_text(size=9))
grid.arrange(plot_age,plot_income,plot_debt_equity,plot_average_balance_feed_index,plot_household_debt_to_equity_ratio,plot_non_worker_percentage,plot_white_collar_percentage,ncol=2)


summary(train$response)
summary(test$response)  
summary(data_smote$response)
