######### ---------------rpart Code for decision trees--------------------###########


library(tm)
library(rpart)

#Load the files into corpus
obamaCorpus <- Corpus(DirSource(directory = "D:/R/Chap 6/Speeches/obama" , encoding="UTF-8"))
romneyCorpus <- Corpus(DirSource(directory = "D:/R/Chap 6/Speeches/romney" , encoding="UTF-8"))

#Merge both the corpus to one big corpus
fullCorpus <- c(obamaCorpus,romneyCorpus)#1-22 (obama), 23-44(romney)

#Do basic processing on the loaded corpus
fullCorpus.cleansed <- tm_map(fullCorpus, removePunctuation)
fullCorpus.cleansed <- tm_map(fullCorpus.cleansed, stripWhitespace)
fullCorpus.cleansed <- tm_map(fullCorpus.cleansed, tolower)
fullCorpus.cleansed <- tm_map(fullCorpus.cleansed, removeWords, stopwords("english"))
fullCorpus.cleansed <- tm_map(fullCorpus.cleansed, PlainTextDocument)


#Create the term document matrix for analysis
full.dtm <- DocumentTermMatrix(fullCorpus.cleansed)

#Remove sparse terms
full.dtm.spars <- removeSparseTerms(full.dtm , 0.6)

#Convert the Document term matrix to data frame for easy manipulation
full.matix <- data.matrix(full.dtm.spars)
full.df <- as.data.frame(full.matix)

#Add the speakers name to dataframe
full.df[,"SpeakerName"] <- "obama"
full.df$SpeakerName[21:44] <- "romney"

#Create the training and Test index
train.idx <- sample(nrow(full.df) , ceiling(nrow(full.df)* 0.6))
test.idx <- (1:nrow(full.df))[-train.idx]

# Select the top 70 terms used in the corpus
freqterms70 <- findFreqTerms( full.dtm.spars, 70)

#Creat the formuls for input to rpart function
outcome <- "SpeakerName"
formula_str <- paste(outcome, paste(freqterms70, collapse=" + "), sep=" ~ ")
formula <- as.formula(formula_str)

fit <- rpart(formula, method="class", data=full.df.train,control=rpart.control(minsplit=5, cp=0.001));

print(fit) #display cp table


printcp(fit) # plot cross-validation results

par(mfrow = c(1,2), xpd = NA)
text(fit, use.n=T)






# ----------------------spam classifier -----------------#
#We will set up the environment by loading all the required library
#The data set used in this code can be download from http://www.aueb.gr/users/ion/data/enron-spam/
#The data is Enron-Spam in pre-processed form: Enron1

install.packages("caret")
require(caret)

install.packages("kernlab")
require(kernlab)

install.packages("e1071")
require(e1071)

install.packages("tm")
require(tm)

install.packages("plyr")
require(plyr)

#Extract the Enron-Spam data set to you local file system in my case its in the below directory
pathName <- "D:/R/Chap 6/enron1"

#The data set has two sub folders "spam" folder contains all the mails that are spam ,and "ham" folder which contains all the mails that are legitimate
emailSubDir <- c("ham","spam")
#Build a Term document matrix , Here we are converting the text to quantitative format for analysis.
GenerateTDMForEMailCorpus <- function(subDir , path){
    # concatenate the variable i.e. the path and the sub-folder name to create the complete path to mail corpus directory.
    #mailDir <- sprintf("%s/%s", path, subDir)
    mailDir <-paste(path, subDir, sep="/")
    #create a corpus using the above computed directory path , we will use DirSource since we are dealing with directories , with encoding UTF-8
    
    
    mailCorpus <- Corpus(DirSource(directory = mailDir , encoding="UTF-8"))
    #create term document matrix
    mail.tdm <- TermDocumentMatrix(mailCorpus)
    #remove sparse terms from TDM for better analysis
    mail.tdm <- removeSparseTerms(mail.tdm,0.7)
    #return the results , which is the list of TDM for spam and ham.
    result <- list(name = subDir , tdm = mail.tdm)
}

# Let’s write a function that can convert Term document matrix to data frame

#We will convert the TDM to data frame and append the type of mail if its a spam or ham the data frame.
BindMailTypeToTDM <- function(individualTDM){
    # Create a numeric matrix , get its transpose so that column contains the words and row contains number of word occurrences in the mail
    mailMatrix <- t(data.matrix(individualTDM[["tdm"]]))
    #convert this matrix into data frame since its easy to work with with data frames.
    mailDataFrame <- as.data.frame(mailMatrix , stringASFactors = FALSE)
    # Add the type of mail to each row in the data frame
    mailDataFrame <- cbind(mailDataFrame , rep(individualTDM[["name"]] , nrow(mailDataFrame)))
    #Give a proper name to the last column of the data frame
    colnames(mailDataFrame)[ncol(mailDataFrame)] <- "MailType"
    return (mailDataFrame)
}

tdmList <- lapply(emailSubDir , GenerateTDMForEMailCorpus , path = pathName)
mailDataFrame <- lapply(tdmList, BindMailTypeToTDM)

# join both the data frames for spam and ham
allMailDataFrame <- do.call(rbind.fill , mailDataFrame)
# fill the empty columns with 0
allMailDataFrame[is.na(allMailDataFrame)] <- 0

#reorder the column for readability
allMailDataFrame_ordered <- allMailDataFrame[ ,c(1:18,20:23,19)]

#Prepare a training set , we are getting about 60% of the rows to train the modal
train.idx <- sample(nrow(allMailDataFrame_ordered) , ceiling(nrow(allMailDataFrame_ordered)* 0.6))
# prepare the test set , with the remaining rows that are not part of training sample
test.idx <- (1:nrow(allMailDataFrame_ordered))[-train.idx]

allMailDataFrame.train <- allMailDataFrame_ordered[train.idx,]
allMailDataFrame.test <- allMailDataFrame_ordered[test.idx,]

trainedModel <- naiveBayes(allMailDataFrame.train[,c(1:22)],allMailDataFrame.train[,c(23)], data = allMailDataFrame.train)
prediction <- predict(trainedModel, allMailDataFrame.test)
confusionMatrix <- confusionMatrix(prediction,allMailDataFrame.test[,c(23)])
confusionMatrix




ctable <- as.table(matrix(c(634   , 1, 466 , 450), nrow = 2, byrow = TRUE))
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix")




####KNN--------
#Load library
install.packages("class")
require(class)

install.packages("tm")
require(tm)

install.packages("plyr")
require(plyr)

install.packages("caret")
require(caret)

# to make sure strings are not converted to nominal or categorical variables we set the below option
options(stringAsFactors = FALSE)

#Extract the Speech data set to you local file system in my case its in the below directory
speechDir <- c("romney","obama")
#The data set has two sub folders "obama" folder contains all the speeches from obama ,and "romney" folder which contains all the speeches that are from romney
pathToSpeeches <- "D:/R/Chap 6/Speeches"

#Data cleaning is a essential step when we do analysis on a text data , like Remove numbers ,Strip whitespaces, Remove punctuation , Remove stop words , change to lowercase.

CleanSpeechText <- function(speechText){
    #We will remove all punctuation characters from text
    speechText.cleansed <- tm_map(speechText, removePunctuation)
    #We will remove all white space from text
    speechText.cleansed <- tm_map(speechText, stripWhitespace)
    #We will convert all the words to lower case
    speechText.cleansed <- tm_map(speechText, tolower)
    #We will remove all stop words related to english
    speechText.cleansed <- tm_map(speechText, removeWords, stopwords("english"))
    #return the cleansed text
    return (speechText.cleansed)
}

#We Build a Term document matrix , Here we are converting the text to quantitative format for analysis.
produceTDM <- function(speechFolder,path){
    #Concinate the strings to get the full path to the respective speeches
    speechDirPath <-paste(path, speechFolder, sep="/")
    #Since its a directory use DirSource to create the corpus.
    speechCorpus <- Corpus(DirSource(directory = speechDirPath , encoding="UTF-8"))
    #Clean this corpus to remove unwanted noise in the text to make our analysis better
    speechCorpus.cleansed <- CleanSpeechText(speechCorpus)
    #Build the term document matrix for this cleansed corpus
    speech.tdm <- TermDocumentMatrix(speechCorpus.cleansed)
    #Remove the sparse terms to improve the prediction
    speech.tdm <- removeSparseTerms(speech.tdm,0.6)
    #Return the result of both the speeches as a list of tdm.
    resultTdmList <- list(name = speechFolder , tdm = speech.tdm)
}

#We will add the speakers name to the TDM matrix for training and testing
addSpeakerName <- function(individualTDM){
    # Create a numeric matrix , get its transpose so that column contains the words and row contains number of word occurrences in the speech
    speech.matix <- t(data.matrix(individualTDM[["tdm"]]))
    #convert this matrix into data frame since its easy to work with with data frames.
    seech.df <- as.data.frame(speech.matix)
    # Add the speakers name to each row in the data frame
    seech.df <- cbind(seech.df , rep(individualTDM[["name"]] , nrow(seech.df)))
    #Give a proper name to the last column of the data frame
    colnames(seech.df)[ncol(seech.df)] <- "SpeakerName"
    return (seech.df)
}

tdmList <- lapply(speechDir , produceTDM , path = pathToSpeeches)
speechDfList <- lapply(tdmList, addSpeakerName)

#Join both the data frames for obama and romney
combinedSpeechDf <- do.call(rbind.fill , speechDfList)
# fill the empty columns with 0
combinedSpeechDf[is.na(combinedSpeechDf)] <- 0

#Prepare a training set , we are getting about 60% of the rows to train the modal
train.idx <- sample(nrow(combinedSpeechDf) , ceiling(nrow(combinedSpeechDf)* 0.6))
#Prepare the test set , with the remaining rows that are not part of training sample
test.idx <- (1:nrow(combinedSpeechDf))[-train.idx]

# Lets create a data frame that only has the speaker names of the training set
combinedSpeechDf.speakers <- combinedSpeechDf[,"SpeakerName"]
# Lets create a data frame that all the attributes except the speaker name
combinedSpeechDf.allAttr <- combinedSpeechDf[,!colnames(combinedSpeechDf) %in% "SpeakerName"]

# Lets use the above training set and test set to create inputs to our classifier
combinedSpeechDf.train <- combinedSpeechDf.allAttr[train.idx,]
combinedSpeechDf.test <- combinedSpeechDf.allAttr[test.idx,]
combinedSpeechDf.trainOutcome <- combinedSpeechDf.speakers[train.idx]
combinedSpeechDf.testOutcome <- combinedSpeechDf.speakers[test.idx]

prediction <- knn(combinedSpeechDf.train ,combinedSpeechDf.test ,combinedSpeechDf.trainOutcome)

# Lets check out the confusion matrix
confusionMatrix <- confusionMatrix(prediction,testOutcome)



#### -------------SVM--------

sv <- svm(combinedSpeechDf.train, combinedSpeechDf.trainOutcome)
pred <-predict(sv,combinedSpeechDf.test)
table(pred, combinedSpeechDf.testOutcome)




# MAXENT---
library(maxent)
data <- read.csv(system.file("data/USCongress.csv.gz",package = "maxent"))

library(tm)
corpus <- Corpus(VectorSource(data$text))
dtm <- TermDocumentMatrix(corpus,
control=list(weighting = weightTfIdf,
language = "english",
tolower = TRUE,
stopwords = TRUE,
removeNumbers = TRUE,
removePunctuation = TRUE,
stripWhitespace = TRUE))



max_model <- maxent(matrix_sp[,1:2000],data$major[1:2000],use_sgd = TRUE,
set_heldout = 200)
save.model(max_model, "Model")
max_model <- load.model(“Model")

results <- predict(max_model, matrix_sp[,2001:2400])
model_tune <- tune.maxent(matrix_sp[,1:5000],+ data$major[1:5000],nfold=3, showall=TRUE)


model_tune
optimal_model <- maxent(matrix_sp[,1:2000],data$major[1:2000],l2_regularizer= 0.2, use_sgd = FALSE)


results <- predict(optimal_model, matrix_sp[,2001:2400])



####RTextools-------
#Load obama speech's
obamaCorpus <- Corpus(DirSource(directory = "D:/R/Chap 6/Speeches/obama" , encoding="UTF-8"))

obamaDataFrame<-data.frame(text=unlist(sapply(obamaCorpus, `[`, "content")),stringsAsFactors=F)

obama.df <- cbind(obamaDataFrame , rep("obama" , nrow(obamaDataFrame)))
colnames(obama.df)[ncol(obama.df)] <- "name"

#Load  romney speech's
romneyCorpus <- Corpus(DirSource(directory = "D:/R/Chap 6/Speeches/romney" , encoding="UTF-8"))

romneyDataFrame<-data.frame(text=unlist(sapply(romneyCorpus, `[`, "content")),stringsAsFactors=F)

romney.df <- cbind(romneyDataFrame , rep("romney" , nrow(romneyDataFrame)))
colnames(romney.df)[ncol(romney.df)] <- "name"

# combine both the speech's into on big data frame
speech.df <- rbind(obama.df, romney.df)

speech_matrix <- create_matrix(speech.df["text"], language="english", weighting=tm::weightTfIdf)

speech_container <- create_container(speech_matrix,as.numeric(factor(speech.df$name)),trainSize=1:2000, testSize=2001:3857, virgin=FALSE)

speech_model <- train_model(speech_container,"SVM")

speech_results <- classify_model(speech_container,speech_model)

speech_analytics <- create_analytics(speech_container, speech_results)

speech_score_summary <- create_scoreSummary(speech_container, speech_results)

summary(speech_results)


summary(speech_analytics)



#Let’s try out multiple algorithms on the same data frame
speech_multi_models <- train_models(speech_container, algorithms=c("MAXENT","SVM"))

speech_multi_results <- classify_models(speech_container,speech_multi_models)

speech_multi_analytics <- create_analytics(speech_container, speech_multi_results)

ensemble_summary <- create_ensembleSummary(speech_multi_analytics@document_summary)

precisionRecallSummary <- create_precisionRecallSummary(speech_container, speech_multi_results, b_value = 1)

scoreSummary <- create_scoreSummary(speech_container, speech_multi_results)

recall_acc <- recall_accuracy (speech_multi_analytics@document_summary$MANUAL_CODE,speech_multi_analytics@document_summary$MAXENTROPY_LABEL)

summary(speech_multi_results)


summary(recall_acc)

summary(scoreSummary)

summary(precisionRecallSummary)

# roc


install.packages('ROCR')


library(ROCR)
data(ROCR.simple)


pred <- prediction( ROCR.simple$predictions, ROCR.simple$labels)
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

lines(x=c(0, 1), y=c(0, 1), col="black")




# precision recall

perf1 <- performance(pred, "prec", "rec")
plot(perf1,colorize=TRUE)



perf1 <- performance(pred, "sens", "spec")
plot(perf1,colorize=TRUE)


# k fold cv
install.packages("cvTools")
install.packages("robustbase")
data("coleman")
fit <- lmrob(Y ~ ., data=coleman)

cvFit(fit, data = coleman, y = coleman$Y, cost = rtmspe, K = 5, R = 10, costArgs = list(trim = 0.1), seed = 1234)


install.packages("boot")
cv.glm(data, glmfit, cost, K)

