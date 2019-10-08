library(Metrics)
library(readr)
source("R/cvIdx.R")
source("R/SMDNN.R")
source("R/SMpred.R")
Markers <- read.csv("D:/Git/data_snp/train.csv")
Markers_val <- read.csv("D:/Git/data_snp/validation.csv")
Markers <- rbind(Markers, Markers_val)
y <- Markers[,dim(Markers)[2]]
Markers <- Markers[,1:(dim(Markers)[2]-1)]
cvSampleList <- cvIdx(length(y),10,1)
# cross validation set
for(i in 1:10){
  cv <- i
  trainIdx <- cvSampleList[[cv]]$trainIdx
  testIdx <- cvSampleList[[cv]]$testIdx
  trainMat <- Markers[trainIdx,]
  trainPheno <- y[trainIdx]
  validIdx <- sample(1:length(trainIdx),floor(length(trainIdx)*0.1))
  validMat <- trainMat[validIdx,]
  validPheno <- trainPheno[validIdx]
  trainMat <- trainMat[-validIdx,]
  trainPheno <- trainPheno[-validIdx]
  testMat <- Markers[testIdx,]
  testPheno <- y[testIdx]
  #'
  # Define local networks' structure
  fullayer_num_hidden <- c(1000,500,100,2)
  fullayer_act_type <- c("tanh", "tanh", "tanh" )
  drop_float <- c(0, 0, 0, 0, 0)
  fnnFrame <- list(fullayer_num_hidden= fullayer_num_hidden,
                   fullayer_act_type = fullayer_act_type,drop_float = drop_float)
  #'
  # Define global networks' structure
  fullayer_num_hidden <- c(20,1)
  fullayer_act_type <- c("tanh")
  drop_float <- c(0, 0, 0)
  globalFrame <- list(fullayer_num_hidden= fullayer_num_hidden,
                      fullayer_act_type = fullayer_act_type,drop_float = drop_float)
  #'
  # Train a SMDNN model
  start_time <- proc.time()

  SMDNN_model <- SMDNN(trainMat,trainPheno,validMat,validPheno,type = "C",subp = 8000,localtype = 'FNN',fnnFrame,globalFrame,device_type = "cpu",gpuNum = "max",
                       eval_metric = "accuracy",num_round = c(10000,10000),array_batch_size= 50,learning_rate = c(0.0001, 0.0001),
                       momentum = 0.5,wd = c(0.00003, 0.02) ,randomseeds = i,initializer_idx = 0.01,verbose =TRUE)

  time_elapse <- proc.time() - start_time
  # Make prediction using the SMDNN model
  pred_train <- SMpred(SMDNN_model, trainMat, subp = 8000)
  pred_test <- SMpred(SMDNN_model, testMat, subp = 8000)

  acc_train <- accuracy(pred_train, trainPheno)
  acc_test <- accuracy(pred_test, testPheno)

  auc_train <- auc(trainPheno, pred_train)
  auc_test <- auc(testPheno, pred_test)

  res <- data.frame(acc_train = acc_train, acc_test = acc_test, auc_train = auc_train, auc_test = auc_test, time_elapse = time_elapse[3])

  write_csv(res, "D:/Git/data_snp/EOMI_res/result.csv", append = TRUE)
  write.table(pred_train, paste0("D:/Git/data_snp/EOMI_res/pred_train",i, ".txt"))
  write.table(pred_test, paste0("D:/Git/data_snp/EOMI_res/pred_test",i, ".txt"))
}
