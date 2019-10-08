library(Metrics)
library(readr)
source("R/cvIdx.R")
source("R/SMDNN.R")
source("R/SMpred.R")
Markers <- read.csv("D:/Git/data_snp/data/GS.csv")
y <- read.csv("D:/Git/data_snp/data/pheno.csv")
Markers <- Markers[,2:dim(Markers)[2]]
y <- y[,4]
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
  conv_kernel <- c("1*18") ## convolution kernels (fileter shape)
  conv_stride <- c("1*1")
  conv_num_filter <- c(8)  ## number of filters
  pool_act_type <- c("relu") ## active function for next pool
  pool_type <- c("max") ## max pooling shape
  pool_kernel <- c("1*4") ## pooling shape
  pool_stride <- c("1*4") ## number of pool kernerls
  fullayer_num_hidden <- c(32,1)
  fullayer_act_type <- c("sigmoid")
  drop_float <- c(0.2,0.1,0.05)
  cnnFrame <- list(conv_kernel =conv_kernel,conv_num_filter = conv_num_filter,
                   conv_stride = conv_stride,pool_act_type = pool_act_type,
                   pool_type = pool_type,pool_kernel =pool_kernel,
                   pool_stride = pool_stride,fullayer_num_hidden= fullayer_num_hidden,
                   fullayer_act_type = fullayer_act_type,drop_float = drop_float)
  #'
  # Define global networks' structure
  fullayer_num_hidden <- c(500,1)
  fullayer_act_type <- c("tanh")
  drop_float <- c(0,0,0)
  globalFrame <- list(fullayer_num_hidden= fullayer_num_hidden,
                      fullayer_act_type = fullayer_act_type,drop_float = drop_float)
  #'
  # Train a SMDNN model
  start_time <- proc.time()

  SMDNN_model <- SMDNN(trainMat,trainPheno,validMat,validPheno,type = "eps",subp = 2000,localtype = 'CNN',cnnFrame,globalFrame,device_type = "cpu",gpuNum = "max",
                       eval_metric = "mae",num_round = c(6000,10000),array_batch_size= 30,learning_rate = c(0.01, 0.01),
                       momentum = 0.5,wd = c(0.00001, 0.02) ,randomseeds = i,initializer_idx = 0.01,verbose =TRUE)

  time_elapse <- proc.time() - start_time
  # Make prediction using the SMDNN model
  pred_train <- SMpred(SMDNN_model, trainMat, subp = 2000)
  pred_test <- SMpred(SMDNN_model, testMat, subp = 2000)

  corr_train <- cor(t(rbind(pred_train, trainPheno)))[1,2]
  corr_test <- cor(t(rbind(pred_test, testPheno)))[1,2]

  MSE_train <- mse(pred_train, trainPheno)
  MSE_test <- mse(pred_test, testPheno)

  res <- data.frame(corr_train = corr_train, corr_test = corr_test, MSE_train = MSE_train, MSE_test = MSE_test, time_elapse = time_elapse[3])

  write_csv(res, "D:/Git/data_snp/data/result.csv", append = TRUE)
  write.table(pred_train, paste0("D:/Git/data_snp/data/pred_train",i, ".txt"))
  write.table(pred_test, paste0("D:/Git/data_snp/data/pred_test",i, ".txt"))
}
