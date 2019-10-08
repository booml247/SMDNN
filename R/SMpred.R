#' @title Make Prediction Using the SM-DNN Model
#' @description The function can make prediction given the test data based on the SM-DNN model.
#' @param testMat A genotype matrix (N x M; N individuals, M markers) for prediction.
#' @param SMDNN_model A list contains local and global networks obtaining from SMDNN function.
#' @param subp A constant for splitting the features. It indicates how many features each subset contains after splitting the data e.g. the data originally contains 2000 features. By setting subp=500, the function split the orginal data into 4 subsets with each subset contains 500 features.
#' @param localtype (String)  This parameter indicates what networks you would like to use for local networks. The default setting for type is CNN-convolutional neural network or FNN-Feed-forward Neural Network, respectively.
#' @author Siqi Liang, Wei-Heng Huang, Faming Liang
#' @examples
#' data(wheat_example)
#' Markers <- wheat_example$Markers
#' y <- wheat_example$y
#' cvSampleList <- cvIdx(length(y),10,1)
#' # cross validation set
#' cvIdx <- 1
#' trainIdx <- cvSampleList[[cvIdx]]$trainIdx
#' testIdx <- cvSampleList[[cvIdx]]$testIdx
#' trainMat <- Markers[trainIdx,]
#' trainPheno <- y[trainIdx]
#' validIdx <- sample(1:length(trainIdx),floor(length(trainIdx)*0.1))
#' validMat <- trainMat[validIdx,]
#' validPheno <- trainPheno[validIdx]
#' trainMat <- trainMat[-validIdx,]
#' trainPheno <- trainPheno[-validIdx]
#' testMat <- Markers[testIdx,]
#' testPheno <- y[testIdx]
#'
#' # Define local networks' structure
#' conv_kernel <- c("1*18") ## convolution kernels (fileter shape)
#' conv_stride <- c("1*1")
#' conv_num_filter <- c(8)  ## number of filters
#' pool_act_type <- c("relu") ## active function for next pool
#' pool_type <- c("max") ## max pooling shape
#' pool_kernel <- c("1*4") ## pooling shape
#' pool_stride <- c("1*4") ## number of pool kernerls
#' fullayer_num_hidden <- c(32,1)
#' fullayer_act_type <- c("sigmoid")
#' drop_float <- c(0.2,0.1,0.05)
#' cnnFrame <- list(conv_kernel =conv_kernel,conv_num_filter = conv_num_filter,
#'                  conv_stride = conv_stride,pool_act_type = pool_act_type,
#'                  pool_type = pool_type,pool_kernel =pool_kernel,
#'                  pool_stride = pool_stride,fullayer_num_hidden= fullayer_num_hidden,
#'                  fullayer_act_type = fullayer_act_type,drop_float = drop_float)
#'
#' # Define global networks' structure
#' fullayer_num_hidden <- c(20,1)
#' fullayer_act_type <- c("sigmoid")
#' drop_float <- c(0,0,0)
#' globalFrame <- list(fullayer_num_hidden= fullayer_num_hidden,
#'                    fullayer_act_type = fullayer_act_type,drop_float = drop_float)
#'
#' # Train a SMDNN model
#' SMDNN_model <- SMDNN(trainMat,trainPheno,validMat,validPheno,type = "eps",subp = 500,localtype = 'CNN',cnnFrame,globalFrame,device_type = "cpu",gpuNum = "max",
#'                     eval_metric = "mae",num_round = c(6000, 6000),array_batch_size= 30,learning_rate = c(0.01, 0.01),
#'                     momentum = 0.5,wd = c(0.00001,0.02),randomseeds = NULL,initializer_idx = 0.01,verbose =TRUE)
#'
#' # Make prediction using the SMDNN model
#' pred_test <- SMpred(SMDNN_model, testMat, subp = 500)


SMpred <- function(SMDNN_model, testMat, subp, localtype = 'CNN'){
  #extract the last hidden layer from the local networks
  hidden_test <- c()
  local_num <- length(SMDNN_model) - 1
  for(i in 1:local_num){
    print(paste0("Retrieving the hidden layer of Local Network: ", i))
    #Split the Features
    if(i != local_num){
      testMat_sub <- testMat[,((i-1)*subp+1):(i*subp)]
    }else{
      testMat_sub <- testMat[,((i-1)*subp+1):dim(testMat)[2]]
    }

    testMat_sub <- data.matrix(t(testMat_sub))
    dim(testMat_sub) <- c(1, nrow(testMat_sub),1,ncol(testMat_sub))


    localnet <- SMDNN_model[[i]]
    if(localtype == 'CNN'){
      #For the case that the local nets are CNN
      internals <- localnet$symbol$get.internals()
      internal_num <- length(internals$outputs)
      fea_symbol <- internals[[internal_num-7]]

      para_num <- length(localnet$arg.params)
      localnet$arg.params[[para_num]] <- NULL
      localnet$arg.params[[para_num-1]] <- NULL

      tmpmodel <- list(symbol = fea_symbol,
                     arg.params = localnet$arg.params,
                     aux.params = localnet$aux.params)
      class(tmpmodel) <- "MXFeedForwardModel"

      hidden <- predict(tmpmodel, testMat_sub)

      hidden_test <- rbind(hidden_test, hidden)
    } else{
      #For the case that the local nets are FNN
      internals <- localnet$symbol$get.internals()
      internal_num <- length(internals$outputs)
      fea_symbol <- internals[[internal_num-7]]

      para_num <- length(localnet$arg.params)
      localnet$arg.params[[para_num]] <- NULL
      localnet$arg.params[[para_num-1]] <- NULL

      tmpmodel <- list(symbol = fea_symbol,
                       arg.params = localnet$arg.params,
                       aux.params = localnet$aux.params)
      class(tmpmodel) <- "MXFeedForwardModel"

      hidden <- predict(tmpmodel, testMat_sub)

      hidden_test <- rbind(hidden_test, hidden)
    }

  }
  hidden_test <- data.matrix(t(hidden_test))

  pred_test <- predict(SMDNN_model[[length(SMDNN_model)]], hidden_test)

  return(pred_test)

}
