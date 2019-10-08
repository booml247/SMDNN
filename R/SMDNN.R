#' @title Train a Split and Merge Deep Neural Network (SM-DNN)
#' @description The function build a Split and Merge Deep Neural Network (SM-DNN) for phenotype prediction
#' @param trainMat  A genotype matrix (N x M; N individuals, M markers) for training model.
#' @param trainPheno  Vector (N * 1) of phenotype for training model.
#' @param validMat A genotype matrix for validing trained model.
#' @param validPheno Pheno Vector (N * 1) of phenotype for validing trained model.
#' @param type (String) CNN can be used as a classification machine or a regression machine. Depending of whether y is a factor or not, the default setting for type is C-classification or eps-regression, respectively.
#' @param subp A constant for splitting the features. It indicates how many features each subset contains after splitting the data e.g. the data originally contains 2000 features. By setting subp=500, the function split the orginal data into 4 subsets with each subset contains 500 features.
#' @param localtype  (String)  This parameter indicates what networks you would like to use for local networks. The default setting for type is CNN-convolutional neural network or FNN-Feed-forward Neural Network, respectively.
#' @param localFrame  A list containing the following element for local networks.
#' Feedforward neural network (FNN) framework:
#' \itemize{
#'     \item{fullayer_num_hidden:}{A numeric (H * 1) number of hidden neurons for H full connected layers, respectively.}
#'     \item{fullayer_act_type:}{A numeric ((H-1) * 1) selecting types of active function from "relu", "sigmoid", "softrelu" and "tanh" for full connected layers.}
#'     \item{drop_float:}{A numeric ((H+1) * 1) number setting the dropout rate of each layer in the networks}
#' }
#' Convolutional neural network (CNN) framework:
#' \itemize{
#'     \item{conv_kernel:} {A vector (K * 1) gives convolutional kernel sizes (width x height) to filter image matrix for K convolutional layers, respectively. }
#'     \item{conv_num_filter:} { A vector (K * 1) gives number of convolutional kernels for K convolutional layers, respectively.}
#'     \item{pool_act_type:} {A vector (K * 1) gives types of active function will define outputs of K convolutional layers which will be an input of corresponding pool layer,
#'     respectively. It include "relu", "sigmoid", "softrelu" and "tanh". }
#'     \item{conv_stride:} {A character (K * 1) strides for K convolutional kernel.}
#'     \item{pool_type:} {A character (K * 1) types of K pooling layers select from "avg", "max", "sum", respectively.}
#'     \item{pool_kernel:} {A character (K * 1) K pooling kernel sizes (width * height) for K pooling layers. }
#'     \item{pool_stride:} {A Character (K * 1) strides for K pooling kernels.}
#'     \item{fullayer_num_hidden:} {A numeric (H * 1) number of hidden neurons for H full connected layers, respectively.}
#'     \item{fullayer_act_type:} {A numeric ((H-1) * 1) selecting types of active function from "relu", "sigmoid", "softrelu" and "tanh" for full connected layers.}
#'     \item{drop_float:} {A numeric ((H+1) * 1) number setting the dropout rate of fully connected layer in the networks}
#' }
#' @param globalFrame A list containing the following element for global networks.
#' \itemize{
#'     \item {fullayer_num_hidden:} {A numeric (H * 1) number of hidden neurons for H full connected layers, respectively.}
#'     \item{fullayer_act_type:} {A numeric ((H-1) * 1) selecting types of active function from "relu", "sigmoid", "softrelu" and "tanh" for full connected layers.}
#'     \item{drop_float:} {A numeric ((H+1) * 1) number setting the dropout rate of each layer in the networks}
#' }
#' @param device_type  Selecting "cpu" or "gpu" device to  construct predict model.
#' @param gpuNum  (Integer) Number of GPU devices, if using multiple GPU (gpuNum > 1), the parameter momentum must greater than 0.
#' @param eval_metric (String) A approach for evaluating the performance of training process, it include "mae", "rmse" and "accuracy", default "mae".
#' @param num_round A vector containing the number of iterations over training data to train the local and global model, default = c(6000, 6000).
#' @param  array_batch_size (Integer) It defines number of samples that going to be propagated through the network for each update weight, default 30.
#' @param learning_rate  A vector containing the learning rate for training process of local and global networks, respectively. The default value is c(0.01, 0.01).
#' @param momentum  (Float, 0~1) Momentum for moving average, default 0.5.
#' @param wd A vector containing the weight decay for local and global networks, respectively. Default c(0.00001,0.02).
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
#' testPheno <- y[testIdx].
#' @param randomseeds  Set the seed used by mxnet device-specific random number.
#' @param initializer_idx  The initialization scheme for parameters.
#' @param verbose  logical (default=TRUE) Specifies whether to print information on the iterations during training.
#' @param \dots Parameters for construncting neural networks used in package "mxnet" (\url{http://mxnet.io/}).
#' @author Siqi Liang, Wei-Heng Huang, Faming Liang
#' @export
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
#'
#' # Define local networks' structure (DeepGS stucture)
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
#'                     momentum = 0.5,wd = c(0.00001,0.02) ,randomseeds = NULL,initializer_idx = 0.01,verbose =TRUE)




SMDNN <- function(trainMat,trainPheno,validMat,validPheno,type = "eps",subp,localtype = 'CNN',localFrame,globalFrame,device_type = "cpu",gpuNum = "max",
                  eval_metric = "mae",num_round = c(6000, 6000),array_batch_size= 30,learning_rate = c(0.01, 0.01),
                  momentum = 0.5,wd = c(0.00001, 0.02),randomseeds = NULL,initializer_idx = 0.01,verbose =TRUE...){
  requireNamespace("mxnet")
  require(mxnet)
  source('R/train_localCNN.R')
  source('R/train_localFNN.R')
  # this function is used to evluate metrics provide a way to evaluate the performance of a learned model.
  evalfun <- switch(eval_metric,
                    accuracy = mx.metric.accuracy,
                    mae = mx.metric.mae,
                    rmse = mx.metric.rmse)
  # select device type(cpu/gpu) and device number according you computer and task.
  if(device_type == "cpu") { device <- mx.cpu()}
  if(device_type == "gpu") { ifelse(gpuNum == "max", device <- mx.gpu(),device <- lapply(0:(gpuNum -1), function(i) {  mx.gpu(i)}))}


  ###############Train Local Networks##############
  hidden_train <- c()
  hidden_valid  <- c()
  SMDNN_model <- list()
  nn_num = floor(dim(trainMat)[2]/subp)

  for(nn in 1:nn_num){
    print(paste0("Training Local Network: ", nn))
    #Split the Features
    if(nn != nn_num){
      trainMat_sub <- trainMat[,((nn-1)*subp+1):(nn*subp)]
      validMat_sub <- validMat[,((nn-1)*subp+1):(nn*subp)]
    }else{
      trainMat_sub <- trainMat[,((nn-1)*subp+1):dim(trainMat)[2]]
      validMat_sub <- validMat[,((nn-1)*subp+1):dim(trainMat)[2]]
    }


    if(localtype == 'CNN'){
      markerImage = paste0("1*",ncol(trainMat_sub))
      localnet <- train_localCNN(trainMat = trainMat_sub, trainPheno = trainPheno,
                                 validMat = validMat_sub, validPheno = validPheno,
                                 type = type, markerImage = markerImage,
                                 cnnFrame = localFrame,device_type = device_type,gpuNum = gpuNum, eval_metric = eval_metric,
                                 num_round = num_round[1],array_batch_size= array_batch_size,learning_rate = learning_rate[1],
                                 momentum = momentum,wd = wd[1], randomseeds = randomseeds, initializer_idx = initializer_idx,
                                 verbose = verbose)

      #Save local networks
      assign(paste0("localnet",nn), localnet[1])
      SMDNN_model <- append(SMDNN_model, eval(parse(text =paste0("localnet",nn))))

      #extract and merge the last hidden layer of local networks
      hidden_train <- rbind(hidden_train, as.array(localnet[[2]][[1]]))
      hidden_valid <- rbind(hidden_valid, as.array(localnet[[3]][[1]]))
    } else if(localtype == 'FNN'){
      #####################FNN######################
      localnet <- train_localFNN(trainMat = trainMat_sub, trainPheno = trainPheno,
                                 validMat = validMat_sub, validPheno = validPheno,
                                 type = type,
                                 fnnFrame = localFrame,device_type = device_type,gpuNum = gpuNum, eval_metric = eval_metric,
                                 num_round = num_round[1],array_batch_size= array_batch_size,learning_rate = learning_rate[1],
                                 momentum = momentum,wd = wd[1], randomseeds = randomseeds, initializer_idx = initializer_idx,
                                 verbose = verbose)

      #Save local networks
      assign(paste0("localnet",nn), localnet[1])
      SMDNN_model <- append(SMDNN_model, eval(parse(text =paste0("localnet",nn))))

      #extract and merge the last hidden layer of local networks
      hidden_train <- rbind(hidden_train, as.array(localnet[[2]][[1]]))
      hidden_valid <- rbind(hidden_valid, as.array(localnet[[3]][[1]]))
    } else{
      stop("Error: the local networks must be FNN or CNN.")
    }
  }

  ###############Train Global Networks##############
  hidden_train <- data.matrix(t(hidden_train))
  hidden_valid <- data.matrix(t(hidden_valid))
  eval.data <- list(data=hidden_valid, label=validPheno)

  # extract full connect set from the cnn frame list.
  drop_float <- globalFrame$drop_float
  fullayer_num_hidden <- globalFrame$fullayer_num_hidden
  fullayer_act_type <- globalFrame$fullayer_act_type
  fullayer_num <- length(fullayer_num_hidden)


  if(length(drop_float)- fullayer_num != 1){
    stop("Error:  the number of dropout layers must one more layer than the full connected  layers.")
  }
  # set full connect frame
  data <- mx.symbol.Variable("data")
  for(ss in 1:max(c(fullayer_num -1,1))){
    if(ss == 1){
      assign(paste0("fullconnect_layer",ss),mx.symbol.FullyConnected(data= data, num_hidden= fullayer_num_hidden[ss]))

    } else if(ss > 1){
      assign(paste0("fullconnect_layer",ss),mx.symbol.FullyConnected(data= get(paste0("drop_layer",ss -1)), num_hidden= fullayer_num_hidden[ss]))
    }
    #
    if(fullayer_num == 1){
      assign(paste0("drop_layer",ss),mx.symbol.Dropout(data= get(paste0("fullconnect_layer",ss)), p = drop_float[ss +1]))
    }
    #  performed below when more than more than one full connnect layer
    if(fullayer_num > 1){
      assign(paste0("fullconnect_Act",ss), mx.symbol.Activation(data= get(paste0("fullconnect_layer",ss)), act_type= fullayer_act_type[ss]))
      assign(paste0("drop_layer",ss),mx.symbol.Dropout(data= get(paste0("fullconnect_Act",ss)), p = drop_float[ss +1]))
    }
  }
  # performed below when more than one full connnect layer
  if(fullayer_num > 1){
    assign(paste0("fullconnect_layer",fullayer_num),mx.symbol.FullyConnected(data= get(paste0("drop_layer",ss)), num_hidden= fullayer_num_hidden[fullayer_num]))
    assign(paste0("drop_layer",fullayer_num),mx.symbol.Dropout(data= get(paste0("fullconnect_layer",fullayer_num)), p = drop_float[fullayer_num +1]))
  }

  if(type == "eps"){
    # linear output layer
    dnn_network <- mx.symbol.LinearRegressionOutput(data= get(paste0("drop_layer",fullayer_num)))
  } else{
    # softmax output layer
    dnn_network <- mx.symbol.SoftmaxOutput(data = get(paste0("drop_layer",fullayer_num)))
  }

  if(!is.null(randomseeds)){mx.set.seed(randomseeds)}
  print("Traing Global Network")
  globalnet <- mx.model.FeedForward.create(dnn_network, X=hidden_train, y=trainPheno,eval.data = eval.data,
                                           ctx= device, num.round= num_round[2], array.batch.size=array_batch_size,
                                           learning.rate=learning_rate[2], momentum=momentum, wd=wd[2],
                                           eval.metric= evalfun,initializer = mx.init.uniform(initializer_idx),
                                           verbose = verbose,
                                           epoch.end.callback=mx.callback.early.stop(bad.steps = 600,verbose = verbose))

  SMDNN_model <- append(SMDNN_model, list(globalnet))
  return(SMDNN_model)
}





