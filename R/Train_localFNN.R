#' @title Train a local FNN
#' @description The function build a local feedforward neural network in Split and Merge Deep Neural Network (SM-DNN) for nonlinear sufficience dimension reduction.
#' @param trainMat  A genotype matrix (N x M; N individuals, M markers) for training model.
#' @param trainPheno  Vector (N * 1) of phenotype for training model.
#' @param validMat A genotype matrix for validing trained model.
#' @param validPheno Vector (N * 1) of phenotype for validing trained model.
#' @param type (String) FNN can be used as a classification machine or a regression machine. Depending of whether trainPheno is a factor or not, the default setting for type is C-classification or eps-regression, respectively.
#' @param fnnFrame  A list containing the following element for Feed-forward neural network (FNN) framework:
#' \itemize{
#'     \item{fullayer_num_hidden:} {A numeric (H * 1) number of hidden neurons for H full connected layers, respectively.
#'     The last full connected layer's number of hidden nerurons must is one.  }
#'     \item{fullayer_act_type:} {A numeric ((H-1) * 1) selecting types of active function from "relu", "sigmoid", "softrelu" and "tanh" for full connected layers.}
#'     \item{drop_float:} {Numeric.}
#' }
#' @param device_type  Selecting "cpu" or "gpu" device to  construct predict model.
#' @param gpuNum  (Integer) Number of GPU devices, if using multiple GPU (gpuNum > 1), the parameter momentum must greater than 0.
#' @param eval_metric (String) A approach for evaluating the performance of training process, it include "mae", "rmse" and "accuracy", default "mae".
#' @param num_round (Integer) The number of iterations over training data to train the model, default = 10.
#' @param  array_batch_size (Integer) It defines number of samples that going to be propagated through the network for each update weight, default 128.
#' @param learning_rate  The learn rate for training process.
#' @param momentum  (Float, 0~1) Momentum for moving average, default 0.9.
#' @param wd A vector containing the weight decay for local and global networks, respectively. Default c(0.00001,0.02)
#' @param randomseeds  Set the seed used by mxnet device-specific random number.
#' @param initializer_idx  The initialization scheme for parameters.
#' @param verbose  logical (default=TRUE) Specifies whether to print information on the iterations during training.
#' @param \dots Parameters for construncting neural networks used in package "mxnet" (\url{http://mxnet.io/}).
#'
#' @author Siqi Liang
#' @export
#' @examples
#' data(wheat_example)
#' Markers <- wheat_example$Markers
#' y <- wheat_example$y
#' cvSampleList <- cvSampleIndex(length(y),10,1)
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
#' fullayer_num_hidden <- c(32,1)
#' fullayer_act_type <- c("sigmoid")
#' drop_float <- c(0.2,0.1,0.05)
#' fnnFrame <- list(fullayer_num_hidden= fullayer_num_hidden,
#'                  fullayer_act_type = fullayer_act_type,drop_float = drop_float)

#'
#' localmodel <- train_localFNN(trainMat = trainMat,trainPheno = trainPheno,
#'                 validMat = validMat,validPheno = validPheno, type = 'eps', markerImage = markerImage,
#'                 fnnFrame = fnnFrame,device_type = "cpu",gpuNum = 1, eval_metric = "mae",
#'                 num_round = 6000,array_batch_size= 30,learning_rate = 0.01,
#'                 momentum = 0.5,wd = c(0.00001,0.02), randomseeds = 0,initializer_idx = 0.01,
#'                 verbose =TRUE)
#' localFNN <- trainlocalFNN[1]
#' hidden_train <- trainlocalFNN[2]
#' hidden_valid <- trainlocalFNN[3]


train_localFNN <- function(trainMat,trainPheno,validMat,validPheno,type,fnnFrame,device_type = "cpu",gpuNum = "max",
                           eval_metric = "mae",num_round = 6000,array_batch_size= 30,learning_rate = 0.01,
                           momentum = 0.5,wd = c(0.00001,0.02) ,randomseeds = NULL,initializer_idx = 0.01,verbose =TRUE...){
  requireNamespace("mxnet")
  require(mxnet)

  # this function is used to evluate metrics provide a way to evaluate the performance of a learned model.
  evalfun <- switch(eval_metric,
                    accuracy = mx.metric.accuracy,
                    mae = mx.metric.mae,
                    rmse = mx.metric.rmse)
  # select device type(cpu/gpu) and device number according you computer and task.
  if(device_type == "cpu") { device <- mx.cpu()}
  if(device_type == "gpu") { ifelse(gpuNum == "max", device <- mx.gpu(),device <- lapply(0:(gpuNum -1), function(i) {  mx.gpu(i)}))}

  trainMat <- data.matrix(trainMat)
  validMat <- data.matrix(validMat)
  eval.data <- list(data=validMat, label=validPheno)


  # extract full connect set from the fnn frame list.
  drop_float <- fnnFrame$drop_float
  fullayer_num_hidden <- fnnFrame$fullayer_num_hidden
  fullayer_act_type <- fnnFrame$fullayer_act_type
  fullayer_num <- length(fullayer_num_hidden)


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
    fnn_network <- mx.symbol.LinearRegressionOutput(data= get(paste0("drop_layer",fullayer_num)))
  } else{
    # softmax output layer
    fnn_network <- mx.symbol.SoftmaxOutput(data = get(paste0("drop_layer",fullayer_num)))
  }

  # Group some output layers for future analysis
  out <- mx.symbol.Group(c(fullconnect_layer3, fullconnect_Act3, drop_layer3, fnn_network))
  # Create an executor
  executor_train <- mx.simple.bind(symbol=out, data=dim(t(trainMat)), ctx=mx.cpu())
  executor_valid <- mx.simple.bind(symbol=out, data=dim(t(validMat)), ctx=mx.cpu())

  if(!is.null(randomseeds)){mx.set.seed(randomseeds)}
  trainlocalFNN <- mx.model.FeedForward.create(fnn_network, X=trainMat, y=trainPheno, eval.data = eval.data,
                                               ctx= device, num.round= num_round, array.batch.size=array_batch_size,
                                               learning.rate=learning_rate, momentum=momentum, wd=wd,
                                               eval.metric= evalfun,initializer = mx.init.uniform(initializer_idx),
                                               verbose = verbose,
                                               epoch.end.callback=mx.callback.early.stop(bad.steps = 600,verbose = verbose))



  # Update parameters
  mx.exec.update.arg.arrays(executor_train, trainlocalFNN$arg.params, match.name=TRUE)
  mx.exec.update.arg.arrays(executor_valid, trainlocalFNN$arg.params, match.name=TRUE)

  mx.exec.update.aux.arrays(executor_train, trainlocalFNN$aux.params, match.name=TRUE)
  mx.exec.update.aux.arrays(executor_valid, trainlocalFNN$aux.params, match.name=TRUE)


  # Select data to use
  mx.exec.update.arg.arrays(executor_train, list(data=mx.nd.array(t(trainMat))), match.name=TRUE)
  mx.exec.update.arg.arrays(executor_valid, list(data=mx.nd.array(t(validMat))), match.name=TRUE)
  # Do a forward pass with the current parameters and data
  mx.exec.forward(executor_train, is.train=FALSE)
  mx.exec.forward(executor_valid, is.train=FALSE)
  # List of outputs available.
  names(executor_train$ref.outputs)

  hidden_train <-as.array(executor_train$ref.outputs[2])
  hidden_dropout_train <- as.array(executor_train$ref.outputs[3])
  hidden_valid <- as.array(executor_valid$ref.outputs[2])
  hidden_dropout_valid <- as.array(executor_valid$ref.outputs[3])

  res <- list(trainlocalFNN, hidden_train, hidden_valid)
  return(res)
}
