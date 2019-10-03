#' @title Train a Split and Merge Deep Neural Network (SM-DNN)
#' @description The function build a Split and Merge Deep Neural Network (SM-DNN) for phenotype prediction
#' @param trainMat  A genotype matrix (N x M; N individuals, M markers) for training model.
#' @param trainPheno  Vector (N * 1) of phenotype for training model.
#' @param validMat A genotype matrix for validing trained model.
#' @param validPheno Pheno Vector (N * 1) of phenotype for validing trained model.
#' @param type (String) CNN can be used as a classification machine or a regression machine. Depending of whether y is a factor or not, the default setting for type is C-classification or eps-regression, respectively.
#' @param subp A constant for splitting the features. It indicates how many features each subset contains after splitting the data e.g. the data originally contains 2000 features. By setting subp=500, the function split the orginal data into 4 subsets with each subset contains 500 features.
#' @param markerImage  (String)  Only for convolutional local networks. This gives a "i * j" image format that the (M x1) markers informations of each individual will be encoded.
#'if the image size exceeds the original snp number, 0 will be polished the lack part,
#' if the image size is less than the original snp number, the last snp(s) will be descaled.
#' @param localFrame  A list containing the following element for local networks.
#' Feedforward neural network (FNN) framework:
#' \itemize{
#'     \item{}{}
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
#'     \item{fullayer_num_hidden:} {A numeric (H * 1) number of hidden neurons for H full connected layers, respectively.
#'     The last full connected layer's number of hidden nerurons must is one.  }
#'     \item{fullayer_act_type:} {A numeric ((H-1) * 1) selecting types of active function from "relu", "sigmoid", "softrelu" and "tanh" for full connected layers.}
#'     \item{drop_float:} {Numeric.}
#' }
#' @param globalFrame A list containing the following element for global networks.
#' \itemize{
#'     \item {fullayer_num_hidden:} {A numeric (H * 1) number of hidden neurons for H full connected layers, respectively.}
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
#' @param wd (Float, 0~1) Weight decay, default 0.
#' @param randomseeds  Set the seed used by mxnet device-specific random number.
#' @param initializer_idx  The initialization scheme for parameters.
#' @param verbose  logical (default=TRUE) Specifies whether to print information on the iterations during training.
#' @param \dots Parameters for construncting neural networks used in package "mxnet" (\url{http://mxnet.io/}).
#'
#' @author Siqi Liang
#' @export
#' @examples




SMDNN <- function(trainMat,trainPheno,validMat,validPheno,type = "eps",subp,markerImage = NULL,localFrame,globalFrame,device_type = "cpu",gpuNum = "max",
                  eval_metric = "mae",num_round = 6000,array_batch_size= 30,learning_rate = 0.01,
                  momentum = 0.5,wd = 0.00001 ,randomseeds = NULL,initializer_idx = 0.01,verbose =TRUE...){

  require('R/Train_localCNN.R')
  #require()

  ###############Train Local Networks##############
  hidden_train <- c()
  hidden_valid  <- c()
  SMDNN_model <- list()
  nn_num = floor(dim(trainMat)[2]/subp)

  for(nn in 1:nn_num){
    #Split the Features
    if(nn != nn_num){
      trainMat_sub <- trainMat[,((nn-1)*subp+1):(nn*subp)]
      validMat_sub <- validMat[,((nn-1)*subp+1):(nn*subp)]
    }else{
      trainMat_sub <- trainMat[,((nn-1)*subp+1):dim(trainMat)[2]]
      validMat_sub <- validMat[,((nn-1)*subp+1):dim(trainMat)[2]]
    }


    if(markerImage){
      localnet <- train_localCNN(trainMat = trainMat_sub, trainPheno = trainPheno,
                                 validMat = validMat_sub, validPheno = validPheno,
                                 type = type, markerImage = markerImage,
                                 cnnFrame = localFrame,device_type = "cpu",gpuNum = 1, eval_metric = "mae",
                                 num_round = 6000,array_batch_size= 30,learning_rate = 0.01,
                                 momentum = 0.5,wd = 0.00001, randomseeds = randomseeds, initializer_idx = 0.01,
                                 verbose = TRUE)

      #Save local networks
      paste0("localnet",nn) <- localnet[1]
      SMDNN_model <- append(SMDNN, paste0("localnet",nn))

      #extract and merge the last hidden layer of local networks
      hidden_train <- rbind(hidden_train, localnet[2])
      hidden_valid <- rbind(hidden_valid, localnet[3])
    } else{
      #####################FNN######################
    }
  }

  ###############Train Global Networks##############
  hidden_train <- t(hidden_train)
  hidden_valid <- t(hidden_valid)
  eval.data <- list(data=hidden_valid, label=validPheno)

  # extract full connect set from the cnn frame list.
  drop_float <- globalFrame$drop_float
  fullayer_num_hidden <- globalFrame$fullayer_num_hidden
  fullayer_act_type <- globalFrame$fullayer_act_type
  if(length(fullayer_num_hidden) - length(fullayer_act_type) != 1){
    stop("Error: the last full connected layer don't need activation.")
  }
  conv_layer_num <- nrow(conv_kernel)
  fullayer_num <- length(fullayer_num_hidden)
  if(fullayer_num_hidden[fullayer_num] != 1){
    stop("Error: the last full connected layer's number of hidden nerurons must is one.")
  }
  if(length(drop_float)- length(fullayer_num_hidden) != 1){
    stop("Error:  the number of dropout layers must one more layer than the full connected  layers.")
  }
  # set full connect frame
  fullayer_num <- length(globalFrame)
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
    outLayer <- mx.symbol.FullyConnected(data= get(paste0("drop_layer",fullayer_num)), num_hidden=1)
    dnn_nerwork <- mx.symbol.LogisticRegressionOutput(outLayer)
  }

  if(!is.null(randomseeds)){mx.set.seed(randomseeds)}
  globalnet <- mx.model.FeedForward.create(dnn_nerwork, X=trainMat, y=trainPheno,eval.data = eval.data,
                                           ctx= device, num.round= num_round, array.batch.size=array_batch_size,
                                           learning.rate=learning_rate, momentum=momentum, wd=wd,
                                           eval.metric= evalfun,initializer = mx.init.uniform(initializer_idx),
                                           verbose = verbose,
                                           epoch.end.callback=mx.callback.early.stop(bad.steps = 600,verbose = verbose))

  SMDNN_model <- apeend(SMDNN_model, globalnet)
  return(SMDNN_model)
}





