#' @title Generate Sample Indices for Training Sets and Test Sets
#' @description  This function generates indices for samples in training and test sets for performing the N-fold cross validation experiment.
#' @param sampleNum  The number of samples needed to be partitioned into training and test sets.
#' @param cross  The fold of cross validation.
#' @param seed  An integer used as the seed for data partition. The default value is NULL. When no specific integer is given, no random seed will be set.
#' @author Siqi Liang, Wei-Heng Huang, Faming Liang
#' @return
#' A list and each element including $trainIdx, $testIdx and $cvIdx.
#'
#' $trainIdx  The index of training samples.
#'
#' $testIdx   The index of testing samples.
#'
#' $cvIdx     The index of cross validation.
#' @export
#' @examples
#'#' ## Load example data ##
#' data(wheat_example)
#' ## 5-fold cross validation
#' b <- cvIdx(sampleNum = 2000, fold = 5, seed = 0)

cvIdx <- function( sampleNum, fold = 5, seed = NULL) {
  if(!seed){
    seed <- randomSeed()
  }

  resList <- list()

  # leave-one-out
  if( fold == sampleNum ){
    vec <- 1:sampleNum
    for( i in 1:sampleNum ){
      resList[[i]] <- list( trainIdx = vec[-i], testIdx = i, cvIdx = i)
    }
  }else {
    #random samples
    set.seed(seed)
    index <- sample(1:sampleNum, sampleNum, replace = FALSE )
    step = floor( sampleNum/fold )

    start <- NULL
    end <- NULL
    train_sampleNums <- rep(0, fold)
    for( i in c(1:fold) ) {
      start <- step*(i-1) + 1
      end <- start + step - 1
      if( i == fold )
        end <- sampleNum

      testIdx <- index[start:end]
      trainIdx <- index[-c(start:end)]
      resList[[i]] <- list( trainIdx = trainIdx, testIdx = testIdx, cvIdx = i)
    }
  }
  names(resList) <- paste0("cv",1:fold)
  resList
}
