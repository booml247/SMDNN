#' Obtain the links between SNPs and Genes for Homo Sapiens
#' @description SGLink is a function which finds the links between SNPs and Genes for Homo Sapiens.
#' @usage SGLink(data)
#' @param data: (character) a vector containing the SNPs' ID e.g. "rs11316244"
#' @return a matrix which descirbes the linkage bwtween SNPs and the genes
#' #' @author Siqi Liang, Wei-Heng Huang, Faming Liang
#' @examples
#' data <- c("rs149619941", "rs71509458", "rs11316244")
#' SGLink(data)
#' @export

SGLink <- function(data){
  source('R/SGMap.R')

  #obtain the mappning between SNPs and the corresponding genes
  mapping <- SGMap(data)

  #Construct the link matrix
  SNPs <- unique(mapping[, 1])
  Genes <- unique(na.omit(mapping[, 2]))

  link <- matrix(0, nrow = length(SNPs), ncol = length(Genes))
  for(i in 1:length(SNPs)){
    #find the index of the ith SNP in the mapping
    index_S <- which(mapping[, 1] %in% SNPs[i])
    #obtain the corresponding genes
    index_G <- match(mapping[index_S, 2], Genes)

    link[i, index_G] <- 1
  }
  rownames(link) <- SNPs
  colnames(link) <- Genes
  return(link)
}
