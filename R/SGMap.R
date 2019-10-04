#' Map SNPs to Genes for Homo Sapiens
#' @description SGMap is a function which map SNPs to Genes.
#' @usage SGMap(data)
#' @param data: a vector containing the SNPs' ID e.g. "rs11316244"
#' @return a dataframe which contains the SNPs with the corresponding genes
#' @examples
#' data <- c("rs149619941", "rs71509458")
#' SGMap(data)
#' @export



SGMap <- function(data){
  #import the map dataset
  gene <- read.table('data/SNP-VEP_res.txt',header = T)
  var <- as.character(data)

  res <- c()

  for(i in 1:length(var)){
    #find corresponding genes
    df <- gene[which(gene[,1]==var[i]),]
    gen <- unique(df[,7])
    gen <- as.character(gen)
    gen <- gen[which(gen != "-")]

    #append to the list
    if(length(gen)==0) {
      res <- append(res, c(var[i], NA))
    } else{
      for(j in 1:length(gen)){
        res <- append(res, c(var[i], gen[j]))
      }
    }
  }
  res <- as.data.frame(matrix(res, ncol = 2, byrow = T))
  names(res) <- c("SNP","Gene")
  return(res)
}


