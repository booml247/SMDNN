#' Obtain the links between SNPs and Genes for Homo Sapiens
#' @description GPLink is a function which finds the links between SNPs and Genes for Homo Sapiens.
#' @usage GPLink(data)
#' @param data: a vector containing the SNPs' ID e.g. "rs11316244"
#' @param systemCode (character) The BridgeDb code associated with the data source or system (default: "En"), e.g., En (Ensembl), L (Entrez), Ch (HMDB), etc. See column two of https://github.com/bridgedb/BridgeDb/blob/master/org.bridgedb.bio/resources/org/bridgedb/bio/datasources.txt.
#' @return a matrix which descirbes the linkage bwtween SNPs and the genes
#' @examples
#' data <- c("ENSG00000153574", "ENSG00000232810", "ENSG00000230023")
#' GPLink(data, 'En')
#' @export


GPLink <- function(data, systemCode = 'En'){
  source('R/GPMap.R')

  #obtain the mappning between SNPs and the corresponding genes
  mapping <- GPMap(data, systemCode)

  #Construct the link matrix
  Genes <- unique(mapping[, 1])
  Paths <- unique(na.omit(mapping[, 2]))

  link <- matrix(0, nrow = length(Genes), ncol = length(Paths))
  for(i in 1:length(Genes)){
    #find the index of the ith SNP in the mapping
    index_G <- which(mapping[, 1] %in% Genes[i])
    #obtain the corresponding genes
    index_P <- match(mapping[index_G, 2], Paths)

    link[i, index_P] <- 1
  }
  rownames(link) <- Genes
  colnames(link) <- Paths
  return(link)
}
