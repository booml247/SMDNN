#' Map Genes to Pathways
#' @description GPMap is a function which map genes to pathways.
#' @usage GPMap(data, species, systemCode)
#' @param data: (character) A vector containing the official ID specified by a data source or system e.g. "ENSG00000230021".
#' @param systemCode (character) The BridgeDb code associated with the data source or system (default: "En"), e.g., En (Ensembl), L (Entrez), Ch (HMDB), etc. See column two of https://github.com/bridgedb/BridgeDb/blob/master/org.bridgedb.bio/resources/org/bridgedb/bio/datasources.txt.
#' @return A dataframe which contains the genes with the corresponding pathways.
#' @examples
#' data <- c("ENSG00000153574", "ENSG00000232810", "ENSG00000230023")
#' GPMap(data, 'En')
#' @export


GPMap <- function(data, systemCode = 'En'){
  #check whether "rWikiPathways" is installed
  if (!requireNamespace("BiocManager", quietly=TRUE)){
    install.packages("BiocManager")
    BiocManager::install("rWikiPathways")
  }

  library(rWikiPathways)

  #map genes to pathways
  res <- c()

  for(i in 1:length(data)){
    pathway <- findPathwayIdsByXref(data[i], systemCode = systemCode)
    if(length(pathway) != 0){
      for(j in 1:length(pathway)){
        res <- append(res, c(data[i], pathway[j]))
      }
    } else{
      res <- append(res, c(data[i], NA))
    }
  }
  res  <- as.data.frame(matrix(res, ncol = 2, byrow = T))
  names(res) <- c("Gene","Pathway")
  return(res)
}
