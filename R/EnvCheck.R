## Set CRAN mirrors
options(repos='http://software.rc.fas.harvard.edu/mirrors/R/') # USA
#options(repos='http://cran.ma.imperial.ac.uk/') # UK
#options(repos='http://brieger.esalq.usp.br/CRAN/') # Brazil

## Check whether the packages are installed
necessary <- c('devtools', 'rWikiPathways')
installed <- necessary %in% installed.packages()[, 'Package']
if (length(necessary[!installed]) >=1){
  install.packages(necessary[!installed])
}