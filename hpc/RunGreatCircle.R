#!/usr/bin/RScript

great_circle <- function(args){

    lon1 <- args[1]
    lat1 <- args[2]
    lon2 <- args[3]
    lat2 <- args[4]
    radius <- 3965
    x <- pi / 180.0
    a <- (90.0-lat1)*(x)
    b <- (90.0-lat2)*(x)
    theta <- (lon2-lon1)*(x)
    c <- acos((cos(a)*cos(b)) +
                  (sin(a)*sin(b)*cos(theta)))
    return(radius*c)
  
}

## create the matrix
n <- 1e06
lon1 <- 42
lat1 <- 0.5
lon2 <- -13
lat2 <- -32
data <- rep(c(lon1,lat1,lon2,lat2),n)
mat <- matrix(data,nrow=n,byrow=TRUE)

answer <- apply(mat,1,great_circle)
