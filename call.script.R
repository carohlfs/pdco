
source('Stanton.R')
library(np)

xvec <- runif(20000)
yvec <- 0.5*xvec + 10*xvec^2 + 0.5*runif(20000)

# Racine's optimal bandwidth estimator
system.time(npregbw(ydat = yvec, xdat = xvec, ckertype = 'epanechnikov'))

# my own optimal bandwidth estimator from Stanton.R
system.time(bandwidth(yvec, xvec, 'CrossV'))
