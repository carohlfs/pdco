#################################################
# bandwidth_grid.R								#
#												#
# This program determines optimal bandwidth and #
# runs non-parametric regressions.              #
#												#
# Chris Rohlfs, 9/26/2013						#
# Updated with graph, 11/06/2013				#
# Renamed & removed extraneous info, 10/11/2024 #
#################################################

require(data.table)
require(parallel)

#### SUM OF EPANECHNIKOV K((XJ-XI)/H) OVER I ##########
# This operation adds up the Epanechnikov kernel across
# all observations j for a single observation. The point
# of this function is to shrink the xVec before summing
# so that we have a less computationally intensive operation.
epanSum <- function(xj,xVec,bw) {
	ind <- abs(xVec-xj)<=bw
	xVec <- xVec[ind]
	return(sum(0.75*(1-((xj-xVec)/bw)^2)))
	}
	
#### SUM OF Y * EPANECHNIKOV K((XJ-XI)/H) OVER I ##########
# Same as above, but the sum is weighted by the Y-values.
# Note that we have to restrict the yVec first and then
# the xVec, because the yVec restriction requires the original
# xVec.
epanYSum <- function(xj,yVec,xVec,bw) {
	ind <- abs(xVec-xj)<=bw
	yVec <- yVec[ind]
	xVec <- xVec[ind]
	return(sum(yVec*0.75*(1-((xj-xVec)/bw)^2)))	
}

#### BANDWIDTH-SPECIFIC CALCULATOR OF CROSS-VALIDATION ####
########## LEAVE-ONE-OUT LEAST SQUARES CRITERION ##########
# This program computes predicted values for a given
# bandwidth based upon the leave-one-out estimator and
# then evaluates the sum of squared residuals for cross-
# validation to evaluate that bandwidth. This formula
# is taken from Li and Racine (2007, pg. 69)
bwCrossV <- function(yVec,xVec,bandw) {
	# For each i, generate the sum across all j of
	# K((Xj-Xi)/h). To convert this sum into the leave-
	# one-out sum, then subtract off K(0) = 0.75,
	# as is done here.	
	# We do the same for the weighted sum of the y-values,
	# where we subtract off K(0)*Y(i)	
	
	# I originally tried mclapply over the entire dataset,
	# but Gal showed me a way to use data.table
	# that's much faster.
	# crossData <- data.table(data.frame(ID=1:length(xVec), valX = xVec))
	# setkey(crossData,ID)
	crossData <- data.table(ID=1:length(xVec), valX = xVec,key='ID')

	# KxKyx <- crossData[,list(
	# 						Kx=epanSum(valX,xVec,bandw),
	# 						Kyx=epanYSum(valX,yVec,xVec,bandw)
	# 						),by=ID])
	KxKyx <- crossData[,list(
							Kx = unlist(mclapply(valX,epanSum,xVec,bandw,mc.cores=16)),
							Kyx =unlist(mclapply(valX,epanYSum,yVec,xVec,bandw,mc.cores=16))
							)]
	# To make the leave-one-out estimators, we subtract off
	# the value itself for each observation.
	# KxKyx. <- KxKyx[,Kx:=Kx-0.75]
	# KxKyx. <- KxKyx.[,Kyx:=Kyx-0.75*yVec]
	KxKyx <- KxKyx[ , c('Kx','Kyx') := list( Kx-0.75,Kyx-0.75*yVec ) ]
	
	# as in the book, we remove from the sum any cases
	# in which the sum of the weight is zero, so that
	# we'd get infinity.
	# KxKyx <- KxKyx[Kx==0,Kyx:=0]
	# KxKyx <- KxKyx[Kx==0,Kx:=1]	
	KxKyx <- KxKyx[ Kx==0, c('Kx','Kyx') := list(1,0) ]

	
	# return the average squared difference between Y and
	# the predicted values Kyx/Kx
	CrossVSSR <- sum(((yVec -KxKyx$Kyx/KxKyx$Kx)^2)/length(yVec))
	return(CrossVSSR)	
}

### BANDWIDTH SPECIFIC CALCULATOR OF AKAIKE INFO CRITERION ###
# This program is used to implement the AIC approach to estimating
# the optimal bandwidth for kernel regression as recommended by
# Li and Racine (2007, pg 27). While the leave-one-out cross-
# validation technique has been proven to be optimal, Li and Racine
# have shown that the AIC works better than leave-out-out for
# small samples, and for large samples, it's about the same.

# Note that the previous documentation for the Stanton estimator
# used a quick-and-dirty bandwidth selector and a Gaussian kernel.
# But with data-driven bandwidth selection, which is regarded as
# more appropriate, the Gaussian kernel (which is less common
# anyways) requires multiplying every Xi with every Xj, whereas
# the Epanechnikov kernel allows us to multiply just the Xi's and Xj's
# for which abs((Xi-Xj)/h) <= 1.
bwAIC <- function(yVec,xVec,bandw) {
	# the AIC relies upon the trace of the "H" matrix, where
	# the (i,j) element of H is K((Xi-Xj)/h)/sum(K) = K(0)/sum(K),
	# where the sum is from 1 to n. Hence, the trace is:
	# sum(K(0)/sum(K((Xi-Xj)/h))), where the inside sum is over
	# i and the outside sum is over j.
	
	# for an Epanechnikov kernel, the formula for K(u) is:
	# if(|u|<=1) 0.75*(1-u^2) else zero.
	
	# Substituting, we see that K(0) = 0.75.
	
	# We use the Epanechnikov rather than the Gaussian kernel 
	# here because it's more standard and because it requires
	# considerably fewer multiplications, making it a
	# lot faster than Gaussian. Here, we call the epanSum
	# function for each observation to get our estimates.
	# As with Cross-Validation, we use data.table to implement
	# that sequence of operations reasonably quickly.
	# Kx is a vector of the denominators in our trace formula above.
	AICData <- data.table(data.frame(ID=1:length(xVec), valX = xVec))
	setkey(AICData,ID)
	KxKyx <- AICData[,list(
							Kx=epanSum(valX,xVec,bandw),
							Kyx=epanYSum(valX,yVec,xVec,bandw)
							),key=ID]
	
	# now, sum 0.75/Kx across j-values to get a single trace statistic.
	Trace <- sum(0.75/KxKyx$Kx)

	# next, we compute our predicted values for Y given the
	# bandwidth bandw as the ratio of Kyx and Kx..
	yHat <- KxKyx$Kyx/KxKyx$Kx
	
	# take the average squared deviation from the predicted value.
	sigma2 <- sum((yVec -yHat)^2)/length(yVec)
	
	# The Akaike Information Criterion is ln(sigma2) + [1+Trace/n]/[1-(Trace+2)/n]	
	AIC <- log(sigma2) + (1+Trace/length(yVec))/(1-(Trace+2)/length(yVec))

	return(AIC)
}

############# BANDWIDTH SELECTOR FOR KERNEL REGRESSION ################
# select optimal bandwidth using Akaike Info Criterion
# or leave-one-out Cross-Validation as described
# in Li and Racine (2007, pp. 69-72).
# type can be specified as "AIC", "CrossV", or "Quick"
# AIC is best for small samples, CrossV is best for large
# samples, and Quick is best for fast results.
bandwidth <- function(yVar,xVar,type) {

	# select upper and bound to feed into the
	# optimization procedure. Let the lower
	# bound be 10e-8.
	top <- (max(xVar,na.rm=TRUE) -min(xVar,na.rm=TRUE))

	if(type=="AIC") {
		# choose the bw0 that minimizes the Akaike Info Criterion.
		return(optimize(function(input) bwAIC(yVar,xVar,input),lower=10e-8,upper=top,maximum=FALSE)$minimum)
	} else if(type=="CrossV"){
		# choose the bw0 that minimizes the Cross-Validation sum of squares.
		res <- optimize(function(input) bwCrossV(yVar,xVar,input),lower=10e-8,upper=top,maximum=FALSE)$minimum
		return(res)
	} else if(type=="Quick"){
		# quick & dirty bandwidth
		return(4*sd(xVar,na.rm=TRUE)/(length(xVar)^(1/3)))
	}
}

######### FUNCTION TO CALCULATE KERNEL ESTIMATE ###########
# This program performs a nonparametric regression
# of squared spread differences on spread levels.
# The built-in packages for nonparametric regression
# ran too slowly, so we code it here manually.

# In addition to a y and x variable for the regression,
# the user specifies a type (AIC, CrossV, or Quick),
# and a vector xEval that gives the points at which
# yhat is evaluated. The default value for xEval is
# the vector of x-variables specified by the user.
kernReg <- function(yVec,xVec,type,xEval=xVec) {
	# determine the bandwidth based upon
	# user-specified type.
	bandw <- bandwidth(yVec,xVec,type)
	# create the x and y variables for our "evaluation
	# sample," i.e., the values that we actually
	# calculate for our plot.
	EvalData <- data.table(data.frame(ID=1:length(xEval), valX = xEval))
	setkey(EvalData,ID)
	KxKyx <- EvalData[,list(Kx=epanSum(valX,xVec,bandw),
							Kyx=epanYSum(valX,yVec,xVec,bandw)
							),key=ID]
	yEval <- KxKyx$Kyx/KxKyx$Kx
	
	predicted <- data.table(yhat = yEval,x = xEval)
	
	# return a data table containing the predicted
	# values and the corresponding evaluation points
	# for x.
	return(predicted)
}
