QL <- function(datfile, mu=0.015, xi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
               minlambda=0, maxlambda=2, transf=0, par=NULL, verbose=FALSE, addPenalty=TRUE, nObs=NA,
               checkGradient=FALSE, gradtol=0.01, useRoptimiser=FALSE, updatePars=NULL,
               sandwich=TRUE, gradMax=1000^2) {
  ## Declare variables to be output from the .C call
  time.start <- proc.time()

  test.seed <- -117
  old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
  on.exit(setRNG(old.seed))

  n.col <- 4

  useParVec <- !is.null(par)
  if (!file.exists(datfile)) {
    stop(paste(datfile, "does not exist"));
  }
  if (useParVec) {
    cat("Input argument 'par' (unrestricted parameters) is used as initial value instead of specified mu, xi, lambda, omega\n");

    npar <- length(par)
    nSup <- (npar-2)/2
    mu <- rep(0.0, n.col) # dummy values
    xi <- rep(0.0, n.col)
    lambda <- rep(0.0, nSup*n.col)
    omega <- rep(0.0, nSup*n.col)
  }
  else {
    nSup <- length(lambda)
    npar <- 2*nSup + 2
    par <- 0 # dummy value
  }
  if (nSup != 1 && nSup != 2) {
    stop("Number of superposition terms != 1, 2\n")
  }
  Hi <- rep(0, npar*npar)
  HiRob <- rep(0, npar*npar)
  nFuncEval <- integer()
  nGradEval <- integer()

  if (!useParVec) {
    if (nSup != length(omega)) {
      stop("lambda and omega must have same length")
    }
    if (any(diff(lambda)>=0)) {
      stop("lambda must be strictly decreasing")
    }
  }
  if (minlambda >= maxlambda) {
    stop("maxlambda must be larger than minlambda")
  }
  else if (transf == 1 && minlambda == 0) {
    stop("minlambda must be positive for transf==1\n")
  }
  if (is.null(updatePars)) {
    updatePars <- rep(TRUE, npar)
  }
  else if (length(updatePars) != npar) {
    stop("updatePars must have length = number of parameters")
  }

  if (is.na(nObs))
    nObs <- -1; # Missing value transmitted to C++ function

  nFuncEval <- 0
  nGradEval <- 0
  nIter <- 0

  if (!useParVec) {
    mu <- c(mu, rep(0.0, n.col-1)) # dummy values
    xi <- c(xi, rep(0.0, n.col-1))
    lambda <- c(lambda, rep(0.0, nSup*(n.col-1)))
    omega <- c(omega, rep(0.0, nSup*(n.col-1)))
  }

  output <- .C("QuasiLikelihood", as.character(datfile), as.integer(nSup),
               as.double(par),
               mu=as.double(mu), xi=as.double(xi), lambda=as.double(lambda), omega=as.double(omega),
               as.double(minlambda), as.double(maxlambda),
               Hi=as.double(Hi),
               HiRob=as.double(HiRob),
               nFuncEval=as.integer(nFuncEval), nGradEval=as.integer(nGradEval),
               as.double(gradtol), as.integer(nObs), as.integer(transf),
               as.integer(useParVec),
               as.integer(addPenalty), as.integer(checkGradient),
               as.integer(verbose),
               as.integer(useRoptimiser),
               as.integer(updatePars),
               as.integer(sandwich),
               as.double(gradMax),
               nIter=as.integer(nIter),
               PACKAGE = "SV")

  time.end <- proc.time()
  time.spent <- time.end - time.start

  obj <- list(mu=output$mu, xi=output$xi, lambda=output$lambda, omega=output$omega,
              Hi=matrix(output$Hi, nrow=npar, byrow=TRUE),
              HiRob=matrix(output$HiRob, nrow=npar, byrow=TRUE),
              nFuncEval=output$nFuncEval,
              nGradEval=output$nGradEval, datfile=datfile, transf=transf, addPenalty=addPenalty, nObs=nObs, nSup=nSup,
              gradtol=gradtol, useRoptimiser=useRoptimiser, sandwich=sandwich, gradMax=gradMax,
              cputime=time.spent, nIter=output$nIter)

  class(obj) <- "ql"
  obj
}

QLmulti <- function(datfile, mu=rep(0.015,2), xi=rep(0.1,3),
                    lambda=rep(c(0.5, 0.05),3),
                    omega=rep(c(0.1, 0.1),3), phi21=0.2,
                    minlambda=c(0,0,0), maxlambda=c(2,2,2), transf=0, par=NULL, verbose=FALSE, addPenalty=TRUE, nObs=NA,
                    checkGradient=FALSE, gradtol=0.01, useRoptimiser=FALSE, updatePars=NULL, sandwich=TRUE, gradMax=1000^2) {
  ## Declare variables to be output from the .C call
  time.start <- proc.time()

  test.seed <- -117
  old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
  on.exit(setRNG(old.seed))

  if (!file.exists(datfile)) {
    stop(paste(datfile, "does not exist"));
  }

  p <- 1
  q <- 2
  n.col <- 4
  useParVec <- !is.null(par)
  if (useParVec) {
    cat("Input argument 'par' (unrestricted parameters) is used as initial value instead of specified mu, xi, lambda, omega, phi21\n");

    npar <- length(par)
    nSup <- (npar - (p+2*q+1))/(2*(p+q))
    mu <- rep(0.0, q*n.col) # dummy values
    xi <- rep(0.0, (p+q)*n.col)
    lambda <- rep(0.0, nSup*(p+q)*n.col)
    omega <- rep(0.0, nSup*(p+q)*n.col)
    phi21 <- rep(0, n.col)
  }
  else {
    nSup <- length(lambda)/(p+q)
    npar <- q + q+p + 2*(q+p)*nSup+1 #mu, xi, lambda, omega, phi21
    par <- 0 # dummy value
  }
  if (nSup != 1 && nSup != 2) {
    stop("Number of superposition terms != 1, 2\n")
  }
  H <- rep(0, npar*npar)
  nFuncEval <- integer()
  nGradEval <- integer()

  if (!useParVec) {
    lambdamat <- matrix(lambda, nrow=p+q, byrow=TRUE)
    if (length(lambda) != length(omega)) {
      stop("lambda and omega must have same length")
    }
    if (any(apply(lambdamat, 1, function(x) {any(diff(x)>=0)}))) {
      stop("lambda must be strictly decreasing")
    }
  }
  if (any(minlambda >= maxlambda)) {
    stop("maxlambda must be larger than minlambda")
  }
  else if (transf == 1 && any(minlambda == 0)) {
    stop("minlambda must be positive for transf==1\n")
  }
  if (is.null(updatePars)) {
    updatePars <- rep(TRUE, npar)
  }
  else if (length(updatePars) != npar) {
    stop("updatePars must have length = number of parameters")
  }

  if (is.na(nObs))
    nObs <- -1; # Missing value transmitted to C++ function
  
  nFuncEval <- 0
  nGradEval <- 0
  nIter <- 0
  
  if (!useParVec) {
    mu <- c(mu, rep(0.0, q*(n.col-1))) # dummy values
    xi <- c(xi, rep(0.0, (p+q)*(n.col-1)))
    lambda <- c(lambda, rep(0.0, nSup*(p+q)*(n.col-1)))
    omega <- c(omega, rep(0.0, nSup*(p+q)*(n.col-1)))
    phi21 <- c(phi21, rep(0, (n.col-1)))
  }
  
  output <- .C("QuasiLikelihoodMulti", as.character(datfile), as.integer(nSup),
               as.double(par),
               mu=as.double(mu), xi=as.double(xi), lambda=as.double(lambda), omega=as.double(omega),
               phi21=as.double(phi21),
               as.double(minlambda), as.double(maxlambda),
               H=as.double(H), nFuncEval=as.integer(nFuncEval), nGradEval=as.integer(nGradEval),
               as.double(gradtol), as.integer(nObs), as.integer(transf),
               as.integer(useParVec),
               as.integer(addPenalty), as.integer(checkGradient),
               as.integer(verbose),
               as.integer(useRoptimiser),
               as.integer(updatePars),
               as.integer(sandwich),
               as.double(gradMax),
               nIter=as.integer(nIter),
               PACKAGE = "SV")
  
  time.end <- proc.time()
  time.spent <- time.end - time.start
  
  obj <- list(mu=output$mu, xi=output$xi, lambda=output$lambda, omega=output$omega, phi21=output$phi21,
              H=matrix(output$H, nrow=npar, byrow=TRUE), nFuncEval=output$nFuncEval,
              nGradEval=output$nGradEval, datfile=datfile, transf=transf, addPenalty=addPenalty, nObs=nObs, nSup=nSup,
              gradtol=gradtol, useRoptimiser=useRoptimiser, sandwich=sandwich, gradMax=gradMax,
              cputime=time.spent, nIter=output$nIter)
  class(obj) <- "ql"
  obj
}

IndirectInference <- function(datfile, nTimes=NA, mu=0.015, xi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                              minlambda=0, maxlambda=2, transf=0, par=NULL, print.level=1, addPenalty=TRUE,
                              nObs=NA,
                              checkGradient=FALSE, ftol=0.001, ftol.weak=1, gradtol=1e-4, useRoptimiser=FALSE,
                              initialSteepestDescent=TRUE, ITMAX=200,
                              test.seed=-117,
                              useQLestimateAsStartPar=TRUE,
                              simfile="", gradMax=1000^2) {
  scoreCriterium <- TRUE
  optWeightMatrix <- TRUE

  time.start <- proc.time()
  nSim <- 1
  
  old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
  on.exit(setRNG(old.seed))

  if (!file.exists(datfile)) {
    stop(paste(datfile, "does not exist"));
  }

  if (nSim == 1) {
    sandwich <- TRUE
  }
  else {
    sandwich <- FALSE
  }

  if (sandwich)
    n.col <- 4 # 7
  else
    n.col <- 4

  useParVec <- !is.null(par)
  if (useParVec) {
    nSup <- (length(par)-2)/2
  }
  else {
    nSup <- length(lambda)
  }
  if (nSup != 1 && nSup != 2) {
    stop("Number of superposition terms != 1, 2\n")
  }
  H <- double()
  nFuncEval <- integer()
  nGradEval <- integer()

  if (!useParVec) {
    if (nSup != length(omega)) {
      stop("lambda and omega must have same length")
    }
    if (any(diff(lambda)>=0)) {
      stop("lambda must be strictly decreasing")
    }
  }
  if (minlambda >= maxlambda) {
    stop("maxlambda must be larger than minlambda")
  }
  else if (transf == 1 && minlambda == 0) {
    stop("minlambda must be positive for transf==1\n")
  }

  if (!is.na(nObs) && !is.na(nTimes)) {
    if (nObs > nTimes) {
      stop("nTimes must be larger or equal to nObs\n")
    }
  }
  if (is.na(nObs))
    nObs <- -1;
  if (is.na(nTimes))
    nTimes <- -1

  if (nObs > nTimes) {
    stop("nTimes must be larger or equal to nObs\n")
  }
  nFuncEval <- 0
  nGradEval <- 0
  nFuncEvalOuter <- 0
  if (is.null(par)) {
    par <- 0
  }
  if (!useParVec) {
    mu <- c(mu, rep(0.0, n.col-1)) # dummy values
    xi <- c(xi, rep(0.0, n.col-1))
    lambda <- c(lambda, rep(0.0, nSup*(n.col-1)))
    omega <- c(omega, rep(0.0, nSup*(n.col-1)))
  }
  else {
    npar <- length(par)
    nSup <- (npar-2)/2
    mu <- rep(0.0, n.col) # dummy values
    xi <- rep(0.0, n.col)
    lambda <- rep(0.0, nSup*n.col)
    omega <- rep(0.0, nSup*n.col)
  }
    

  nSimAll <- 0
  muSim <- rep(0.0, nSim)
  xiSim <- rep(0.0, nSim)
  lambdaSim <- rep(0.0, nSim*nSup)
  omegaSim <- rep(0.0, nSim*nSup)
  funcval <- rep(0.0, nSim)
  nIter <- rep(0, nSim)
  convergence <- 0
  error <- 0

  output <- .C("IndirectInference", as.character(datfile), as.integer(nSup), as.integer(nSim),
               as.integer(nTimes),
               as.double(par),
               mu=as.double(mu), xi=as.double(xi), lambda=as.double(lambda), omega=as.double(omega),
               muSim=as.double(muSim), xiSim=as.double(xiSim), lambdaSim=as.double(lambdaSim), omegaSim=as.double(omegaSim),
               as.double(minlambda), as.double(maxlambda),
               as.double(H), nFuncEval=as.integer(nFuncEval), nGradEval=as.integer(nGradEval),
               nFuncEvalOuter=as.integer(nFuncEvalOuter),
               as.double(ftol), as.double(ftol.weak), 
               as.double(gradtol), as.integer(nObs), as.integer(transf),
               as.integer(useParVec),
               as.integer(addPenalty), as.integer(checkGradient),
               as.integer(print.level),
               as.integer(useRoptimiser),
               as.integer(initialSteepestDescent),
               as.integer(ITMAX),
               as.integer(nSimAll),
               funcval=as.double(funcval),
               nIter=as.integer(nIter),
               convergence=as.integer(convergence),
               as.character(simfile),
               error=as.integer(error),
               as.integer(useQLestimateAsStartPar),
               as.double(gradMax),
               as.integer(scoreCriterium),
               as.integer(optWeightMatrix),
               PACKAGE = "SV")

  time.end <- proc.time()
  time.spent <- time.end - time.start
  
  if (output$error == 1) {
    cat("Error in indirect inference. No output\n");
    obj <- vector("list", 0)
  }
  else {
    obj <- list(mu=output$mu, xi=output$xi, lambda=output$lambda, omega=output$omega,
                muSim=output$muSim, xiSim=output$xiSim, lambdaSim=output$lambdaSim, omegaSim=output$omegaSim,
                nFuncEval=output$nFuncEval,
                nGradEval=output$nGradEval, nFuncEvalOuter=output$nFuncEvalOuter, datfile=datfile, nSim=nSim,
                nSimAll=output$nSimAll,
                transf=transf, addPenalty=addPenalty, nObs=nObs, nSup=nSup,
                gradtol=gradtol, useRoptimiser=useRoptimiser, initialSteepestDescent=initialSteepestDescent,
                sandwich=sandwich, funcval=output$funcval, nIter=output$nIter,
                convergence=output$convergence,
                useQLestimateAsStartPar=useQLestimateAsStartPar,
                gradMax=gradMax,
                simfile=simfile,
                cputime=time.spent)
    class(obj) <- "indirect"
  }
  obj
}

WriteToFile <- function(ylogret, filename) {
  y <- c(1, cumprod(exp(ylogret/100.0)))
  write(y, filename)
}

test <- function(npar, initialSteepestDescent=TRUE,
                 print.level = 0,
                 test.seed=-117) {
  old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
  on.exit(setRNG(old.seed))


  nFuncEvalOuter <- 0
  par <- rep(0, npar)

  output <- .C("test",
               par=as.double(par),
               as.integer(npar),
               nFuncEvalOuter=as.integer(nFuncEvalOuter),
               as.integer(print.level),
               as.integer(initialSteepestDescent),
               PACKAGE = "SV")

  obj <- list(nFuncEvalOuter=output$nFuncEvalOuter, initialSteepestDescent=initialSteepestDescent,
              estimate=output$par)
  obj
}

CheckContinuity <- function(datfile, nTimes=NA, par=NULL, mu=0.015, xi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                            minlambda=0, maxlambda=2, transf=0, useParVec=FALSE, verbose=FALSE, addPenalty=TRUE, nObs=NA,
                            checkGradient=FALSE, nEval=100, delta=0.001, ind.par=NULL, gradtol=1e-4,
                            useRoptimiser=FALSE, initialSteepestDescent=TRUE, test.seed=-117, scoreCriterium=TRUE, optWeightMatrix=TRUE,
                            profileGradient=TRUE) {
  old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
  on.exit(setRNG(old.seed))

  if (useParVec) {
    nSup <- (length(par)-2)/2
  }
  else {
    nSup <- length(lambda)
  }
  H <- double()
  nFuncEval <- integer()
  nGradEval <- integer()

  if (!useParVec) {
    if (nSup != length(omega)) {
      stop("lambda and omega must have same length")
    }
    if (any(diff(lambda)>=0)) {
      stop("lambda must be strictly decreasing")
    }
  }
  if (minlambda >= maxlambda) {
    stop("maxlambda must be larger than minlambda")
  }
  else if (transf == 1 && minlambda == 0) {
    stop("minlambda must be positive for transf==1\n")
  }

  if (is.na(nObs))
    nObs <- -1;
  if (is.na(nTimes))
    nTimes <- -1

  nFuncEval <- 0
  nGradEval <- 0
  if (is.null(par)) {
    par <- 0.0
  }

  if (nSup == 1)
    headers <- c("mu", "lambda", "xi", "omega2")
  else
    headers <- c("mu", paste("lambda_", 1:nSup, sep=""), "xi", paste("omega2_", 1:nSup, sep=""))

  npar <- 2*nSup+2
  if (is.null(ind.par)) {
    ind.par <- rep(1, npar)
  }
  else if (length(ind.par) != npar) {
    stop("ind.par should have same length as the parameter vector")
  }
  headers <- headers[which(ind.par==1)]

  if (profileGradient)
    headers <- c(headers, "Gradient")
  
  nparOut <- sum(ind.par) + as.integer(profileGradient)
  xOut <- rep(0.0, nparOut*nEval);
  xOut.transf <- rep(0.0, nparOut*nEval);
  fOut <- rep(0.0, nparOut*nEval);
  output <- .C("CheckContinuity", as.character(datfile), as.integer(nSup),
               as.integer(nTimes),
               as.double(par), as.integer(ind.par),
               mu=as.double(mu), xi=as.double(xi), lambda=as.double(lambda), omega=as.double(omega),
               as.double(minlambda), as.double(maxlambda),
               as.double(H), nFuncEval=as.integer(nFuncEval), nGradEval=as.integer(nGradEval),
               as.double(gradtol), as.integer(nObs), as.integer(transf),
               as.integer(useParVec),
               as.integer(addPenalty), as.integer(checkGradient),
               as.integer(verbose),
               as.integer(nEval), as.double(delta), xOut = as.double(xOut),
               xOut.transf= as.double(xOut.transf),
               fOut=as.double(fOut),
               as.integer(useRoptimiser),
               as.integer(initialSteepestDescent),
               as.integer(scoreCriterium),
               as.integer(optWeightMatrix),
               as.integer(profileGradient),
               PACKAGE = "SV")

  xOut <- matrix(output$xOut, ncol=nparOut)
  xOut.transf <- matrix(output$xOut.transf, ncol=nparOut)
  fOut <- matrix(output$fOut, ncol=nparOut)
  ncols <- min(2, nparOut)
  nrows <- ceiling(nparOut/ncols)

  cat("Number of function evaluations:", output$nFuncEval, "\n")
  cat("Number of gradient evaluations:", output$nGradEval, "\n")
  
  par(mfrow=c(nrows, ncols))
  for (i in 1:nparOut) {
    plot(xOut.transf[,i], fOut[,i], type="l", xlab=headers[i], ylab="weighted distance", main=headers[i])
  }
  o <- list(x=xOut, xtr=xOut.transf, f=fOut)
  invisible(o)
}

SimulateVolatility <- function(nSim=1, nTimes, par=NULL, mu=0.0, xi=0.5, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                               minlambda=0, maxlambda=2, transf=0, useParVec=FALSE, verbose=FALSE, test.seed=NULL) {
  if (!is.null(test.seed)) {
    old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
    on.exit(setRNG(old.seed))
  }

  if (useParVec) {
    nSup <- (length(par)-2)/2
  }
  else {
    nSup <- length(lambda)
  }

  if (!useParVec) {
    if (nSup != length(omega)) {
      stop("lambda and omega must have same length")
    }
    if (any(diff(lambda)>=0)) {
      stop("lambda must be strictly decreasing")
    }
  }
  if (minlambda >= maxlambda) {
    stop("maxlambda must be larger than minlambda")
  }
  else if (transf == 1 && minlambda == 0) {
    stop("minlambda must be positive for transf==1\n")
  }
  if (is.null(par)) {
    par <- 0
  }

  sigma2 <- rep(0.0, nSim*nTimes)
  logYRet <- rep(0.0, nSim*nTimes)

  
  output <- .C("SimulateVolatility", as.integer(nSup), as.integer(nSim), as.integer(nTimes),
               as.double(par),
               mu=as.double(mu), xi=as.double(xi), lambda=as.double(lambda), omega=as.double(omega),
               as.double(minlambda), as.double(maxlambda),
               as.integer(transf),
               as.integer(useParVec),
               as.integer(verbose),
               logYRet=as.double(logYRet),
               sigma2=as.double(sigma2),
               PACKAGE = "SV")

  if (nSim == 1)
    par(mfrow=c(2,1))
  else
    par(mfrow=c(nSim, 2))
  
  logYRet <- matrix(output$logYRet, nrow=nSim, byrow=TRUE)
  sigma2 <- matrix(output$sigma2, nrow=nSim, byrow=TRUE)
  for (i in 1:nSim) {
    plot(logYRet[i,], type="l", main="Log return")
    plot(sigma2[i,], type="l", main=expression(sigma^2))
  }
  obj <- list(logYRet=logYRet, sigma2=sigma2)
  invisible(obj)
}

SimulateVolatilityMulti <- function(nSim=1, nTimes, par=NULL, mu=rep(0.015,2), xi=rep(0.1,3),
                                    lambda=rep(c(0.5, 0.05),3),
                                    omega=rep(c(0.1, 0.1),3), phi21=0.2,
                                    minlambda=c(0,0,0), maxlambda=c(2,2,2), transf=0, useParVec=FALSE, verbose=FALSE) {
  ##  test.seed <- -117
  ##  old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
  ##  on.exit(setRNG(old.seed))

  p <- 1
  q <- 2
  if (useParVec) {
    npar <- length(par)
    nSup <- (npar - (p+2*q+1))/(2*(p+q))
  }
  else {
    nSup <- length(lambda)/(p+q)
  }

  if (!useParVec) {
    if (nSup*(p+q) != length(omega)) {
      stop("lambda and omega must have same length")
    }
    lambdamat <- matrix(lambda, nrow=p+q, byrow=TRUE)
    if (any(apply(lambdamat, 1, function(x) {any(diff(x)>=0)}))) {
      stop("lambda must be strictly decreasing")
    }
  }
  if (any(minlambda >= maxlambda)) {
    stop("maxlambda must be larger than minlambda")
  }
  else if (transf == 1 && any(minlambda == 0)) {
    stop("minlambda must be positive for transf==1\n")
  }
  if (is.null(par)) {
    par <- 0
  }

  sigma2 <- rep(0.0, (p+q)*nTimes)
  logYRet <- rep(0.0, q*nTimes)

  
  output <- .C("SimulateVolatilityMulti", as.integer(nSup), as.integer(nSim), as.integer(nTimes),
               as.double(par),
               mu=as.double(mu), xi=as.double(xi), lambda=as.double(lambda), omega=as.double(omega),
               as.double(phi21),
               as.double(minlambda), as.double(maxlambda),
               as.integer(transf),
               as.integer(useParVec),
               as.integer(verbose),
               logYRet=as.double(logYRet),
               sigma2=as.double(sigma2),
               PACKAGE = "SV")

  logYRet <- matrix(output$logYRet, nrow=q, byrow=TRUE)
  sigma2 <- matrix(output$sigma2, nrow=p+q, byrow=TRUE)
  ##  par(mfrow=c(3,2))
  for (k in 1:q) {
    plot(logYRet[k,], type="l", main=paste("Log return data", k))
  }
  for (k in 1:(p+q)) {
    plot(sigma2[k,], type="l", main=paste("sigma^2 prosess", k))
  }
  obj <- list(logYRet=logYRet, sigma2=sigma2)
  invisible(obj)
}

SimulationStudy <- function(nRep, methods=c("ql", "indirect"),
                            nSim=1, nObs=1000, nTimes=1000, mu=0.015, xi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                            minlambda=0, maxlambda=2, transf=0, par=NULL, print.level=1, addPenalty=TRUE,
                            ftol=0.1, ftol.weak=1, gradtol=1e-4, useRoptimiser=FALSE,
                            initialSteepestDescent=TRUE, ITMAX=200, savefile="saveres.txt",
                            test.seed=-117, sandwich=TRUE,
                            writeSimDataToFile=FALSE, gradMax=1000^2,
                            scoreCriterium=TRUE, optWeightMatrix=TRUE) {
  time.start <- proc.time()
  
  old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
  on.exit(setRNG(old.seed))

  if (nObs > nTimes) {
    stop("nTimes must be larger or equal to nObs\n")
  }

  if (nSim == 1) {
    sandwichIndirect <- TRUE
  }
  else {
    sandwichIndirect <- FALSE
  }

  methods2 <- rep(0, 2)
  names(methods2) <- c("ql", "indirect")
  methods2[methods] <- 1

  nFuncEval <- 0
  nFuncEvalOuter <- 0
  nGradEval <- 0
  error <- 0
  
  nSup <- length(lambda)
  nMethods <- sum(methods2)

  savefile2 <- paste(savefile, "_extra", sep="")
  
  if (is.null(par)) {
    par <- 0
    mu <- c(mu, rep(0.0, 2*nRep*nMethods-1)) # dummy values
    xi <- c(xi, rep(0.0, 2*nRep*nMethods-1))
    lambda <- c(lambda, rep(0.0, 2*nRep*nSup*nMethods-nSup))
    omega <- c(omega, rep(0.0, 2*nRep*nSup*nMethods-nSup))
  }
  else {
    mu <- rep(0.0, 2*nRep*nMethods) # dummy values
    xi <- rep(0.0, 2*nRep*nMethods)
    lambda <- rep(0.0, 2*nRep*nSup*nMethods)
    omega <- rep(0.0, 2*nRep*nSup*nMethods)
  }
  covmu <- rep(0.0, nRep*nMethods) # dummy values
  covxi <- rep(0.0, nRep*nMethods)
  covlambda <- rep(0.0, nRep*nSup*nMethods)
  covomega <- rep(0.0, nRep*nSup*nMethods)

  iters <- rep(0.0, nRep*nSim)
  funcval <- rep(0.0, nRep*nSim)
  nSimIndTot <- rep(0, nRep)
  
  output <- .C("SimulationStudy", as.integer(nRep), as.integer(methods2),
               as.integer(nSup), as.integer(nSim), as.integer(nObs), as.integer(nTimes),
               as.double(par),
               mu=as.double(mu), xi=as.double(xi), lambda=as.double(lambda), omega=as.double(omega),
               covmu=as.integer(covmu), covxi=as.integer(covxi), covlambda=as.integer(covlambda), covomega=as.integer(covomega),
               funcval=as.double(funcval), iters=as.integer(iters),
               nSimIndTot=as.integer(nSimIndTot),
               as.double(minlambda), as.double(maxlambda),
               nFuncEval=as.integer(nFuncEval), nGradEval=as.integer(nGradEval),
               nFuncEvalOuter=as.integer(nFuncEvalOuter),
               as.double(ftol), as.double(ftol.weak),
               as.double(gradtol),
               as.integer(transf),
               as.integer(addPenalty),
               as.integer(print.level),
               as.integer(useRoptimiser),
               as.integer(initialSteepestDescent),
               as.integer(ITMAX),
               error=as.integer(error),
               as.character(savefile),
               as.character(savefile2),
               as.integer(sandwich),
               as.integer(sandwichIndirect),
               as.integer(writeSimDataToFile),
               as.double(gradMax),
               as.integer(scoreCriterium),
               as.integer(optWeightMatrix),
               PACKAGE = "SV")
  if (output$error == 1) {
    cat("Error in simulaton study. No output\n");
    obj <- vector("list", 0)
  }
  else {
    time.end <- proc.time()
    time.spent <- time.end - time.start
    
    obj <- list(mu=output$mu, xi=output$xi, lambda=output$lambda, omega=output$omega,
                covmu=output$covmu, covxi=output$covxi, covlambda=output$covlambda, covomega=output$covomega,
                funcval=output$funcval,
                nFuncEval=output$nFuncEval,
                nGradEval=output$nGradEval, nFuncEvalOuter=output$nFuncEvalOuter,
                nSimIndTot=output$nSimIndTot, nSup=nSup, nSim=nSim, nObs=nObs, nTimes=nTimes,
                transf=transf, addPenalty=addPenalty,
                gradtol=gradtol, useRoptimiser=useRoptimiser, initialSteepestDescent=initialSteepestDescent,
                gradMax=gradMax,
                sandwich=sandwich,
                cputime=time.spent,
                savefile=savefile)
  }
  class(obj) <- "simstudy"
  obj
}

print.ql <- function(x, ...) {
  print.default(x, ...)
}

print.indirect <- function(x, ...) {
  print.default(x, ...)
  ##  cat("Number of outer function evaluations: ", x$nFuncEvalOuter, "\n")
}

SummaryCommon <- function(x) {
  cat("Number of function evaluations: ", x$nFuncEval, "\n")
  cat("Number of gradient evaluations: ", x$nGradEval, "\n")
  cat("Cpu time used: ", x$cputime[1], "\n")
}

summary.ql <- function(object, ...) {
  cat("Summary of quasi-likelihood object\n")

  SummaryCommon(object)
  if ("nIter" %in% names(object))
    cat("Number of iterations: ", object$nIter, "\n")
  
  mat <- MakeTableQL(object)
  cat("Coefficients:\n")
  print(round(mat,4))
  ##  print.ql(object, ...)
}

summary.indirect <- function(object, ...) {
  cat("Summary of indirect inference object\n")
  cat("Number of outer function evaluations: ", object$nFuncEvalOuter, "\n")
  SummaryCommon(object)
  if ("nIter" %in% names(object))
    cat("Number of iterations: ", object$nIter, "\n")
  ##  if ("convergence" %in% names(object)) {
  ##    convergence.txt <- c("No", "Weak", "OK")
  ##    cat("Convergence: ", convergence.txt[object$convergence+1], "\n")
  ##  }
    
  if ("funcval" %in% names(object))
    cat("Function value: ", object$funcval, "\n")
  if (object$sandwich) {
    mat <- MakeTableQL(object)
  }
  else {
    mat <- MakeTableIndirect(object)
  }
  print(round(mat,4))
  ##  print.indirect(object, ...)
}

MakeTableQL <- function(object) {
  ## mu: est, sd, lower, upper

  multi <- "phi21" %in% names(object)
  
  mu <- matrix(object$mu, ncol=4)
  xi <- matrix(object$xi, ncol=4)
  lambda <- matrix(object$lambda, ncol=4)
  omega <- matrix(object$omega, ncol=4)
  nSup <- object$nSup
  mat <- rbind(mu,xi,lambda,omega)
  if (multi) {
    phi21 <- matrix(object$phi21, ncol=4)
    mat <- rbind(mat,phi21)
  }

  if (multi) {
    numbers <- paste(rep(1:3, each=nSup), rep(1:nSup,3),sep="_")
    nm <- c(paste("mu", 1:nrow(mu), sep="_"),
            paste("xi", 1:nrow(xi), sep="_"),
            paste("lambda", numbers, sep="_"),
            paste("omega", numbers, sep="_"),
            "phi21")
  }
  else {
    nm <- c("mu",
            "xi",
            paste("lambda", 1:nSup, sep="_"),
            paste("omega", 1:nSup, sep="_"))
  }
  dimnames(mat) <- list(nm, c("Estimate", "Std. Error", 
                              "Lower", "Upper"))
  mat
}

MakeTableIndirect <- function(object) {
  mat <- cbind(object$muSim, object$xiSim, object$lambdaSim, object$omegaSim)
  nsup <- length(object$lambda)
  conf <- apply(mat, 2, quantile, prob=c(0.025, 0.975))
  mat.print <- data.frame(Mean=apply(mat, 2, mean),
                          SD=apply(mat, 2, sd),
                          Lower=conf[1,],
                          Upper=conf[2,])
  ##  colnames(mat.print) <- c("Mean", "SD", "Lower", "Upper")
  rownames(mat.print) <- c("mu", "xi", paste(rep(c("lambda_", "omega_"), each=nsup), rep(1:nsup, 2), sep=""))
  mat.print
}

summary.simstudy <- function(object, ...) {
  cat("Summary of Simulation study object\n")
  cat("Cpu time used: ", object$cputime[1], "\n")
  prefix <- object$savefile
  suffix <- ""
  dat <- ReadSimStudy(object$nSim, object$nSup, prefix=prefix, suffix=suffix, fromTitan=TRUE)
  PrintMCTable(dat)
}

PrintMCTable <- function(dat, lambda1.limit=2.0, fasit=NULL) {
  if (1) {
    ind <- dat[,"lambdaQL_1_est"] <= lambda1.limit
    cat(sum(ind), " lambda_1 estimates <= ", lambda1.limit, "\n")
    cat(sum(!ind), " lambda_1 estimates > ", lambda1.limit, "\n")
    dat <- dat[ind,]
  }
  ind.est <- grep("est", names(dat))
  est.mean <- apply(dat[,ind.est], 2, mean)
  est.median <- apply(dat[,ind.est], 2, median)
  est.sd <- apply(dat[,ind.est], 2, sd)
  est.sem <- est.sd/sqrt(nrow(dat))


  ind.cov <- grep("cov", names(dat))
  est.cov <- apply(dat[,ind.cov], 2, mean)

  ind.sd <- grep("sd", names(dat))
  sd.mean <- sqrt(apply(dat[,ind.sd]^2, 2, mean))
  sd.med <- apply(dat[,ind.sd], 2, median)

  mat <- cbind(est.mean, est.sem, est.sd, sd.mean, sd.med, est.cov)
  rownames(mat) <- unlist(strsplit(rownames(mat), "_est"))

  if (!is.null(fasit)) {
    ind1 <- grep("QL", rownames(mat))
    ind2 <- grep("II", rownames(mat))
    fasit2 <- rep(NA, nrow(mat))
    fasit2[ind1] <- fasit
    fasit2[ind2] <- fasit
    bias <- mat[,"est.mean"] - fasit2
    rmse <- sqrt(bias^2 + mat[,"est.sd"]^2)
    k <- ncol(mat)
    mat <- cbind(true=fasit2, mat, bias, rmse)
  }
  
  round(mat, 4)
}

ReadSimStudy <- function(nSimIndirect, nSup, methods=c("QL", "II"), prefix="../../Titan/Results2/", infix="_dollar_m1", fromTitan=TRUE, suffix="*.res") {
  if (fromTitan) {
    cat("Merge results files from Titan folder\n")
    resfile <- "merged_titan.dat"
    x <- system(paste("cat ", prefix, suffix, " > ", resfile, sep=""), intern=TRUE)
  }
  else {
    cat("Merge results files from Simstudy/R* folders\n")
    resfile <- "merged.dat"
    x <- system(paste("cat ../R?/*", infix, ".res > ", resfile, sep=""), intern=TRUE)
  }
  on.exit(file.remove(resfile))
  
  dat <- read.table(resfile, header=FALSE)

  nMethods <- length(methods)
  nPar <- 2 + 2*nSup

  if (nMethods == 1) {
    N1 <- 2 + 6*nMethods*(1+nSup) + 2*nSimIndirect
    N3 <- 2 + 6*nMethods*(1+nSup) + 2*nSimIndirect + 2*nPar
    if (N1 != ncol(dat) && N3 != ncol(dat)) {
      stop("Number of columns ", ncol(dat), " != ", N1, ", or ", N3)
    }
  }
  else {
    N1 <- 2 + 6*nMethods*(1+nSup) + 2*nSimIndirect
    N2 <- 2 + 6*nMethods*(1+nSup) + 2*nSimIndirect + nSimIndirect*nPar
    N3 <- 2 + 6*nMethods*(1+nSup) + 2*nSimIndirect + nSimIndirect*nPar + 2*nPar
    if (N1 != ncol(dat) && N2 != ncol(dat) && N3 != ncol(dat)) {
      stop("Number of columns ", ncol(dat), " != ", N1, ", ", N2, ", or ", N3)
    }
  }
  nm <- c("It",
          paste(paste("mu", rep(methods,each=2), sep=""), rep(c("est", "sd"), nMethods), sep="_"),
          paste(paste("xi", rep(methods,each=2), sep=""), rep(c("est", "sd"), nMethods), sep="_"),
          paste(paste(paste("lambda", rep(methods, each=2*nSup), sep=""), rep(c(1:nSup), 2*nMethods), sep="_"), rep(c("est", "sd"), each=nSup), sep="_"),
          paste(paste(paste("omega", rep(methods, each=2*nSup), sep=""), rep(c(1:nSup), 2*nMethods), sep="_"), rep(c("est", "sd"), each=nSup), sep="_"),
          paste("covmu", methods, sep=""),
          paste("covxi", methods, sep=""),
          paste(paste(paste("covlambda", rep(methods, each=nSup), sep=""), rep(c(1:nSup), nMethods), sep="_"),  sep="_"),
          paste(paste(paste("covomega", rep(methods, each=nSup), sep=""), rep(c(1:nSup), nMethods), sep="_"),  sep="_"),
          paste("func", c(1:nSimIndirect), sep="_"),
          paste("itersim", c(1:nSimIndirect), sep="_"),
          "nsimall")
  if (N3 == ncol(dat)) {
    nm <- c(nm, paste("SD_un", 1:nPar, sep="_"))
    nm <- c(nm, paste("robSD_un", 1:nPar, sep="_"))
  }
  if (nMethods == 2 && (N2 == ncol(dat) || N3 == ncol(dat))) {
    nm <- c(nm, paste("parsim", 1:nPar*nSimIndirect, sep="_"))
  }
  colnames(dat) <- nm
  
  dat
}
