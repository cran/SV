QL <- function(datfile, mu=0.015, psi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
               minlambda=0, maxlambda=2, transf=0, par=NULL, verbose=FALSE, addPenalty=TRUE, nObs=NA,
               checkGradient=FALSE, gradtol=0.01, useRoptimiser=FALSE, updatePars=NULL,
               sandwich=TRUE) {
  ## Declare variables to be output from the .C call

  n.col <- 4

  useParVec <- !is.null(par)
  if (!file.exists(datfile)) {
    stop(paste(datfile, "does not exist"));
  }
  if (useParVec) {
    cat("Input argument 'par' (unrestricted parameters) is used as initial value instead of specified mu, psi, lambda, omega\n");

    npar <- length(par)
    nSup <- (npar-2)/2
    mu <- rep(0.0, n.col) # dummy values
    psi <- rep(0.0, n.col)
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
  H <- rep(0, npar*npar)
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

  if (!useParVec) {
    mu <- c(mu, rep(0.0, n.col-1)) # dummy values
    psi <- c(psi, rep(0.0, n.col-1))
    lambda <- c(lambda, rep(0.0, nSup*(n.col-1)))
    omega <- c(omega, rep(0.0, nSup*(n.col-1)))
  }

  output <- .C("QuasiLikelihood", as.character(datfile), as.integer(nSup),
               as.double(par),
               mu=as.double(mu), psi=as.double(psi), lambda=as.double(lambda), omega=as.double(omega),
               as.double(minlambda), as.double(maxlambda),
               H=as.double(H), nFuncEval=as.integer(nFuncEval), nGradEval=as.integer(nGradEval),
               as.double(gradtol), as.integer(nObs), as.integer(transf),
               as.integer(useParVec),
               as.integer(addPenalty), as.integer(checkGradient),
               as.integer(verbose),
               as.integer(useRoptimiser),
               as.integer(updatePars),
               as.integer(sandwich),
               PACKAGE = "SV")
  obj <- list(mu=output$mu, psi=output$psi, lambda=output$lambda, omega=output$omega,
              H=matrix(output$H, nrow=npar, byrow=TRUE),
              nFuncEval=output$nFuncEval,
              nGradEval=output$nGradEval, datfile=datfile, transf=transf, addPenalty=addPenalty, nObs=nObs, nSup=nSup,
              gradtol=gradtol, useRoptimiser=useRoptimiser, sandwich=sandwich)
  class(obj) <- "ql"
  obj
}

QLmulti <- function(datfile, mu=rep(0.015,2), psi=rep(0.1,3),
                    lambda=rep(c(0.5, 0.05),3),
                    omega=rep(c(0.1, 0.1),3), phi21=0.2,
                    minlambda=c(0,0,0), maxlambda=c(2,2,2), transf=0, par=NULL, verbose=FALSE, addPenalty=TRUE, nObs=NA,
                    checkGradient=FALSE, gradtol=0.01, useRoptimiser=FALSE, updatePars=NULL, sandwich=TRUE) {
  ## Declare variables to be output from the .C call

  if (!file.exists(datfile)) {
    stop(paste(datfile, "does not exist"));
  }

  p <- 1
  q <- 2
  n.col <- 4
  useParVec <- !is.null(par)
  if (useParVec) {
    cat("Input argument 'par' (unrestricted parameters) is used as initial value instead of specified mu, psi, lambda, omega, phi21\n");

    npar <- length(par)
    nSup <- (npar - (p+2*q+1))/(2*(p+q))
    mu <- rep(0.0, q*n.col) # dummy values
    psi <- rep(0.0, (p+q)*n.col)
    lambda <- rep(0.0, nSup*(p+q)*n.col)
    omega <- rep(0.0, nSup*(p+q)*n.col)
    phi21 <- rep(0, n.col)
  }
  else {
    nSup <- length(lambda)/(p+q)
    npar <- q + q+p + 2*(q+p)*nSup+1 #mu, psi, lambda, omega, phi21
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

  
  if (!useParVec) {
    mu <- c(mu, rep(0.0, q*(n.col-1))) # dummy values
    psi <- c(psi, rep(0.0, (p+q)*(n.col-1)))
    lambda <- c(lambda, rep(0.0, nSup*(p+q)*(n.col-1)))
    omega <- c(omega, rep(0.0, nSup*(p+q)*(n.col-1)))
    phi21 <- c(phi21, rep(0, (n.col-1)))
  }
  
  output <- .C("QuasiLikelihoodMulti", as.character(datfile), as.integer(nSup),
               as.double(par),
               mu=as.double(mu), psi=as.double(psi), lambda=as.double(lambda), omega=as.double(omega),
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
               PACKAGE = "SV")
  obj <- list(mu=output$mu, psi=output$psi, lambda=output$lambda, omega=output$omega, phi21=output$phi21,
              H=matrix(output$H, nrow=npar, byrow=TRUE), nFuncEval=output$nFuncEval,
              nGradEval=output$nGradEval, datfile=datfile, transf=transf, addPenalty=addPenalty, nObs=nObs, nSup=nSup,
              gradtol=gradtol, useRoptimiser=useRoptimiser, sandwich=sandwich)
  class(obj) <- "ql"
  obj
}

IndirectInference <- function(datfile, nSim=10, mu=0.015, psi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                              minlambda=0, maxlambda=2, transf=0, par=NULL, print.level=1, addPenalty=TRUE,
                              nObs=NA,
                              checkGradient=FALSE, ftol=0.1, ftol.weak=1, gradtol=1e-4, useRoptimiser=FALSE,
                              initialSteepestDescent=TRUE,
                              test.seed=-117) {
  old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
  on.exit(setRNG(old.seed))

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

  if (is.na(nObs))
    nObs <- -1;
  
  nFuncEval <- 0
  nGradEval <- 0
  nFuncEvalOuter <- 0
  if (is.null(par)) {
    par <- 0
  }

  nSimAll <- 0
  muSim <- rep(0.0, nSim)
  psiSim <- rep(0.0, nSim)
  lambdaSim <- rep(0.0, nSim*nSup)
  omegaSim <- rep(0.0, nSim*nSup)
  error <- 0

  output <- .C("IndirectInference", as.character(datfile), as.integer(nSup), as.integer(nSim),
               as.double(par),
               mu=as.double(mu), psi=as.double(psi), lambda=as.double(lambda), omega=as.double(omega),
               muSim=as.double(muSim), psiSim=as.double(psiSim), lambdaSim=as.double(lambdaSim), omegaSim=as.double(omegaSim),
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
               as.integer(nSimAll),
               error=as.integer(error),
               PACKAGE = "SV")

  if (output$error == 1) {
    cat("Error in indirect inference. No output\n");
    obj <- vector("list", 0)
  }
  else {
    obj <- list(mu=output$mu, psi=output$psi, lambda=output$lambda, omega=output$omega,
                muSim=output$muSim, psiSim=output$psiSim, lambdaSim=output$lambdaSim, omegaSim=output$omegaSim,
                nFuncEval=output$nFuncEval,
                nGradEval=output$nGradEval, nFuncEvalOuter=output$nFuncEvalOuter, datfile=datfile, nSim=nSim,
                nSimAll=output$nSimAll,
                transf=transf, addPenalty=addPenalty, nObs=nObs,
                gradtol=gradtol, useRoptimiser=useRoptimiser, initialSteepestDescent=initialSteepestDescent)
    class(obj) <- "indirect"
  }
  obj
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

CheckContinuity <- function(datfile, par=NULL, mu=0.015, psi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                            minlambda=0, maxlambda=2, transf=0, useParVec=FALSE, verbose=FALSE, addPenalty=TRUE, nObs=NA,
                            checkGradient=FALSE, nEval=100, delta=0.001, ind.par=NA, gradtol=1e-3,
                            useRoptimiser=FALSE, initialSteepestDescent=TRUE, test.seed=-117) {
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
  
  nFuncEval <- 0
  nGradEval <- 0
  if (is.null(par)) {
    par <- 0.0
  }

  if (nSup == 1)
    headers <- c("mu", "lambda", "psi", "omega2")
  else
    headers <- c("mu", paste("lambda_", 1:nSup, sep=""), "psi", paste("omega2_", 1:nSup, sep=""))

  npar <- 2*nSup+2
  if (is.null(ind.par)) {
    ind.par <- rep(1, npar)
  }
  else if (length(ind.par) != npar) {
    stop("ind.par should have same length as the parameter vector")
  }
  headers <- headers[which(ind.par==1)]
  
  nparOut <- sum(ind.par)
  xOut <- rep(0.0, nparOut*nEval);
  xOut.transf <- rep(0.0, nparOut*nEval);
  fOut <- rep(0.0, nparOut*nEval);
  output <- .C("CheckContinuity", as.character(datfile), as.integer(nSup),
               as.double(par), as.integer(ind.par),
               mu=as.double(mu), psi=as.double(psi), lambda=as.double(lambda), omega=as.double(omega),
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
               PACKAGE = "SV")

  xOut <- matrix(output$xOut, ncol=nparOut)
  xOut.transf <- matrix(output$xOut.transf, ncol=nparOut)
  fOut <- matrix(output$fOut, ncol=nparOut)
  ncols <- min(2, nparOut)
  nrows <- ceiling(nparOut/ncols)
  
  par(mfrow=c(nrows, ncols))
  for (i in 1:nparOut) {
    plot(xOut.transf[,i], fOut[,i], type="l", xlab=headers[i], ylab="weighted distance", main=headers[i])
  }
  o <- list(x=xOut, xtr=xOut.transf, f=fOut)
  invisible(o)
}

SimulateVolatility <- function(nSim=10, nTimes, par=NULL, mu=0.015, psi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                               minlambda=0, maxlambda=2, transf=0, useParVec=FALSE, verbose=FALSE) {
  ##  test.seed <- -117
  ##  old.seed <- setRNG(kind = "default", seed = test.seed, normal.kind = "default")
  ##  on.exit(setRNG(old.seed))

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

  sigma2 <- rep(0.0, nTimes)
  logYRet <- rep(0.0, nTimes)

  
  output <- .C("SimulateVolatility", as.integer(nSup), as.integer(nSim), as.integer(nTimes),
               as.double(par),
               mu=as.double(mu), psi=as.double(psi), lambda=as.double(lambda), omega=as.double(omega),
               as.double(minlambda), as.double(maxlambda),
               as.integer(transf),
               as.integer(useParVec),
               as.integer(verbose),
               logYRet=as.double(logYRet),
               sigma2=as.double(sigma2),
               PACKAGE = "SV")

  par(mfrow=c(2,1))
  plot(output$logYRet, type="l", main="Log return")
  plot(output$sigma2, type="l", main="sigma^2")
  obj <- list(logYRet=output$logYRet, sigma2=output$sigma2)
  invisible(obj)
}

SimulateVolatilityMulti <- function(nSim=10, nTimes, par=NULL, mu=rep(0.015,2), psi=rep(0.1,3),
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
               mu=as.double(mu), psi=as.double(psi), lambda=as.double(lambda), omega=as.double(omega),
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
                            nSim=10, nTimes=1000, mu=0.015, psi=0.1, lambda=c(0.5, 0.05), omega=c(0.1, 0.1),
                            minlambda=0, maxlambda=2, transf=0, par=NULL, print.level=1, addPenalty=TRUE,
                            ftol=0.1, ftol.weak=1, gradtol=1e-4, useRoptimiser=FALSE,
                            initialSteepestDescent=TRUE,
                            test.seed=-117) {
  cat("Simulation study\n")

  methods2 <- rep(0, 2)
  names(methods2) <- c("ql", "indirect")
  methods2[methods] <- 1

  nFuncEval <- 0
  nFuncEvalOuter <- 0
  nGradEval <- 0
  error <- 0

  nSup <- length(lambda)
  
  output <- .C("SimulationStudy", as.integer(nRep), as.integer(methods2),
               as.integer(nSup), as.integer(nSim), as.integer(nTimes),
               as.double(par),
               mu=as.double(mu), psi=as.double(psi), lambda=as.double(lambda), omega=as.double(omega),
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
               error=as.integer(error),
               PACKAGE = "SV")
  if (output$error == 1) {
    cat("Error in indirect inference. No output\n");
    obj <- vector("list", 0)
  }
  else {
    obj <- list(mu=output$mu, psi=output$psi, lambda=output$lambda, omega=output$omega, nFuncEval=output$nFuncEval,
                nGradEval=output$nGradEval, nFuncEvalOuter=output$nFuncEvalOuter, nSim=nSim,
                nSimAll=output$nSimAll,
                transf=transf, addPenalty=addPenalty,
                gradtol=gradtol, useRoptimiser=useRoptimiser, initialSteepestDescent=initialSteepestDescent)
  }
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
}

summary.ql <- function(object, ...) {
  cat("Summary of quasi-likelihood object\n")

  SummaryCommon(object)
  
  mat <- MakeTableQL(object)
  cat("Coefficients:\n")
  print(round(mat,4))
  ##  print.ql(object, ...)
}

summary.indirect <- function(object, ...) {
  cat("Summary of indirect inference object\n")
  SummaryCommon(object)
  mat <- MakeTableIndirect(object)
  print(round(mat,4))
  ##  print.indirect(object, ...)
}

MakeTableQL <- function(object) {
  ## mu: est, sd, lower, upper

  multi <- "phi21" %in% names(object)
  
  mu <- matrix(object$mu, ncol=4)
  psi <- matrix(object$psi, ncol=4)
  lambda <- matrix(object$lambda, ncol=4)
  omega <- matrix(object$omega, ncol=4)
  nSup <- object$nSup
  mat <- rbind(mu,psi,lambda,omega)
  if (multi) {
    phi21 <- matrix(object$phi21, ncol=4)
    mat <- rbind(mat,phi21)
  }

  if (multi) {
    numbers <- paste(rep(1:3, each=nSup), rep(1:nSup,3),sep="_")
    nm <- c(paste("mu", 1:nrow(mu), sep="_"),
            paste("psi", 1:nrow(psi), sep="_"),
            paste("lambda", numbers, sep="_"),
            paste("omega", numbers, sep="_"),
            "phi21")
  }
  else {
    nm <- c("mu",
            "psi",
            paste("lambda", 1:nSup, sep="_"),
            paste("omega", 1:nSup, sep="_"))
  }
  dimnames(mat) <- list(nm, c("Estimate", "Std. Error", 
                              "Lower", "Upper"))
  mat
}

MakeTableIndirect <- function(object) {
  mat <- cbind(object$muSim, object$psiSim, object$lambdaSim, object$omegaSim)
  nsup <- length(object$lambda)
  conf <- apply(mat, 2, quantile, prob=c(0.025, 0.975))
  mat.print <- data.frame(Mean=apply(mat, 2, mean),
                          SD=apply(mat, 2, sd),
                          Lower=conf[1,],
                          Upper=conf[2,])
  ##  colnames(mat.print) <- c("Mean", "SD", "Lower", "Upper")
  rownames(mat.print) <- c("mu", "psi", paste(rep(c("lambda_", "omega_"), each=nsup), rep(1:nsup, 2), sep=""))
  mat.print
}
