L=5

rrr<-function(Y,T,V,X,p0,D_LB,D_add,max_iter,b,alpha_estimator,lam,nu,cfact_val,gamma_estimator,bias,ell){
  
  #* Setup for kernel-based alpha_hat
  if(alpha_estimator == 2){
    #// Must load Python scripts here to work with parallelization
    #// Note: envir = NULL stops Py objects being automatically passed into R
    if (missing(cfact_val)){
      source_python("mean_embedding_model.py", envir = NULL, convert = FALSE)
    } else {
      source_python("mean_embedding_model_alt.py", envir = NULL, convert = FALSE)
    }
    
    #// Instantiate the class, which will cause automatic fitting of an appropriate kernel using all folds
    py$lam = r_to_py(lam)
    py$nu = r_to_py(nu)
    py$Y = r_to_py(Y)
    py$T = r_to_py(T)
    py$VX = r_to_py(cbind(V, X))
    py_run_string("model = MeanEmbeddingModel(np.asarray(T).reshape(-1,1), np.asarray(VX), lam = lam, nu = nu)")
    
    #// If there is more than one choice for lambda, do this to choose lambda using leave-one-out cross-validation
    py_run_string("model.cal_lambda(np.asarray(Y).reshape(-1,1), np.asarray(T).reshape(-1,1), np.asarray(VX))")
  }
  
  ##################
  # sample splitting
  ##################
  
  n=nrow(X)
  folds <- split(sample(n, n,replace=FALSE), as.factor(1:L))
  
  Psi_tilde=numeric(0)
  
  for (l in 1:L){
    
    Y.l=Y[folds[[l]]]
    Y.nl=Y[-folds[[l]]]
    
    T.l=T[folds[[l]]]
    T.nl=T[-folds[[l]]]
    
    V.l=V[folds[[l]]]
    V.nl=V[-folds[[l]]]
    
    X.l=X[folds[[l]],]
    X.nl=X[-folds[[l]],]
    
    n.l=length(T.l)
    n.nl=length(T.nl)
    
    #############
    # get stage 1 (on nl)
    #############
    
    stage1_estimators<-get_stage1(Y.nl,T.nl,V.nl,X.nl,p0,D_LB,D_add,max_iter,cfact_val,b,alpha_estimator,gamma_estimator)
    alpha_hat=stage1_estimators[[1]]
    gamma_hat=stage1_estimators[[2]]
    print(paste0('fold: ',l))
    
    ############
    #get stage 2 (on l)
    ############
    
    Psi_tilde.l=rep(0,n.l)
    for (i in 1:n.l){
      if(bias){ #plug-in
        Psi_tilde.l[i]=psi_tilde_bias(Y.l[i],T.l[i],V.l[i],X.l[i,],m,alpha_hat,gamma_hat,ell) # without subtracting theta_hat
      }else{ #DML
        Psi_tilde.l[i]=psi_tilde(Y.l[i],T.l[i],V.l[i],X.l[i,],m,alpha_hat,gamma_hat,ell) # without subtracting theta_hat
      }
    }
    
    Psi_tilde=c(Psi_tilde,Psi_tilde.l)
    
    #print(paste0('theta_hat: '))
    #print(paste0(round(mean(Psi_tilde.l),2)))
    
  }
  
  ########
  # output
  ########
  
  #point estimation
  ate=mean(Psi_tilde)
  
  #influences
  Psi=Psi_tilde-ate
  
  var=mean(Psi^2)
  se=sqrt(var/n)
  
  out<-c(table(T)[[2]],table(T)[[1]],ate,se)
  
  return(out)
}

#######
# print
#######

printer<-function(out){
  print(paste(" treated: ",out[1], " untreated: ", out[2], "   ATE:    ",round(out[3],2), "   SE:   ", round(out[4],2), sep=""))
}

for_tex<-function(out){
  print(paste(" & ",out[1], " & ", out[2], "   &    ",round(out[3],2), "   &   ", round(out[4],2), sep=""))
}