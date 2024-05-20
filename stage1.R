source('stage1_lasso.R')

arg_Forest<- list(clas_nodesize=1, reg_nodesize=5, ntree=1000, na.action=na.omit, replace=TRUE)
arg_Nnet<- list(size=8,  maxit=1000, decay=0.01, MaxNWts=10000,  trace=FALSE)

get_stage1<-function(Y,T,V,X,p0,D_LB,D_add,max_iter,cfact_val,b,alpha_estimator,gamma_estimator,k = 10){
  
  p=length(b(T[1],V[1],X[1,]))
  n=length(T)
  MNG<-get_MNG(Y,T,V,X,b)
  B=MNG[[4]]
  
  ###########
  # alpha hat
  ###########
  if(alpha_estimator==0){ # dantzig
    
    rho_hat=RMD_stable(Y,T,V,X,p0,D_LB,D_add,max_iter,b,1,0)
    alpha_hat<-function(d,v,z){
      return(b(d,v,z)%*%rho_hat)
    }
    
  } else if(alpha_estimator==1){ # lasso
    
    rho_hat=RMD_stable(Y,T,V,X,p0,D_LB,D_add,max_iter,b,1,1)
    
    alpha_hat<-function(d,v,z){
      return(b(d,v,z)%*%rho_hat)
    }
    
  } else if(alpha_estimator == 2){  # kernel
    py$T_nl = r_to_py(T)
    py$VX_nl = r_to_py(cbind(V, X))
    
    if (missing(cfact_val)){
      py_run_string("model.fit_alpha(np.asarray(T_nl).reshape(-1,1), np.asarray(VX_nl))")
    } else {
      py$cfact_val = r_to_py(cfact_val)
      py_run_string("model.fit_alpha(np.asarray(T_nl).reshape(-1,1), np.asarray(VX_nl), cfact_val)")
    }
    
    alpha_hat <- function(d, v, z){
      py$d_l = r_to_py(d)
      py$vz_l = r_to_py(t(as.matrix(c(v, z))))  # Workaround since z is 1D vector here, not 2D row vector
      py_run_string("alpha_hat_py = model.predict(np.asarray(d_l).reshape(-1,1), np.asarray(vz_l))", convert = TRUE)
      return(py$alpha_hat_py)
    }
  }
  
  ###########
  # gamma hat
  ###########
  if(gamma_estimator==0){ # dantzig
    
    beta_hat=RMD_stable(Y,T,V,X,p0,D_LB,D_add,max_iter,b,0,0)
    gamma_hat<-function(d,v,z){
      return(b(d,v,z)%*%beta_hat)
    }
    
  } else if(gamma_estimator==1){ # lasso
    
    beta_hat=RMD_stable(Y,T,V,X,p0,D_LB,D_add,max_iter,b,0,1)
    gamma_hat<-function(d,v,z){ 
      return(b(d,v,z)%*%beta_hat)
    }
    
  } else if(gamma_estimator==2){ # random forest
    
    forest<- do.call(randomForest, append(list(x=B,y=Y), arg_Forest))
    gamma_hat<-function(d,v,z){
      return(predict(forest,newdata=b(d,v,z), type="response"))
    }
    
  } else if(gamma_estimator==3){ # neural net
    
    # scale down, de-mean, run NN, scale up, remean so that NN works well
    maxs_B <- apply(B, 2, max)
    mins_B <- apply(B, 2, min)
    
    maxs_Y<-max(Y)
    mins_Y<-min(Y)
    
    # hack to ensure that constant covariates do not become NA in the scaling
    const=maxs_B==mins_B
    keep=(1-const)*1:length(const)
    
    NN_B<-B
    NN_B[,keep]<-scale(NN_B[,keep], center = mins_B[keep], scale = maxs_B[keep] - mins_B[keep])
    
    NN_Y<-scale(Y, center = mins_Y, scale = maxs_Y - mins_Y)
    
    nn<- do.call(nnet, append(list(x=NN_B,y=NN_Y), arg_Nnet))
    gamma_hat<-function(d,v,z){
      
      test<-t(as.vector(b(d,v,z)))
      NN_b<-test
      NN_b[,keep]<-scale(t(NN_b[,keep]), 
                         center = mins_B[keep], 
                         scale = maxs_B[keep] - mins_B[keep])
      
      NN_Y_hat<-predict(nn,newdata=NN_b)
      Y_hat=NN_Y_hat*(maxs_Y-mins_Y)+mins_Y
      
      return(Y_hat)
    }
    
  }
  
  return(list(alpha_hat,gamma_hat))
  
}
