coverage_experiment = function(data,n,n_iter = 100,v_vals,ch_vals,report_bias = FALSE,CI_vals, alpha_estimator = 0, lam, nu, cfact_val, gamma_estimator = 0, b=b2, hist=FALSE){
  ###Inputs
  #data: either a data generating function that takes as input n or list(Y,T,X)
  #n: dimension of X  if data is a function(X is n x n) (irrelevant is data is a list)
  #n_iter: number of iterations each coverage experiment is run over
  #v_vals: list of v values to iterate over
  #ch_vals: list of bandwidth hyperparameters to try
  #report_bias: indicates whether or not to run a trial of PCR with biasing
  #CI_vals: list of confidence intervals to test for (as percents, so 95% CI = 95)
  #alpha_estimator: alpha for PCR (number 2-5)
  #gamma_estimator: gamma for PCR (number 4-7)
  #b:dictionary
  #hist: TRUE for output to include a vector theta_hat of all estimates for a histogram
  
  ###Outputs
  #results: data frame where each row gives data for 1 coverage experiment
  
  ###The code does the following:
  ###run a coverage experiment for each combination of v, ch, and biases:
  ### For n_iter rounds, randomly draw data (if data is a function)
  ###   Run local DML with alpha_estimator and gamma_estimator as your alpha and gamma estimators
  ###   Record from the trial 1.CATE 2.SE 3.which confidence intervals had target in them
  ###Each row in the output corresponds to 1 combination of v, ch, and biases
  ###As 1 row in the data frame, record:
  ###   1. the CATE averaged over all trials
  ###   2. the SE average over all trials
  ###   3. the coverage of each confidence interval in CI_vals
  
  ######
  # data
  ######
  
  #checks if data is a function or list
  if (length(data)!=1){
    Y = data[[1]]
    T = data[[2]]
    V = data[[3]]
    X = data[[4]]
  } else {
    DF = data(n)
    Y = DF[[1]]
    T = DF[[2]]
    V = DF[[3]]
    X = DF[[4]]
  }
  
  #################
  # hyperparameters (Dantzig and Lasso)
  #################
  
  p=length(b1(T[1],V[1],X[1,]))
  p0=ceiling(p/4) 
  if (p>60){
    p0=ceiling(p/40)
  }
  D_LB=0 #each diagonal entry of \hat{D} lower bounded by D_LB
  D_add=.2 #each diagonal entry of \hat{D} increased by D_add. 0.1 for 0, 0,.2 otw
  max_iter=10 #max number iterations in Dantzig selector iteration over estimation and weights

  ############
  # initialize
  ############
  
  params = c()
  targets= c()
  algs = c()
  biases = c()
  ATEs = c()
  SEs = c()
  CIs = matrix(0,length(CI_vals),0)
  
  #num_of_specs is the number of different variations for each v value
  num_of_specs = length(ch_vals)*(report_bias+1)
  
  # for export when hist=TRUE
  ATE_mat_out = matrix(0,n_iter,num_of_specs)
  
  ######
  # loop - v values (param)
  ######
  
  #first loop; iterates over v values
  for (param in v_vals){
    
    #Uses to keep track of ATE, SE, and how ofter CI contains target
    CI_mat = matrix(0,length(CI_vals),num_of_specs)
    ATE_mat = matrix(0,n_iter,num_of_specs)
    SE_mat = matrix(0,n_iter,num_of_specs)
    
    ######
    # loop - experiment iteration (iter)
    ######
    
    for (iter in 1:n_iter){
      
      print(paste(param,iter))
      
      #generate clean data (if simulated)
      if (length(data)==1){
        DF = data(n)
        Y = DF[[1]]
        T = DF[[2]]
        V = DF[[3]]
        X = DF[[4]]
        CATE=DF[[6]]
      }
      
      ######
      # loop - bandwidth hyperparameter values (j)
      ######
      
      #loop over different ch_vals
      for (j in 1:length(ch_vals)){
        ch_val = ch_vals[j]
        
        h=ch_val*sd(V)*n^(-0.2)
        denom=mean(dnorm((V-param)/h))
          
        ell=function(v){
          num=dnorm((v-param)/h)
          out=num/denom
          return(out)
        }
        
        ##############
        # debiased PCR
        ##############
        
        unbiased_out<-rrr(Y,T,V,X,p0,D_LB,D_add,max_iter,b,alpha_estimator,lam,nu,cfact_val,gamma_estimator,0,ell)
        printer(unbiased_out)
        
        ATE_mat[iter,j] = unbiased_out[3]
        SE_mat[iter,j] = unbiased_out[4]
        
        ######
        # loop - significance level (a)
        ######
        
        target=CATE(param)
        
        for (a in 1:length(CI_vals)){
          #test whether CI contains the target value |theta_hat-theta_0| \leq q_{a/2} se(theta_hat)
          if (abs(unbiased_out[3]-target)<(qnorm(0.5*CI_vals[a]/100+0.5)*unbiased_out[4])){
            #CI[1] = CI[1]+1
            CI_mat[a,j] = CI_mat[a,j]+1
          }
        }
        
        ############
        # biased PCR
        ############
        
        if (report_bias){
          
          biased_out<-rrr(Y,T,V,X,p0,D_LB,D_add,max_iter,b,alpha_estimator,lam,nu,cfact_val,gamma_estimator,1,ell)
          printer(biased_out)
          
          ATE_mat[iter,j+length(ch_vals)] = biased_out[3]
          SE_mat[iter,j+length(ch_vals)] = biased_out[4]
          
          ######
          # loop - significance level (a)
          ######
          
          for (a in 1:length(CI_vals)){
            if (abs(biased_out[3]-target)<qnorm(0.5*CI_vals[a]/100+0.5)*biased_out[4]){
              #CI[1] = CI[1]+1
              CI_mat[a,j+length(ch_vals)] = CI_mat[a,j+length(ch_vals)]+1
            }
          }
        }
        
        
      } # end loop over different ch_vals
      
    } #end of loop for iterations
    
    #################
    # collect results - for fixed noise parameter
    #################
    
    #updating lists that will make up the result data frame
    params =c(params,rep(paste(param,collapse = ' '),num_of_specs))
    targets =c(targets,rep(paste(target,collapse = ' '),num_of_specs))

    # ave ATE across iterations
    for (j in 1:num_of_specs){
      ATEs = c(ATEs,sum(ATE_mat[,j])/n_iter)
    }
    
    # ave SE across iterations
    for (j in 1:num_of_specs){
      #SEs = c(SEs,sum(SE_mat[,j]^2)^0.5/n_iter)
      SEs = c(SEs,sum(SE_mat[,j])/n_iter) # updated formula
    }
    
    # bias and alg settings
    if (report_bias){
      algs = c(algs,rep(paste("ch =",ch_vals),2))
      biases = c(biases,c(rep(0,length(ch_vals)),rep(1,length(ch_vals))))
    } else {
      algs = c(algs,rep(paste("ch =",ch_vals),1))
      biases = c(biases,rep(0,num_of_specs))
    }
    
    CIs = cbind(CIs,CI_mat)
    
    if(hist){
      ATE_mat_out=ATE_mat # export vector of estimates from last v value
    }
    
  } #end of loop over v values
  
  #################
  # collect results - across v values
  #################
  
  results = data.frame('v' = params,
                       'CATE_v' = targets,
                       'mean_est' = ATEs,
                       'mean_SE' = SEs,
                       'bias'= biases,
                       'alg' = algs)
  
  #add columns for each confidence level tested
  for (a in 1:length(CI_vals)){
    pr=CIs[a,]/n_iter
    results[[paste("CI",CI_vals[a])]] = pr
    results[[paste("CI",CI_vals[a],"SE")]] = sqrt(pr*(1-pr)/n_iter)
  }
  
  if(hist){
    return(list(results,ATE_mat))
  } else {
    return(results)
  }
  
}

