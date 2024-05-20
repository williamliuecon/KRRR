#################
# uncorrupted DGP
#################

sim_local = function(n){
 
  e1=runif(n,-0.5,0.5)
  e2=runif(n,-0.5,0.5)
  e3=runif(n,-0.5,0.5)
  e4=runif(n,-0.5,0.5)
  
  V=e1
  X=cbind(1+2*V+e2,1+2*V+e3,(V-1)^2+e4)
  
  pi=inv.logit(0.5*(V+X[,1]+X[,2]+X[,3]),min=0.05,max=0.95) # propensities away from zero and one
  T=rbinom(n,1,pi)
  
  nu = rnorm(n,0,1/16)
  Y0=matrix(0,n,1)
  Y1=V*X[,1]*X[,2]*X[,3]+nu
  
  Y=Y1*T+Y0*(1-T)
  
  CATE<-function(v){
    out=v*(1+2*v)^2*(v-1)^2
    return(out)
  }
  
  return(list(Y,T,V,X,pi,CATE))
  
}