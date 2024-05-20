######
# norm
######

two.norm <- function(x){
  return(sqrt(x %*% x))
} 

############
# dictionary
############

b0 <- function(d,v,z){
  return(c(d,v,z))
}

b1<-function(d,v,z){
  return(c(1,d,v,z))
}

b2<-function(d,v,z){
  return(c(1,d,v,z,d*v,d*z,v*z))
}

b3<-function(d,v,z){
  return(c(1,d,v,z,
           d*v,d*z,v*z,v^2,z^2,
           d*v*z,d*v^2,d*z^2,v*z^2,v^2*z,v^3,z^3))
}

b4<-function(d,v,z){
  return(c(1,d,v,z,
           d*v,d*z,v*z,v^2,z^2,
           d*v*z,d*v^2,d*z^2,v*z^2,v^2*z,v^3,z^3,
           d*v*z^2,d*v^2*z,d*v^3,d*z^3,v*z^3,v^2*z^2,v^3*z,v^4,z^4))
}

b5<-function(d,v,z){
  return(c(1,d,v,z,
           d*v,d*z,v*z,v^2,z^2,
           d*v*z,d*v^2,d*z^2,v*z^2,v^2*z,v^3,z^3,
           d*v*z^2,d*v^2*z,d*v^3,d*z^3,v*z^3,v^2*z^2,v^3*z,v^4,z^4,
           d*v^4,d*v^3*z,d*v^2*z^2,d*v*z^3,d*z^4,v^5,v^4*z,v^3*z^2,v^2*z^3,v*z^4,z^5))
}

############
# functional
############

m<-function(y,d,v,z,gamma){ #all data arguments to make interchangeable with m2
  return(gamma(1,v,z))
}

m2<-function(y,d,v,z,gamma){
  return(y*gamma(d,v,z))
}

###########
# influence
###########

psi_tilde<-function(y,d,v,z,m,alpha,gamma,ell){
  
  #alpha_avg <<- c((alpha_avg[2]*alpha_avg[1]+alpha(d,z))/(alpha_avg[2]+1),alpha_avg[2]+1)
  return(ell(v)*m(y,d,v,z,gamma)+ell(v)*alpha(d,v,z)*(y-gamma(d,v,z)))
}

psi_tilde_bias<-function(y,d,v,z,m,alpha,gamma,ell){
  return(ell(v)*m(y,d,v,z,gamma))
}

#################
# sufficient stat
#################

get_MNG<-function(Y,T,V,X,b){
  
  p=length(b(T[1],V[1],X[1,]))
  n.nl=length(T)
  
  B=matrix(0,n.nl,p)
  M=matrix(0,p,n.nl)
  N=matrix(0,p,n.nl)
  
  for (i in 1:n.nl){
    B[i,]=b(T[i],V[i],X[i,])
    M[,i]=m(Y[i],T[i],V[i],X[i,],b)
    N[,i]=m2(Y[i],T[i],V[i],X[i,],b)  # this is a more general formulation for N
  }
  
  M_hat=rowMeans(M)
  N_hat=rowMeans(N)
  G_hat=t(B)%*%B/n.nl
  
  return(list(M_hat,N_hat,G_hat,B))
}

