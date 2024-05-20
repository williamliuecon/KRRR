########
# set up
########

rm(list=ls())
set.seed(2024)

#* R SETUP
library("doFuture")  # used for parallel computing
library("foreach")  # used for parallel computing
library("foreign")
library("dplyr")
library("ggplot2")
library("quantreg")  # used for rq.fit.sfn
library("nnet")	 # used for mulitnom
library("randomForest")
library("keras")
library("mvtnorm")
library("Rlab")
library("gtools")
library("ExtDist")
library("DescTools")
library("tictoc")

#* PYTHON SETUP
library("reticulate")  # used for Python for the Riesz representer

#// Set path for virtual environment
# env_path = paste("mean_embedding_env", sep = "")

#// You can use this to clear the environmental variable "RETICULATE_PYTHON" if it has been set previously, e.g.,
#// by use_python() or use_virtualenv(), or as a result of reticulate detecting a virtual environment directory within
#// the working directory.
#// NOTE: If it keeps getting reset and you find that you are unable to change the Python executable path,
#// reticulate may be trying to use an incorrect lockfile. In this case, delete renv.lock and start over.
# Sys.setenv(RETICULATE_PYTHON = "")

#// Use this to create the virtual env; change the name of the environment or the path to avoid conflicts if necessary.
#//   "Virtual environments are by default located at ~/.virtualenvs (accessed with the virtualenv_root() function).
#//    You can change the default location by defining the WORKON_HOME environment variable."
#// Do not create the virtual environment in a synced (Dropbox, Google Drive, etc.) directory because the resulting file
#// locking can cause the setup process to fail:
#//   "The process cannot access the file because it is being used by another process"
# virtualenv_create(env_path, version = ">=3.9", requirements = "requirements.txt")

#// If setup has failed, you can remove the environment using this:
# virtualenv_remove(env_path, confirm = FALSE)

#// This will be implicitly run once reticulate is loaded:
# use_virtualenv(env_path, required = TRUE)

############
# data
############

#* Binned treatment
# real_data_fn = function(ignored_arg){  # Accepts argument without using it to avoid error message
#   bin_mid_vec = c(145, 375, 625, 875, 1125, 1375, 1625, 1875)
#   
#   load("JCdata.RData")
#   data = JCdata[JCdata$d >= 40 & JCdata$d <= 2000, ]
#   data %<>%
#     mutate(bin_id = cut(d, breaks = c(250, 500, 750, 1000, 1250, 1500, 1750, 2000)),  # Use NA to represent "[40, 250]"
#            bin_id = addNA(bin_id))
#   levels(data$bin_id) = c(levels(data$bin_id)[-8], "[40,250]")  # Recode NA as "[40, 250]"
#   data$bin_id %<>%
#     relevel("[40,250]") %>%
#     as.numeric()
#   data$bin_mid = bin_mid_vec[data$bin_id]
#   data = data.matrix(data)
#   
#   Y = data[, "m"]
#   T = data[, "bin_mid"]
#   V = data[, "age"]
#   X = data[, -c(1, 2, 3, 5, 69, 70)]
#   
#   n = nrow(data)
#   pi = matrix(NA, nrow = n, ncol = 1)
#   
#   CATE <- function(v){  # Here, is a meaningless function that always returns 0
#     return(0)
#   }
# 
#   return(list(Y,T,V,X,pi,CATE))
# }

#* Binary treatment
real_data_fn = function(ignored_arg){  # Accepts argument without using it to avoid error message
  load("JCdata.RData")
  data = mutate(JCdata, d = d > 0)
  data = data.matrix(data)

  n = nrow(data)
  pi = matrix(NA, nrow = n, ncol = 1)

  CATE <- function(v){  # Here, is a meaningless function that always returns 0
    return(0)
  }

  Y = data[, "m"]
  T = data[, "d"]
  V = data[, "age"]
  X = data[, -c(1, 2, 3, 5)]

  return(list(Y,T,V,X,pi,CATE))
}

test=real_data_fn(0)
Y=test[[1]]
T=test[[2]]
V=test[[3]]
X=test[[4]]

###########
# algorithm
###########

source('primitives.R')
source('stage1.R')
source('stage2.R')

##########
# coverage
##########

#alpha_estimator: 0 dantzig, 1 lasso, 2 kernel
#gamma_estimator: 0 dantzig, 1 lasso, 2 rf, 3 nn

source('coverage.R')

#######
# table
#######
tic()
lam_list = list(0.05)  # List of choices for lambda
nu = 0.05
# v_vals=16:24
# sample_sizes = c(dim(X)[1])  # Ignored, vestigial argument
# dictionaries = list(b2, b4)  # Use list(b2,b4) for final results
# gamma_estimators = c(1,2,3)  # lasso, rf, nn
# ch_vals = c(0.25, 0.5, 1.0)  # Hyperparameter used to calculate bandwidth
# report_bias = TRUE

v_vals=c(18, 20, 22)
sample_sizes = c(dim(X)[1])  # Ignored, vestigial argument
dictionaries = list(b2, b4)  # Use list(b2,b4) for final results
gamma_estimators = c(1)  # lasso, rf, nn
ch_vals = c(0.25)
report_bias = FALSE

#* Not parallelized
# for (bin in 1:8){
#   bin_mid_vec = c(145, 375, 625, 875, 1125, 1375, 1625, 1875)
#   bin_mid = bin_mid_vec[bin]
#   
#   m <- function(y, d, v, z, gamma){  # Redefine this from "primitives.R" here as appropriate
#     return(gamma(bin_mid, v, z))
#   }
#   
#   for (n in sample_sizes){
#     print(paste0("sample size: ", n))
#   
#     for (dict in dictionaries){
#       dict_size = length(dict(T[1],V[1],X[1,]))
#       print(paste0("dictionary: p=", dict_size))
#   
#       for (gamma_estimator in gamma_estimators){
#   
#         print(paste0("CEF: ", gamma_estimator))
#         set.seed(1)
#         results = coverage_experiment(data = real_data_fn,
#                                       n = n,
#                                       n_iter = 1,
#                                       v_vals=v_vals,
#                                       ch_vals = ch_vals,
#                                       report_bias = report_bias,
#                                       CI_vals = c(80,95),
#                                       alpha_estimator = 2,
#                                       lam = lam_list,
#                                       nu = nu,
#                                       cfact_val = bin_mid,
#                                       gamma_estimator = gamma_estimator,
#                                       b=dict)
#         filename = paste0("results/bin_",bin,"/n",n,"_p",dict_size,"_CEF",gamma_estimator,".csv")
#         print(filename)
#         write.csv(results,filename)
#       }
#     }
#   }
# }

#* Binned treatment
# sample_sizes_par = rep(sample_sizes, each = length(gamma_estimators)*length(dictionaries))
# dictionaries_par = rep(rep(dictionaries, each = length(gamma_estimators)), length(sample_sizes))
# gamma_estimators_par = rep(gamma_estimators, length(dictionaries)*length(sample_sizes))
# 
# 
# for (bin in 1:8){
#   bin_mid_vec = c(145, 375, 625, 875, 1125, 1375, 1625, 1875)
#   bin_mid = bin_mid_vec[bin]
#   
#   m <- function(y, d, v, z, gamma){  # Redefine this from "primitives.R" here as appropriate
#     return(gamma(bin_mid, v, z))
#   }
# 
#   future::plan("multisession", workers = 6)  # requires 6 CPU cores
#   foptions = list(seed = TRUE)  # Option for parallel-safe RNG
#   results_list = foreach (n = sample_sizes_par,
#                           dict = dictionaries_par,
#                           gamma_estimator = gamma_estimators_par,
#                           .options.future = foptions) %dofuture% {  # .inorder = TRUE by default
#                             dict_size = length(dict(T[1],V[1],X[1,]))
#                             print(paste0("sample size: ",n))
#                             print(paste0("dictionary: p=",dict_size))
#                             print(paste0("CEF: ",gamma_estimator))
#   
#                             results = coverage_experiment(data = real_data_fn,
#                                                           n = n,
#                                                           n_iter = 1,  # n_iter = 1 because not simulation
#                                                           v_vals = v_vals,
#                                                           ch_vals = ch_vals,
#                                                           report_bias = report_bias,
#                                                           CI_vals = c(80,95),
#                                                           alpha_estimator = 2,
#                                                           lam = lam_list,
#                                                           nu = nu,
#                                                           cfact_val = bin_mid,
#                                                           gamma_estimator = gamma_estimator,
#                                                           b = dict)
#                           }
# 
#   i = 1
#   for(n in sample_sizes){
#     for(dict in dictionaries){
#       for(gamma_estimator in gamma_estimators){
#         dict_size = length(dict(T[1],V[1],X[1,]))
#   
#         filename = paste0("results/bin_",bin,"/n",n,"_p",dict_size,"_CEF",gamma_estimator,".csv")
#         print(filename)
#         write.csv(results_list[i], filename)
#         i = i + 1
#       }
#     }
#   }
# }
# toc()


#* Binary treatment
# sample_sizes_par = rep(sample_sizes, each = length(gamma_estimators)*length(dictionaries))
# dictionaries_par = rep(rep(dictionaries, each = length(gamma_estimators)), length(sample_sizes))
# gamma_estimators_par = rep(gamma_estimators, length(dictionaries)*length(sample_sizes))
# 
# future::plan("multisession", workers = 2)  # requires 6 CPU cores  ################
# foptions = list(seed = TRUE)  # Option for parallel-safe RNG
# results_list = foreach (n = sample_sizes_par,
#                         dict = dictionaries_par,
#                         gamma_estimator = gamma_estimators_par,
#                         .options.future = foptions) %dofuture% {  # .inorder = TRUE by default
#                           dict_size = length(dict(T[1],V[1],X[1,]))
#                           print(paste0("sample size: ",n))
#                           print(paste0("dictionary: p=",dict_size))
#                           print(paste0("CEF: ",gamma_estimator))
# 
#                           results = coverage_experiment(data = real_data_fn,
#                                                         n = n,
#                                                         n_iter = 1,  # n_iter = 1 because not simulation
#                                                         v_vals = v_vals,
#                                                         ch_vals = ch_vals,
#                                                         report_bias = report_bias,
#                                                         CI_vals = c(80,95),
#                                                         alpha_estimator = 2,
#                                                         lam = lam_list,
#                                                         nu = nu,
#                                                         gamma_estimator = gamma_estimator,
#                                                         b = dict)
#                         }
# 
# i = 1
# for(n in sample_sizes){
#   for(dict in dictionaries){
#     for(gamma_estimator in gamma_estimators){
#       dict_size = length(dict(T[1],V[1],X[1,]))
# 
#       filename = paste0("results/binary/n", n, "_p", dict_size, "_CEF", gamma_estimator, ".csv")
#       print(filename)
#       write.csv(results_list[i], filename)
#       i = i + 1
#     }
#   }
# }
# toc()

for (n in sample_sizes){
  print(paste0("sample size: ", n))

  for (dict in dictionaries){
    dict_size = length(dict(T[1],V[1],X[1,]))
    print(paste0("dictionary: p=", dict_size))

    for (gamma_estimator in gamma_estimators){

      print(paste0("CEF: ", gamma_estimator))
      set.seed(1)
      results = coverage_experiment(data = real_data_fn,
                                    n = n,
                                    n_iter = 1,
                                    v_vals=v_vals,
                                    ch_vals = ch_vals,
                                    report_bias = report_bias,
                                    CI_vals = c(80,95),
                                    alpha_estimator = 2,
                                    lam = lam_list,
                                    nu = nu,
                                    gamma_estimator = gamma_estimator,
                                    b=dict)
      filename = paste0("results/binary/n", n, "_p", dict_size, "_CEF", gamma_estimator, ".csv")
      print(filename)
      write.csv(results,filename)
    }
  }
}
toc()