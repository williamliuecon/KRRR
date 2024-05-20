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

real_data_fn = function(ignored_arg){  # Accepts argument without using it to avoid error message
  data = read.csv("https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/401k.csv")
  n = nrow(data)
  pi = matrix(NA, nrow = n, ncol = 1)
  
  CATE <- function(v){  # Here, is a meaningless function that always returns 0
    return(0)
  }
  
  temp = as.matrix(data["inc"])
  min_Y = quantile(temp, 0.01)
  max_Y = quantile(temp, 0.99)
  data = data[(data$inc >= min_Y) & (data$inc <= max_Y), ]
  
  Y = data[, "net_tfa"]
  T = data[, "e401"]
  V = data[, "age"]
  X = as.matrix(select(data, -c(e401, p401, a401, tw, tfa, net_tfa, tfa_he,
                       hval, hmort, hequity,
                       nifa, net_nifa, net_n401, ira,
                       dum91, icat, ecat, zhat,
                       i1, i2, i3, i4, i5, i6, i7,
                       a1, a2, a3, a4, a5,
                       age)))
  
  Y %<>% as.numeric()  # Variables must all be floats to prevent integer overflow
  T %<>% as.numeric()
  V %<>% as.numeric()
  X = matrix(as.numeric(X), nrow = nrow(X))
  
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
lam_list = list(0.1, 0.05, 0.01)  # List of choices for lambda
nu = 0.001

## With many v_vals, the program crashes when run in parallel. Workaround TBD: loop over individual v_vals.
v_vals = c(25, 30, 35, 40, 45, 50, 55, 60, 65)  # max range is 24-65
sample_sizes = c(dim(X)[1])  # Ignored, vestigial argument
dictionaries = list(b2, b4)  # b2 low-dim, b4 high-dim
gamma_estimators = c(1, 2, 3)  # 1 lasso, 2 rf, 3 nn
ch_vals = c(0.25)  # Hyperparameter used to calculate bandwidth
report_bias = FALSE



sample_sizes_par = rep(sample_sizes, each = length(gamma_estimators)*length(dictionaries))
dictionaries_par = rep(rep(dictionaries, each = length(gamma_estimators)), length(sample_sizes))
gamma_estimators_par = rep(gamma_estimators, length(dictionaries)*length(sample_sizes))

future::plan("multisession", workers = 6)  # Number of CPU cores to use
foptions = list(seed = TRUE)  # Option for parallel-safe RNG
results_list = foreach (n = sample_sizes_par,
                        dict = dictionaries_par,
                        gamma_estimator = gamma_estimators_par,
                        .options.future = foptions) %dofuture% {  # .inorder = TRUE by default
                          dict_size = length(dict(T[1],V[1],X[1,]))
                          print(paste0("sample size: ",n))
                          print(paste0("dictionary: p=",dict_size))
                          print(paste0("CEF: ",gamma_estimator))

                          results = coverage_experiment(data = real_data_fn,
                                                        n = n,
                                                        n_iter = 1,  # n_iter = 1 because not simulation
                                                        v_vals = v_vals,
                                                        ch_vals = ch_vals,
                                                        report_bias = report_bias,
                                                        CI_vals = c(80,95),
                                                        alpha_estimator = 2,
                                                        lam = lam_list,
                                                        nu = nu,
                                                        gamma_estimator = gamma_estimator,
                                                        b = dict)
                        }

i = 1
for(n in sample_sizes){
  for(dict in dictionaries){
    for(gamma_estimator in gamma_estimators){
      dict_size = length(dict(T[1],V[1],X[1,]))

      filename = paste0("results/401k/n", n, "_p", dict_size, "_CEF", gamma_estimator, ".csv")
      print(filename)
      write.csv(results_list[i], filename)
      i = i + 1
    }
  }
}
toc()

# for (n in sample_sizes){
#   print(paste0("sample size: ", n))
# 
#   for (dict in dictionaries){
#     dict_size = length(dict(T[1],V[1],X[1,]))
#     print(paste0("dictionary: p=", dict_size))
# 
#     for (gamma_estimator in gamma_estimators){
#       print(paste0("CEF: ", gamma_estimator))
#       set.seed(1)
#       results = coverage_experiment(data = real_data_fn,
#                                     n = n,
#                                     n_iter = 1,
#                                     v_vals=v_vals,
#                                     ch_vals = ch_vals,
#                                     report_bias = report_bias,
#                                     CI_vals = c(80,95),
#                                     alpha_estimator = 2,
#                                     lam = lam_list,
#                                     nu = nu,
#                                     gamma_estimator = gamma_estimator,
#                                     b=dict)
#       filename = paste0("results/401k/n", n, "_p", dict_size, "_CEF", gamma_estimator, ".csv")
#       print(filename)
#       write.csv(results,filename)
#     }
#   }
# }
# toc()