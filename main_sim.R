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
# simulation
############

source('simulate.R')

test=sim_local(100)
CATE=test[[6]]
CATE(0)
v_vals=seq(from=-0.5,to=.5,length.out=101)
out=CATE(v_vals)
plot(v_vals,out)

n=100
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
lam_list = list(0.01)  # List of choices for lambda
nu = 0                 # Atheoretical ridge parameter for numerical purposes
n_iter = 100           # 500 for final

v_vals = seq(from = -0.25, to = .25, length.out = 3)
sample_sizes = c(100)
dictionaries = list(b2, b4)  # Use list(b2,b4) for final results
gamma_estimators = c(1,2,3)  # lasso, rf, nn



#* Parallelized
sample_sizes_par = rep(sample_sizes, each = length(gamma_estimators)*length(dictionaries))
dictionaries_par = rep(rep(dictionaries, each = length(gamma_estimators)), length(sample_sizes))
gamma_estimators_par = rep(gamma_estimators, length(dictionaries)*length(sample_sizes))

future::plan("multisession", workers = 6)  # requires 6 CPU cores
foptions = list(seed = TRUE)  # Option for parallel-safe RNG
results_list = foreach (n = sample_sizes_par,
                        dict = dictionaries_par,
                        gamma_estimator = gamma_estimators_par,
                        .options.future = foptions) %dofuture% {  # .inorder = TRUE by default
  dict_size = length(dict(T[1],V[1],X[1,]))
  print(paste0("sample size: ",n))
  print(paste0("dictionary: p=",dict_size))
  print(paste0("CEF: ",gamma_estimator))

  results = coverage_experiment(data = sim_local,
                                n = n,
                                n_iter = n_iter,
                                v_vals = v_vals,
                                ch_vals = c(0.25, 0.5, 1.0),
                                report_bias = TRUE,
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

      filename = paste0("results/simulation/n", n, "_p", dict_size, "_CEF", gamma_estimator, ".csv")
      print(filename)
      write.csv(results_list[i], filename)
      i = i + 1
    }
  }
}
toc()

#* Not parallelized
# for(n in sample_sizes){
#   print(paste0("sample size: ",n))
# 
#   for(dict in dictionaries){
#     dict_size=length(dict(T[1],V[1],X[1,]))
#     print(paste0("dictionary: p=",dict_size))
# 
#     for(gamma_estimator in gamma_estimators){
# 
#       print(paste0("CEF: ",gamma_estimator))
#       set.seed(1)
#       results = coverage_experiment(data = sim_local,
#                                     n = n,
#                                     n_iter = n_iter,
#                                     v_vals=v_vals,
#                                     ch_vals = c(0.25,0.5,1.0),
#                                     report_bias = TRUE,
#                                     CI_vals = c(80,95),
#                                     alpha_estimator = 2,
#                                     lam = lam_list,
#                                     nu = nu,
#                                     gamma_estimator = gamma_estimator,
#                                     b=dict)
#       filename=paste0("results/simulation/n",n,"_p",dict_size,"_CEF",gamma_estimator,".csv")
#       print(filename)
#       write.csv(results,filename)
#     }
#   }
# }
# toc()
