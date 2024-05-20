R and Python code for "Kernel Ridge Riesz Representers: $L_2$ Rates and Mis-specification" (Singh, 2024).

### SETUP
1. Make sure that the "renv" package is installed.
2. Use the command "renv::activate()" (or "renv::restore()") and then select "1" to "activate" the project, which means allowing renv to be used for the project.
3. Use the command "renv::restore()" and then select "y" to create the R and Python virtual environments. (The Python virtual environment is named "mean_embedding_env". Change the name of this environment in the file "renv.lock" if this name is already used for a different environment.)

### ESTIMATION
main_sim.R creates the tables for the simulation experiment, and was run in parallel for the final results.
main_real_v2.R creates the tables for the 401(k) data, and was run in serial for the final results.
(Running them differently can result in slightly different results due to RNG.)

The tables are saved as .csv files in the directories "results/simulation" and "results/401k".

Adjust the code parameters as appropriate, and place the .csv files into the correspondingly named subdirectories. The code parameters are:
1. lam_list – a list of lambdas to choose from using LOOCV.
   * lam_list=c(0.05,0.01,0.005) was used for the final simulation results.
   * lam_list=c(0.1,0.05,0.01) was used for the final 401(k) results.
2. nu – a atheoretical ridge parameter for numerical purposes.
   * nu=0 was used for the final simulation results.
   * nu=0.001 was used for the final 401(k) results.
3. n_iter – the number of iterations (Monte Carlo draws) for the simulation.
   * n_iter=500 was used for the final simulation results.
