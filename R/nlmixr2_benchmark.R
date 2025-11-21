library(nlmixr2)
library(deSolve) # For ode function in simulation

two.compartments <- function() {
  ini({
    log_k12 <- log(0.5); label("k12") 
    log_k21 <- log(0.5); label("k21") 
    log_k_el <- log(0.2); label("k_el") 
    
    # Initial conditions for A1 and A2
    A1_0 <- 4; label("Initial A1 amount")
    A2_0 <- 0; label("Initial A2 amount") 
    
    eta.k12 ~ 0.04
    eta.k21 ~ 0.04 
    eta.k_el ~ 0.04
    add.sd.DVID1 <- 0.1; label("Additive Error DVID1") 
    add.sd.DVID2 <- 0.1; label("Additive Error DVID2")
  })
  model({
    k12 <- exp(log_k12 + eta.k12)
    k21 <- exp(log_k21 + eta.k21)
    k_el <- exp(log_k_el + eta.k_el)
    
    A1(0) = A1_0
    A2(0) = A2_0
    
    d/dt(A1) <- k21 * A2 - k12 * A1 - k_el * A1
    d/dt(A2) <- k12 * A1 - k21 * A2
    
    A1 ~ add(add.sd.DVID1) | A1
    A2 ~ add(add.sd.DVID2) | A2
  })
}

# SIMULATE DATA 
pk_ode_system <- function(t, y, parms) {
  with(as.list(c(y, parms)), {
    dA1dt <- k21 * A2 - k12 * A1 - k_el * A1
    dA2dt <- k12 * A1 - k21 * A2
    list(c(dA1dt, dA2dt))
  })
}
set.seed(42)

# True population parameters
V1 <- 15.0  # volume of compartment 1
V2 <- 50.0
Q <- 10.0  # intercompartmental clearance
true_k_el_pop <- 0.15 # elimination rate of compartment 1

true_k12_pop <- Q / V1 # 0.667
true_k21_pop <- Q / V2 # 0.2

# Variance of eta (log-transformed individual deviations)
true_omega <- matrix(c(0.1^2, 0.0, 0.0,
                       0.0, 0.1^2, 0.0,
                       0.0, 0.0, 0.2^2), nrow = 3, byrow = TRUE)
true_residual_sigma_A1 <- 0.1 
true_residual_sigma_A2 <- 0.1 
num_individuals <- 50
time_span_start <- 0
time_span_end <- 24
nb_steps <- 20 
time_steps <- seq(from = time_span_start, to = time_span_end, length.out = nb_steps)
all_individual_data_list <- list()

# Initial conditions for ODEs 
A1_initial_sim <- 4 
A2_initial_sim <- 0
y0_sim <- c(A1 = A1_initial_sim, A2 = A2_initial_sim)

# Helper function for multivariate normal random numbers (replacement for mvrnorm)
rmvnorm_custom <- function(n = 1, mu, Sigma) {
  p <- ncol(Sigma)
  if (missing(mu)) { mu <- rep(0, p) }
  L <- chol(Sigma)
  Z <- matrix(rnorm(n * p), nrow = p, ncol = n)
  X <- mu + t(L) %*% Z
  return(t(X))
}

etas = rmvnorm_custom(n = num_individuals, mu = c(0, 0, 0), Sigma = true_omega)
for (i in 1:num_individuals) {
  # Simulate individual random effects (eta_i ~ N(0, Omega))
  eta_i <- etas[i,]
  
  # Simulate true individual parameters (log-normal distribution)
  k12_ind <- true_k12_pop * exp(eta_i[1])
  k21_ind <- true_k21_pop * exp(eta_i[2])
  k_el_ind <- true_k_el_pop * exp(eta_i[3])
  
  individual_params_sim <- c(k12 = k12_ind, k21 = k21_ind, k_el = k_el_ind)
  
  # Solve ODEs for true means
  sol_true <- ode(
    y = y0_sim,
    times = time_steps,
    func = pk_ode_system,
    parms = individual_params_sim,
    method = "bdf"
  )
  true_A1 <- sol_true[, "A1"]
  true_A2 <- sol_true[, "A2"]
  
  # Add noise to both A1 and A2 observations
  observed_A1 <- true_A1 + rnorm(length(time_steps), mean = 0, sd = true_residual_sigma_A1)
  observed_A2 <- true_A2 + rnorm(length(time_steps), mean = 0, sd = true_residual_sigma_A2)
  
  # Ensure observations are not negative
  observed_A1[observed_A1 < 0] <- 0
  observed_A2[observed_A2 < 0] <- 0
  
  # Combine A1 and A2 observations for this individual with DVID
  df_A1 <- data.frame(
    ID = i,
    TIME = time_steps,
    DV = observed_A1,
    EVID = 0,  # Observation event
    AMT = NA,  # No dose in this row
    II = NA,
    SS = 0,
    CMT = NA,  
    DVID = 'A1',  # DVID=1 for A1 observations
    MDV = 0   
  )
  df_A2 <- data.frame(
    ID = i,
    TIME = time_steps,
    DV = observed_A2,
    EVID = 0,  # Observation event
    AMT = NA,  # No dose in this row
    II = NA,
    SS = 0,
    CMT = NA,  
    DVID = 'A2',  # DVID=2 for A2 observations
    MDV = 0
  )
  
  all_individual_data_list[[i]] <- rbind(df_A1, df_A2)
}

data <- do.call(rbind, all_individual_data_list) 
# sort data by ID, then TIME, then DVID
data <- data[order(data$ID, data$TIME, data$DVID), ]

data$ytype <- NULL 
data$CMT <- NULL # Ensure CMT is removed if not used or set to NA

# Preview the simulated data structure
head(data)

fit <- nlmixr2(two.compartments, data, est="saem", saemControl(print=0, nmc=3, nBurn=20,nEm=20))
print(fit)
