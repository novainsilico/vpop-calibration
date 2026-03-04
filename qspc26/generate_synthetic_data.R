library(rxode2)
library(dplyr)
library(randtoolbox)
library(ggplot2)

# Global params

# Simulation model for generating data
tmdd_model <- RxODE({
    k_D <- 0.05
    k_deg <- 0.5
    k_off <- 1.0
    k_eP <- 0.2
    # Pop parameters
    R0 = exp(mu.R0 + eta.R0)
    k_eL = exp(mu.k_eL + eta.k_eL)
    Vc = exp(mu.Vc + eta.Vc)

    k_on = k_off / k_D
    k_syn = R0 * k_deg

    L(0) = 0
    R(0) = R0
    P(0) = 0

    d/dt(L) = -k_eL * L - k_on * L * R + k_off * P
    d/dt(R) = k_syn - k_deg * R - k_on * L * R + k_off * P
    d/dt(P) = k_on * L * R - k_off * P - k_eP * P
    DV = log(L / Vc) + add.err
})

time_steps <- seq(0, 50, 4)

events <- et(timeUnits="hr") %>%
  et(amt=100, ii=12, until=30, cmt="L") %>%
  et(time_steps, evid=0, cmt="L")


generate_training_data <- function(nb_patients) {
  ids <- 1:nb_patients

  # Generate Sobol samples for 3 parameters: k_eL, R0, Vc
  samples = sobol(length(ids), 3)

  # Define the ranges for the training patient sampling
  low=array(c(-2,-1,-1), dim=c(1,3))
  high=array(c(1,1,3), dim=c(1,3))
  low_rep = low[col(samples)]
  high_rep = high[col(samples)]

  scaled_samples = low_rep + (high_rep - low_rep) * samples

  colnames(scaled_samples)=c("mu.k_eL", "mu.R0", "mu.Vc")

  # No added variability
  params = as.data.frame(scaled_samples) %>%
    mutate(id=ids, add.err = 0, eta.k_eL = 0, eta.R0 = 0, eta.Vc = 0)
  events_full <- events |> et(id=ids)
  sol <- rxSolve(tmdd_model, params=params, events = events_full, returnType = "data.frame")
  out_data <- sol %>%
    select(id, time, DV, R0, k_eL, Vc) %>%
    rename(value = DV) %>%
    mutate(output_name = "DV", protocol_arm="identity")
  return(out_data)
}

train_data <- generate_training_data(500)

ggplot(train_data,aes(x=time,y=DV, group=id))+
  geom_line(alpha=0.1)+
  geom_point(alpha=0.1)

write.csv(x = train_data, file = "qspc26/data/gp_training.csv", row.names = F)

simulate_model<- function(nb_patients) {
  ids = 1:nb_patients
  sol = rxSolve(
    tmdd_model,
    events = events |> et(id=ids),
    params = c(
      mu.k_eL = -0.6,
      mu.R0 = 0.0,
      mu.Vc = 1.0
    ),
    omega = lotri(eta.k_eL ~ 0.2, eta.R0 ~ 0.15, eta.Vc ~ 0.5),
    sigma = lotri(add.err ~ 0.15),
    nSub = nb_patients
  )
}

generate_syn_data <- function(nb_patients) {
  sol <- simulate_model(nb_patients)
  event_table <- sol$get.EventTable()
  out_dv <-
    data.frame(
      id = sol$id,
      time = sol$time,
      DV = sol$DV,
      evid = 0,
      k_D = unique(sol$k_D),
      k_deg = unique(sol$k_deg),
      k_off = unique(sol$k_off),
      k_eP = unique(sol$k_eP)
    )
  out_data <- event_table %>%
    left_join(out_dv, by = c("id", "time", "evid"))
  return(out_data)
}


for (nb_patients in c(100,200,300,400,500,1000,2000,5000)) {
  out <- generate_syn_data(nb_patients)
  file = paste0("qspc26/data/obs_data_", nb_patients, ".csv")
  write.csv(x = out, file = file, row.names = F)
}
