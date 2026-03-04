library(rxode2)
library(dplyr)
library(randtoolbox)
library(ggplot2)

# Global params

# Simulation model for generating data
tmdd_model_true <- RxODE({
    k_D <- 0.5
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

    d/dt(L) = -k_eL * L - k_on * L * R / Vc + k_off * P
    d/dt(R) = k_syn - k_deg * R - k_on * L * R / Vc + k_off * P
    d/dt(P) = k_on * L * R / Vc - k_off * P - k_eP * P
    DV = log(L / Vc) + add.err
})

time_steps <- seq(0, 24, 3)

event.obs <- et(timeUnits="hr") %>%
  et(time_steps, evid=0, cmt="L")
event.dose_1 <- event.obs %>%
  et(amt=100, ii=12, until=13, cmt="L")
event.dose_2 <- event.obs %>%
  et(amt=10, ii=12, until=13, cmt="L")

generate_training_data <- function(nb_patients_tot) {
  # Separate patients in half
  nb_patients = floor(nb_patients_tot / 2)
  ids_dose1 <- 1:nb_patients
  ids_dose2 <- nb_patients + ids_dose1
  ids <- 1:nb_patients_tot
  id_to_protocol <-
    data.frame(id_all = ids_dose1,
               id = ids_dose1,
               protocol_arm = "arm_1") %>%
    add_row(id_all = ids_dose2,
            id = ids_dose1,
            protocol_arm = "arm_2")

  # Generate Sobol samples for 3 parameters: k_eL, R0
  nb_params = 3
  samples = sobol(nb_patients_tot, nb_params)

  # Define the ranges for the training patient sampling
  low = array(c(-3, -2, -1), dim = c(1, nb_params))
  high = array(c(1, 2, 3), dim = c(1, nb_params))
  low_rep = low[col(samples)]
  high_rep = high[col(samples)]

  scaled_samples = low_rep + (high_rep - low_rep) * samples

  colnames(scaled_samples) = c("mu.k_eL", "mu.R0", "mu.Vc")

  # No added variability
  params = as.data.frame(scaled_samples) %>%
    mutate(
      id = ids,
      add.err = 0,
      eta.k_eL = 0,
      eta.R0 = 0,
      eta.Vc= 0,
    )
  events_full <-
    etRbind(event.dose_1 |> et(id = ids_dose1),
            event.dose_2 |> et(id = ids_dose2), id = "unique")
  sol <-
    rxSolve(
      tmdd_model_true,
      params = params,
      events = events_full,
      returnType = "data.frame",
      atol = 1e-10,
      rtol = 1e-8
    )
  out_data <- sol %>%
    select(id, time, DV, R0, k_eL, Vc) %>%
    rename(value = DV, id_all=id) %>%
    mutate(output_name = "DV") %>%
    left_join(id_to_protocol, by=c("id_all")) %>%
    select(id, time, protocol_arm, output_name, value, R0, k_eL, Vc)
  return(out_data)
}

train_data <- generate_training_data(500)

ggplot(train_data,aes(x=time,y=value, group=id, color=as.factor(id)))+
  geom_line(alpha=0.1)+
  geom_point(alpha=0.1)+
  facet_wrap("protocol_arm") +
  theme(legend.position = "none")

write.csv(x = train_data, file = "qspc26/tmdd_benchmark/data/gp_training.csv", row.names = F)

simulate_model<- function(nb_patients) {
  nb_patients_half = floor(nb_patients / 2)
  ids_dose1 <- 1:nb_patients_half
  ids_dose2 <- nb_patients_half + ids_dose1

  events_full <-
    etRbind(event.dose_1 |> et(id = ids_dose1),
            event.dose_2 |> et(id = ids_dose2), id="unique")

  sol = rxSolve(
    tmdd_model_true,
    events = events_full,
    params = c(
      mu.k_eL = -0.5,
      mu.R0 = 0.0,
      mu.Vc = 1.0
    ),
    omega = lotri(eta.k_eL ~ 0.25, eta.R0 ~ 0.25, eta.Vc ~0.25),
    sigma = lotri(add.err ~ 0.25),
    nSub = nb_patients,
    atol = 1e-10,
    rtol = 1e-8
  )
  return(list(sol=sol))
}

generate_syn_data <- function(nb_patients) {
  res <- simulate_model(nb_patients)
  sol <- res$sol
  event_table <- sol$get.EventTable()
  true_params <- data.frame(id=sol$id, true_k_eL=sol$k_eL, true_R0=sol$R0, true_Vc = sol$Vc) %>%
    unique()
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
    ) %>%
    mutate(protocol_arm = ifelse(id > floor(nb_patients/2), "arm_2", "arm_1")) # not sure this is safe
  out_data <- event_table %>%
    left_join(out_dv, by = c("id", "time", "evid")) %>%
    left_join(true_params, by=c("id"))
  return(out_data)
}


for (nb_patients in c(100,200,300,400,500,1000,2000,5000)) {
  out <- generate_syn_data(nb_patients)
  file = paste0("qspc26/tmdd_benchmark/data/obs_data_", nb_patients, ".csv")
  write.csv(x = out, file = file, row.names = F, quote=F)
}
