library(nlmixr2)
library(rxode2)
library(dplyr)
library(microbenchmark)
library(ggplot2)
library(stringr)

# Generic model
tmdd_model <- function() {
  ini({
    mu.k_eL <- 0.0
    mu.R0 <- 0.0
    mu.Vc <- 0.0

    eta.k_eL ~ 0.5
    eta.R0 ~ 0.5
    eta.Vc ~ 0.5

    add.err <- 0.5
  })
  model({
    k_eL <- exp(mu.k_eL + eta.k_eL)
    R0 <- exp(mu.R0 + eta.R0)
    Vc <- exp(mu.Vc + eta.Vc)

    k_on = k_off / k_D
    k_syn = R0 * k_deg

    L(0) = 0
    R(0) = R0
    P(0) = 0

    d/dt(L) = -k_eL * L - k_on * L * R / Vc + k_off * P
    d/dt(R) = k_syn - k_deg * R - k_on * L * R / Vc + k_off * P
    d/dt(P) = k_on * L * R / Vc - k_off * P - k_eP * P

    DV = log(L / Vc)
    DV ~ add(add.err)
  })
}

print(tmdd_model)

fit_to_data <- function(data) {
  options <- saemControl(
    print = 10,
    nBurn = 100,
    nEm = 100,
    nmc = 5,
    nu = c(2, 2, 2),
    rxControl = rxControl(atol=1e-10, rtol=1e-8)
  )
  fit <- nlmixr2(tmdd_model,
                      data,
                      "saem",
                      options,
                      tableControl())
  pop_params <- c(fit$fixef, diag(fit$omega))
  ebe <-
    data.frame(
      id = fit$ID,
      k_eL = fit$k_eL,
      R0 = fit$R0,
      Vc = fit$Vc
    ) %>% unique()
  return(list(pop=pop_params, ebe=ebe))
}

file <- paste0("qspc26/tmdd_benchmark/data/obs_data_200.csv")
data_full <- read.csv(file)
data_one_dose <- data_full %>%
  filter(protocol_arm == "arm_1")

out_full <- fit_to_data(data_full)
out_one_dose <- fit_to_data(data_one_dose)

print(out_full$pop)
print(out_one_dose$pop)

write.csv(x=out_full$ebe,file="./qspc26/tmdd_benchmark/outputs/ebe_nlmixr_2dose.csv",row.names = F,quote=F)
write.csv(x=out_one_dose$ebe,file="./qspc26/tmdd_benchmark/outputs/ebe_nlmixr_1dose.csv",row.names = F,quote=F)
