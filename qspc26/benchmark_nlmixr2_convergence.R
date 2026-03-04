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
    eta.Vc ~0.5

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

    d/dt(L) = -k_eL * L - k_on * L * R + k_off * P
    d/dt(R) = k_syn - k_deg * R - k_on * L * R + k_off * P
    d/dt(P) = k_on * L * R - k_off * P - k_eP * P

    DV = log(L / Vc)
    DV ~ add(add.err)
  })
}

print(tmdd_model)

file <- paste0("qspc26/data/obs_data_100.csv")
data <- read.csv(file)
options <- saemControl(
  print = 10,
  nBurn = 100,
  nEm = 100,
  nmc = 3,
  nu = c(5, 1, 1),
  logLik = F
)
fit <-nlmixr2(tmdd_model,
          data,
          est = "saem",
          options,
          tableControl(cwres = FALSE))
print(fit$fixef)
