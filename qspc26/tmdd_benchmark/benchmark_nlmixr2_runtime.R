library(nlmixr2)
library(rxode2)
library(dplyr)
library(microbenchmark)
library(ggplot2)
library(stringr)

# Generic model
tmdd_model <- function() {
  ini({
    mu.k_eL <- 1.0
    mu.R0 <- 1.0
    mu.Vc <- 1.0

    eta.k_eL ~ 0.1
    eta.R0 ~ 0.1
    eta.Vc ~0.1

    add.err <- 0.1
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

fit_nlmixr <- function(nb_patients) {
  file <- paste0("qspc26/tmdd_benchmark/data/obs_data_",nb_patients,".csv")
  data <- read.csv(file)
  options <- saemControl(
    print = 0,
    nBurn = 100,
    nEm = 100,
    nmc = 1,
    nu = c(1, 1, 1),
    logLik = F
  )
  fit <-
    nlmixr2(tmdd_model,
            data,
            est = "saem",
            options)
  return(fit)
}

# benchmark the execution time
# Exclude 5k as it breaks Rstudio
nb_patients_list <- c(100,200,300,400,500,1000,2000,5000)

tests <- lapply(nb_patients_list, function(nb_patients) {bquote(fit_nlmixr(.(nb_patients)))})
res <- microbenchmark(list=tests, times = 5)
res_inc <- res %>%
  mutate(nb_patients = as.numeric(str_extract_all(expr,"\\d+")), time = time / 1e9) %>%
  select(!expr)
ggplot(res_inc, aes(y=time, x=nb_patients, group=nb_patients)) + geom_boxplot(width=50)

write.csv(x=res_inc, file="qspc26/tmdd_benchmark/performance_nlmixr.csv", row.names = F)
