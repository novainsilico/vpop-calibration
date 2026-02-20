library(nlmixr2)
library(rxode2)

tmdd_model <- function() {
  ini({
    k_D <- fix(1.0)
    k_deg <- fix(1.0)
    inj <- fix(10.0)
    k_off <- fix(1.0)

    log_k_eL <- log(5e-2)
    log_k_eP <- log(1e-1)
    log_R0 <- log(3.0)
    log_Vc <- log(3.0)

    eta.k_eL ~ 0.25
    eta.k_eP ~ 0.2
    eta.R0 ~ 0.1
    eta.Vc ~ 0.1

    prop.err <- 0.05
  })
  model({
    k_eL = exp(log_k_eL + eta.k_eL)
    k_eP = exp(log_k_eP + eta.k_eP)
    R0 = exp(log_R0 + eta.R0)
    Vc = exp(log_Vc + eta.Vc)

    k_on = k_off / k_D
    k_syn = R0 * k_deg

    L(0) = inj / Vc
    R(0) = R0
    P(0) = 0

    d/dt(L) = - k_eL * L - k_on * L * R + k_off * P
    d/dt(R) = k_syn - k_deg * R - k_on * L * R + k_off * P
    d/dt(P) = k_on * L * R - k_off * P - k_eP * P

    L ~ prop(prop.err)
  })
}

data <- read.csv("tmdd_synthetic_data_nlmixr2.csv")

fit <- nlmixr2(tmdd_model, data, est = "saem", saemControl(print=50, nBurn=200, nEm=300), tableControl(cwres = TRUE))

vpcPlot(fit, n=500, show=list(L=TRUE), log_y = TRUE)
