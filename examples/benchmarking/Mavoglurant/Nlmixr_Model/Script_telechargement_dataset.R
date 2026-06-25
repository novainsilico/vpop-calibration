dat <- nlmixr2data::mavoglurant
dat$occ = unlist(with(dat, tapply(EVID, ID, function(x) cumsum(x>0))))
dat = subset(dat, occ==1)
dat = subset(dat, EVID>0 | DV>0)
dat$CMT[dat$CMT == 0]  <- 1;
dat$CMT[dat$EVID == 1]  <- "Venous_Blood" ## Compartment dosed to is Venous Blood
dat$CMT[dat$EVID != 1]  <- "C15" ## Observing C15

write.csv(dat,
          file = "~/git/vpop-calibration/examples/benchmarking/Mavoglurant/Mavoglurant_Benchmark_Dataset",
          row.names = FALSE)
