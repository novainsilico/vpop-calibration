# Plain rxode2 translation of the ODE system in ../nlmixr2_Model/Mavoglurant_convergence.R
# (no ini()/eta -- this is a deterministic forward simulation at fixed parameter values, to
# match what Solver_comparison.ipynb does for the SBML/Simwork backends, not a population fit).
#
# R + rxode2 aren't part of this repo's Nix flake. Run this with, e.g.:
#
#   nix develop --impure --expr \
#     'let flake = builtins.getFlake "nixpkgs"; pkgs = flake.legacyPackages.x86_64-linux; in
#      pkgs.mkShell { buildInputs = [
#        (pkgs.rWrapper.override { packages = with pkgs.rPackages; [ rxode2 dplyr ]; })
#        pkgs.gcc pkgs.gnumake ]; }' \
#     --command Rscript rxode2_sweep.R
#
# rxode2 compiles the model to C at runtime; if that fails with an `as`/libz ABI error, the
# ambient shell's PATH is leaking in a mismatched binutils -- rerun with a clean environment,
# e.g. prefix with `env -i HOME="$HOME" USER="$USER" PATH="/run/current-system/sw/bin:/usr/bin:/bin"`.
#
# Writes rxode2_sweep.csv, read by Solver_comparison.ipynb. Keep the parameter grid below in
# sync with that notebook's BASELINE/FACTORS/WT/DOSE/time_h -- they were written together.

library(rxode2)
library(dplyr)

mod <- rxode2({
  CO  = (187.00*WT^0.81)*60/1000
  QHT = 4.0 *CO/100
  QBR = 12.0*CO/100
  QMU = 17.0*CO/100
  QAD = 5.0 *CO/100
  QSK = 5.0 *CO/100
  QSP = 3.0 *CO/100
  QPA = 1.0 *CO/100
  QLI = 25.5*CO/100
  QST = 1.0 *CO/100
  QGU = 14.0*CO/100
  QHA = QLI - (QSP + QPA + QST + QGU)
  QBO = 5.0 *CO/100
  QKI = 19.0*CO/100
  QRB = CO - (QHT + QBR + QMU + QAD + QSK + QLI + QBO + QKI)
  QLU = QHT + QBR + QMU + QAD + QSK + QLI + QBO + QKI + QRB

  VLU = (0.76 *WT/100)/1.051
  VHT = (0.47 *WT/100)/1.030
  VBR = (2.00 *WT/100)/1.036
  VMU = (40.00*WT/100)/1.041
  VAD = (21.42*WT/100)/0.916
  VSK = (3.71 *WT/100)/1.116
  VSP = (0.26 *WT/100)/1.054
  VPA = (0.14 *WT/100)/1.045
  VLI = (2.57 *WT/100)/1.040
  VST = (0.21 *WT/100)/1.050
  VGU = (1.44 *WT/100)/1.043
  VBO = (14.29*WT/100)/1.990
  VKI = (0.44 *WT/100)/1.050
  VAB = (2.81 *WT/100)/1.040
  VVB = (5.62 *WT/100)/1.040
  VRB = (3.86 *WT/100)/1.040

  BP = 0.61
  fup = 0.028
  fub = fup/BP

  KbLU = exp(0.8334)
  KbHT = exp(1.1205)
  KbSK = exp(-.5238)
  KbSP = exp(0.3224)
  KbPA = exp(0.3224)
  KbLI = exp(1.7604)
  KbST = exp(0.3224)
  KbGU = exp(1.2026)
  KbKI = exp(1.3171)

  S15 = VVB*BP/1000
  C15 = Venous_Blood/S15

  d/dt(Lungs) = QLU*(Venous_Blood/VVB - Lungs/KbLU/VLU)
  d/dt(Heart) = QHT*(Arterial_Blood/VAB - Heart/KbHT/VHT)
  d/dt(Brain) = QBR*(Arterial_Blood/VAB - Brain/KbBR/VBR)
  d/dt(Muscles) = QMU*(Arterial_Blood/VAB - Muscles/KbMU/VMU)
  d/dt(Adipose) = QAD*(Arterial_Blood/VAB - Adipose/KbAD/VAD)
  d/dt(Skin) = QSK*(Arterial_Blood/VAB - Skin/KbSK/VSK)
  d/dt(Spleen) = QSP*(Arterial_Blood/VAB - Spleen/KbSP/VSP)
  d/dt(Pancreas) = QPA*(Arterial_Blood/VAB - Pancreas/KbPA/VPA)
  d/dt(Liver) = QHA*Arterial_Blood/VAB + QSP*Spleen/KbSP/VSP + QPA*Pancreas/KbPA/VPA + QST*Stomach/KbST/VST + QGU*Gut/KbGU/VGU - CLint*fub*Liver/KbLI/VLI - QLI*Liver/KbLI/VLI
  d/dt(Stomach) = QST*(Arterial_Blood/VAB - Stomach/KbST/VST)
  d/dt(Gut) = QGU*(Arterial_Blood/VAB - Gut/KbGU/VGU)
  d/dt(Bones) = QBO*(Arterial_Blood/VAB - Bones/KbBO/VBO)
  d/dt(Kidneys) = QKI*(Arterial_Blood/VAB - Kidneys/KbKI/VKI)
  d/dt(Arterial_Blood) = QLU*(Lungs/KbLU/VLU - Arterial_Blood/VAB)
  d/dt(Venous_Blood) = QHT*Heart/KbHT/VHT + QBR*Brain/KbBR/VBR + QMU*Muscles/KbMU/VMU + QAD*Adipose/KbAD/VAD + QSK*Skin/KbSK/VSK + QLI*Liver/KbLI/VLI + QBO*Bones/KbBO/VBO + QKI*Kidneys/KbKI/VKI + QRB*Rest_of_Body/KbRB/VRB - QLU*Venous_Blood/VVB
  d/dt(Rest_of_Body) = QRB*(Arterial_Blood/VAB - Rest_of_Body/KbRB/VRB)

  logC15 = log(C15)
})

BASELINE <- list(
  KbBR  = exp(1.1),
  CLint = exp(7.6),
  KbAD  = exp(2),
  KbBO  = exp(0.03),
  KbRB  = exp(0.3)
)
KBMU <- exp(0.3)
WT <- 82.1
DOSE <- 50.0
FACTORS <- c(0.5, 1.0, 2.0)

time_h <- seq(0.1, 48, length.out = 60)

combos <- list()
seen <- character(0)
for (pname in names(BASELINE)) {
  for (factor in FACTORS) {
    params <- BASELINE
    params[[pname]] <- BASELINE[[pname]] * factor
    key <- paste(sapply(params, function(x) sprintf("%.10g", x)), collapse = "|")
    if (key %in% seen) next
    seen <- c(seen, key)
    combos[[length(combos) + 1]] <- list(
      id = sprintf("%s_x%g", pname, factor),
      params = params
    )
  }
}
cat(sprintf("%d distinct parameter combinations\n", length(combos)))

ev <- et(amt = DOSE, cmt = "Venous_Blood", time = 0) %>%
  et(time_h)

all_results <- list()
for (combo in combos) {
  p <- combo$params
  full_params <- c(
    KbBR = p$KbBR, KbMU = KBMU, KbAD = p$KbAD, CLint = p$CLint,
    KbBO = p$KbBO, KbRB = p$KbRB, WT = WT
  )
  sol <- rxSolve(
    mod,
    params = full_params,
    events = ev,
    method = "liblsoda",
    atol = 1e-6,
    rtol = 1e-6,
    hini = 1e-6
  )
  df <- as.data.frame(sol)
  df <- df[df$time > 0, c("time", "logC15")]
  # write.csv() only preserves ~15 significant digits for doubles, while Python's
  # np.linspace produces full float64 (~17) -- joining on the raw time_h float later would
  # silently fail for most rows. Carry a positional index instead: rxSolve preserves the
  # requested event-table time order, so row i here is exactly time_h[i].
  df$time_idx <- seq_along(time_h) - 1
  df$id <- combo$id
  all_results[[combo$id]] <- df
}

result <- bind_rows(all_results)
result$concentration <- exp(result$logC15)
result$solver <- "nlmixr2/rxode2"
names(result)[names(result) == "time"] <- "time_h"

write.csv(
  result[, c("id", "time_idx", "time_h", "concentration", "solver")],
  "/home/eliott.tixier/git/vpop-calibration/ACOP26/Mavoglurant/Simplified_Mavoglurant/rxode2_sweep.csv",
  row.names = FALSE
)
cat("wrote", nrow(result), "rows\n")
