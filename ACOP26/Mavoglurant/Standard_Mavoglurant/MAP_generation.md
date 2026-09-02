Instructions for data generation:

Use the notebooks Sbml_Model/Mavoglurant_convergence.ipynb and Simwork_Model/Mavoglurant_convergence.ipynb with the following priors:

prior_pdu = {
    "model_intrinsic": {
        "KbBR": {"prior": np.exp(1.1)},
        "KbMU": {"prior": np.exp(0.3)},
        "KbBO": {"prior": np.exp(0.03)},
        "KbAD": {"prior": np.exp(2)},
        "KbRB": {"prior": np.exp(0.3)},   
    }
    "pdu": {
        "CLint": {"prior": np.exp(7.6), "prior_omega": 1},
    },
    "pdk": {"WT", "dose"},
    "error_model": {
        "logC15": {"error_type": "additive", "sigma": 0.5},
    },
}

For nlmixr2 use Mavoglurant_convergence.R with the following parameters:

  ini({
    ##theta=exp(c(1.1, .3, 2, 7.6, .003, .3))
    lKbBR = 1.1
    lKbMU = 0.3
    lKbAD = 2
    lCLint = 7.6
    lKbBO = 0.03
    lKbRB = 0.3
    eta.LClint ~ 1
    add.err <- 0.5
  })
  model({
    KbBR = exp(lKbBR)
    KbMU = exp(lKbMU)
    KbAD = exp(lKbAD)
    CLint= exp(lCLint + eta.LClint)
    KbBO = exp(lKbBO)
    KbRB = exp(lKbRB)