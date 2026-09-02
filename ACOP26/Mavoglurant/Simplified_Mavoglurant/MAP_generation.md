Instructions for data generation:
This is the standard Mavoglurant model where the initial value of the KbMU parameter has been fixed to exp(0.3) to avoid identfiability issues.
Use the notebooks Sbml_Model/Mavoglurant_convergence.ipynb and Simwork_Model/Mavoglurant_convergence.ipynb with the following priors (make sure to remove KbMU from the inputs as well):

prior_pdu = {
    "model_intrinsic": {
        "KbBR": {"prior": np.exp(1.1)},
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
    lKbAD = 2
    lCLint = 7.6
    lKbBO = 0.03
    lKbRB = 0.3
    eta.LClint ~ 1
    add.err <- 0.5
  })
  model({
    KbBR = exp(lKbBR)
    KbAD = exp(lKbAD)
    CLint= exp(lCLint + eta.LClint)
    KbBO = exp(lKbBO)
    KbRB = exp(lKbRB)
    KbMU = exp(0.3)