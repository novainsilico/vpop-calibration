import pytest
import torch
import pandas as pd

from vpop_calibration import *


def test_analytical_saem(np_rng):
    def logistic_growth(lambda1, lambda2, lambda3, t):
        y = lambda1 / (1 + torch.exp(-(t - lambda2) / lambda3))
        return y

    params = ["lambda1", "lambda2", "lambda3"]
    outputs = ["circ"]
    struct_model = StructuralAnalytical(logistic_growth, params, outputs)

    patient_df = pd.DataFrame({"id": ["tree-1", "tree-2"]})
    df = patient_df.merge(pd.DataFrame({"output_name": outputs}), how="cross")
    time_steps = [0.0, 1.0, 3.0]
    df = df.merge(pd.DataFrame({"time": time_steps}), how="cross")
    df["value"] = np_rng.normal(0, 1)

    init_log_mi = {"lambda2": 0.0, "lambda3": 0.0}
    init_log_pdu = {
        "lambda1": {"mean": 0.0, "sd": 0.5},
    }
    constraints = {"lambda1": {"low": 0.0, "high": 300}}
    covariate_map = None
    init_res_var = [100.0]
    nlme_model = NlmeModel(
        structural_model=struct_model,
        patients_df=patient_df,
        init_log_MI=init_log_mi,
        init_PDU=init_log_pdu,
        covariate_map=covariate_map,
        init_res_var=init_res_var,
        error_model_type="additive",
        num_chains=1,
        constraints=constraints,
    )

    optimizer = PySaem(nlme_model, df)
    optimizer.run()
    nlme_model.compute_ebe()
    nlme_model.sample_conditional_distribution(nb_samples=2)
    plot_individual_map_estimates(nlme_model)
    plot_all_individual_map_estimates(nlme_model)
    plot_map_estimates_gof(nlme_model)
    plot_weighted_residuals(nlme_model, "iwres")
    plot_weighted_residuals(nlme_model, "pwres")
    plot_weighted_residuals(nlme_model, "npde")
