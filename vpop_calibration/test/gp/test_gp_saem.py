import pandas as pd

from vpop_calibration import *


def test_gp_saem(np_rng):
    time_steps = pd.DataFrame({"time": [0.0, 1.0]})
    params = ["k1", "k2", "k3", "time"]
    patients_training = pd.DataFrame(
        {"id": ["p1", "p2"], "k1": [1.0, 2.0], "k2": [3.0, 4.0], "k3": [5.0, 6.0]}
    )
    outputs = pd.DataFrame({"output_name": ["s1", "s2"]})
    protocol_arms = pd.DataFrame({"protocol_arm": ["arm-A", "arm-B"]})

    training_df = (
        patients_training.merge(outputs, how="cross")
        .merge(protocol_arms, how="cross")
        .merge(time_steps, how="cross")
    )
    training_df["value"] = np_rng.normal()

    gp = GP(training_df, params)

    patient_df = pd.DataFrame(
        {"id": ["patient-1", "patient-2"], "protocol_arm": ["arm-B", "arm-A"]}
    )
    obs_df = patient_df.merge(outputs, how="cross").merge(time_steps, how="cross")
    obs_df["value"] = np_rng.normal()
    obs_df = obs_df.sample(frac=0.7)

    init_mi = {"k1": 0.0}
    init_pdu = {"k2": {"mean": 0.0, "sd": 0.1}, "k3": {"mean": 0.0, "sd": 0.1}}
    init_res_var = [0.5, 0.5]

    struct_model = StructuralGp(gp)

    nlme_model = NlmeModel(
        struct_model,
        patient_df,
        init_mi,
        init_pdu,
        init_res_var,
        None,
    )
    optimizer = PySaem(nlme_model, obs_df)
    optimizer.run()
    check_surrogate_validity_gp(nlme_model)
    plot_map_estimates(nlme_model)
    plot_weighted_residuals(nlme_model, "iwres")
    plot_weighted_residuals(nlme_model, "pwres")
    plot_weighted_residuals(nlme_model, "npde")
