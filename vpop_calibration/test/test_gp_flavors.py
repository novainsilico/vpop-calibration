import pandas as pd
import numpy as np

from vpop_calibration import *

# Create a dummy training data frame
patients = ["patient-01", "patient-02"]
nb_patients = len(patients)
obsIds = ["obs-01", "obs-02"]
protocol_arms = ["arm-A", "arm-B"]
time_steps = np.arange(0, 10.0, 1.0)
patient_descriptors = ["k1", "k2", "k3"]
gp_params = [*patient_descriptors, "time"]
rng = np.random.default_rng()
training_df = pd.DataFrame({"id": patients})
for descriptor in patient_descriptors:
    training_df[descriptor] = rng.normal(0, 1, nb_patients)
training_df = training_df.merge(
    pd.DataFrame({"protocol_arm": protocol_arms}), how="cross"
)
training_df = training_df.merge(pd.DataFrame({"time": time_steps}), how="cross")
training_df = training_df.merge(pd.DataFrame({"output_name": obsIds}), how="cross")
training_df["value"] = rng.normal(0, 1, training_df.shape[0])

implemented_kernels = ["RBF", "SMK", "Deep-RBF"]
implemented_var_strat = ["IMV", "LMCV"]
implemented_mll = ["ELBO", "PLL"]


def gp_init_flavor(var_strat, kernel, mll):
    gp = GP(
        training_df,
        gp_params,
        var_strat=var_strat,
        mll=mll,
        kernel=kernel,
        nb_latents=2,
        nb_features=5,
        num_mixtures=3,
        nb_training_iter=2,
    )
    gp.train()


def test_all_gp_flavors():
    for kernel in implemented_kernels:
        for var_strat in implemented_var_strat:
            for mll in implemented_mll:
                gp_init_flavor(var_strat, kernel, mll)


def test_batching_1():
    gp = GP(training_df, gp_params, nb_training_iter=2)
    gp.train(mini_batching=True, mini_batch_size=8)


def test_batching_2():
    gp = GP(training_df, gp_params, nb_training_iter=2)
    gp.train(mini_batching=True, mini_batch_size=None)


def test_eval_with_valid():
    gp = GP(training_df, gp_params, nb_training_iter=2)
    gp.eval_perf()


def test_eval_no_valid():
    gp = GP(training_df, gp_params, nb_training_iter=2, training_proportion=1)
    gp.eval_perf()
